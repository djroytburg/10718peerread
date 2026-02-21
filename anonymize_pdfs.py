#!/usr/bin/env python3
"""NER-enhanced anonymization using spaCy for robust entity detection.

Combines:
1. Structural filtering (Title -> Abstract boundary)
2. NER-based PERSON/ORG entity detection
3. Acknowledgements section removal
"""

import argparse
import json
import re
from pathlib import Path
from copy import deepcopy

try:
    import spacy
except ImportError:
    print("ERROR: spaCy not installed. Run: pip install spacy")
    print("Then download model: python -m spacy download en_core_web_sm")
    exit(1)


def load_nlp():
    """Load spaCy NER model."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("ERROR: spaCy model not found. Run: python -m spacy download en_core_web_sm")
        exit(1)
    return nlp


def find_abstract_index(texts):
    """Find where Abstract section header is."""
    for i, item in enumerate(texts):
        if item.get('label') == 'section_header':
            text = item.get('text', '').strip().lower()
            if text == 'abstract':
                return i
    return None


def find_acknowledgements_range(texts):
    """Find acknowledgements section (start to References)."""
    start_idx = None

    for i, item in enumerate(texts):
        label = item.get('label', '')
        text = item.get('text', '').strip().lower()

        if label == 'section_header':
            if 'acknowledgement' in text or 'acknowledgment' in text:
                start_idx = i
            elif start_idx is not None and 'reference' in text:
                return (start_idx, i)

    return None


def contains_author_entities(text, nlp, cache=None):
    """Check if text contains PERSON or ORG entities using NER.

    Args:
        text: Text to analyze
        nlp: spaCy NLP model
        cache: Optional dict to cache results (for performance)

    Returns:
        (bool, list of entity texts) - True if PERSON/ORG found
    """
    if not text or len(text) < 3:
        return False, []

    # Check cache
    if cache is not None and text in cache:
        return cache[text]

    # Run NER
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        # Look for person names and organizations
        if ent.label_ in ('PERSON', 'ORG'):
            entities.append((ent.text, ent.label_))

    result = (len(entities) > 0, entities)

    # Cache result
    if cache is not None:
        cache[text] = result

    return result


def is_email_text(text):
    """Detect if text contains an email address."""
    return bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))


def is_author_footnote_marker(text):
    """Detect orphaned footnote markers like '∗', '†', '1', '2'."""
    text_stripped = text.strip()
    # Single character that's a symbol or digit
    if len(text_stripped) == 1 and text_stripped in '∗†‡§¶#123456789':
        return True
    # Just numbers and spaces
    if re.match(r'^[0-9,\s]+$', text_stripped) and len(text_stripped) < 20:
        return True
    return False


def is_author_related_footnote(text):
    """Check if footnote text contains author-related content."""
    text_lower = text.lower()

    # Keywords that appear in author footnotes
    author_keywords = [
        'corresponding author', 'equal contribution', 'author',
        'work was done', 'these authors', 'affiliation',
        'university', 'institute', 'school of', 'department',
        'email', '@', 'contributed equally'
    ]

    return any(kw in text_lower for kw in author_keywords)


def should_remove_element(idx, item, abstract_idx, ack_range, nlp, ner_cache):
    """Decide if a text element should be removed.

    Uses hybrid approach:
    1. Structural: Remove between title and Abstract
    2. NER: Detect PERSON/ORG entities in front matter
    3. Pattern: Emails, footnote markers, author footnotes

    Args:
        idx: Index in texts array
        item: Text item dict
        abstract_idx: Index where Abstract starts (None if not found)
        ack_range: Tuple (start, end) for acknowledgements
        nlp: spaCy NER model
        ner_cache: Cache for NER results
    """
    text = item.get('text', '').strip()
    label = item.get('label', '')

    # Keep title (first element)
    if idx == 0:
        return False

    # REMOVE acknowledgements section
    if ack_range:
        start, end = ack_range
        if start <= idx < end:
            return True

    # Front matter: before Abstract
    if abstract_idx is not None and idx < abstract_idx:
        # Keep Abstract header itself
        if label == 'section_header' and text.lower() == 'abstract':
            return False

        # Remove everything else in front matter
        # This is the aggressive structural approach
        return True

    # After Abstract: use NER to catch any leaked entities
    # This catches authors mentioned in acknowledgements or misplaced content
    if abstract_idx is not None and idx >= abstract_idx:
        # Skip very long paragraphs (likely actual content, not author info)
        if len(text) > 500:
            return False

        # Check for emails (high confidence author info)
        if is_email_text(text):
            return True

        # Check for orphaned footnote markers
        if is_author_footnote_marker(text):
            return True

        # AGGRESSIVE: Remove ALL footnotes in first 30 elements
        # Most papers have author footnotes appearing shortly after Abstract
        if label == 'footnote' and idx < 30:
            # But keep footnotes that are clearly scientific (citations, equations)
            if not any(indicator in text for indicator in ['[', ']', 'http', 'https', 'See ']):
                return True

        # Check for author-related footnote content anywhere in front section
        if label == 'footnote' and idx < 50:
            if is_author_related_footnote(text):
                return True

        # ALSO check 'text' elements in early positions for author-related content
        # Sometimes author footnotes are mislabeled as 'text'
        if label == 'text' and idx < 30 and len(text) < 100:
            if is_author_related_footnote(text):
                return True

        # Pattern-based fallback for names that NER misses
        # Catch isolated names shortly after Abstract that NER doesn't detect
        # Use relative position from Abstract (not absolute index)
        if label == 'text' and idx < abstract_idx + 20 and len(text) < 50:
            # Pattern: Two or three capitalized words with optional numbers/symbols
            # e.g., "Sylvain Le Corff 1", "Rupak Majumdar"
            name_pattern = r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}(?:\s+[0-9∗†‡,]+)?$'
            if re.match(name_pattern, text.strip()):
                # Filter out common false positives
                false_positives = [
                    'find photo', 'training stage', 'sequence length',
                    'diagnostic tools', 'memory phase', 'original prompt'
                ]
                if text.lower() not in false_positives:
                    return True

        # Use NER for short suspicious text near Abstract
        # Only check within first ~30 elements after Abstract to avoid false positives
        if idx < abstract_idx + 30 and len(text) < 200:
            has_entities, entities = contains_author_entities(text, nlp, ner_cache)
            if has_entities:
                # Additional validation: check if it's likely author info
                # vs. just mentioning a person/org in the abstract

                # If it's a footnote and has entities, remove it
                if label == 'footnote':
                    return True

                # If the entire text is just the entity (name or org), remove it
                entity_texts = [e[0] for e in entities]
                combined_entities = ' '.join(entity_texts)
                if len(combined_entities) / max(len(text), 1) > 0.5:
                    # More than 50% of text is entity names
                    return True

    return False


def anonymize_json(doc, nlp):
    """Anonymize a parsed PDF JSON using NER + structural analysis."""
    doc_anon = deepcopy(doc)
    texts = doc_anon.get('texts', [])

    # Find key section boundaries
    abstract_idx = find_abstract_index(texts)
    ack_range = find_acknowledgements_range(texts)

    print(f"  Abstract at: {abstract_idx}, Acknowledgements: {ack_range}")

    if abstract_idx is None:
        print("  WARNING: No Abstract found - may have incomplete anonymization")
        abstract_idx = 0

    # Filter texts with NER
    anonymized_texts = []
    removed_count = 0
    ner_cache = {}  # Cache NER results for performance

    for i, item in enumerate(texts):
        if should_remove_element(i, item, abstract_idx, ack_range, nlp, ner_cache):
            removed_count += 1
        else:
            anonymized_texts.append(item)

    doc_anon['texts'] = anonymized_texts

    # Clean up groups
    if 'groups' in doc_anon:
        doc_anon['groups'] = [g for g in doc_anon['groups']
                              if g.get('label') != 'key_value_area']

    return doc_anon, removed_count


def main():
    parser = argparse.ArgumentParser(
        description="NER-enhanced PDF anonymization using spaCy"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with .pdf.json files")
    parser.add_argument("--output-dir",
                        help="Output directory (default: input-dir + '_anonymized_v4')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output, just show what would be removed")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files (for testing)")
    args = parser.parse_args()

    # Load NER model
    print("Loading spaCy NER model...")
    nlp = load_nlp()
    print(f"Loaded: {nlp.meta['name']} v{nlp.meta['version']}\n")

    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(str(input_dir) + '_anonymized_v4')

    json_files = sorted(input_dir.glob("*.pdf.json"))
    if args.limit:
        json_files = json_files[:args.limit]

    print(f"Found {len(json_files)} files")
    if not args.dry_run:
        output_dir.mkdir(exist_ok=True)
        print(f"Output: {output_dir}\n")
    else:
        print("DRY RUN\n")

    total_removed = 0

    for i, filepath in enumerate(json_files, 1):
        with open(filepath) as f:
            doc = json.load(f)

        print(f"[{i}/{len(json_files)}] {filepath.name}")
        doc_anon, removed = anonymize_json(doc, nlp)
        total_removed += removed
        print(f"  Removed: {removed}")

        if not args.dry_run:
            with open(output_dir / filepath.name, 'w') as f:
                json.dump(doc_anon, f, ensure_ascii=False, indent=2)

        if i % 25 == 0:
            print()  # Blank line every 25 files

    print(f"\nTotal removed: {total_removed}")
    if not args.dry_run:
        print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
