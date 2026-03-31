#!/usr/bin/env python3
"""Batch anonymize parsed PDFs using spaCy NER (v4 approach).

Processes parsed Docling JSONs in parallel using multiprocessing.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Load spaCy model (shared across workers via fork)
nlp = spacy.load("en_core_web_sm")


def find_acknowledgements_range(texts):
    """Find the index range of the Acknowledgements section."""
    ack_start = None
    for i, item in enumerate(texts):
        text_lower = item.get('text', '').lower().strip()
        if item.get('label') == 'section_header' and 'acknowledge' in text_lower:
            ack_start = i
            break

    if ack_start is None:
        return None

    # Find end: next section header or end of document
    for i in range(ack_start + 1, len(texts)):
        if texts[i].get('label') == 'section_header':
            return (ack_start, i)

    return (ack_start, len(texts))


def get_ner_entities(text, nlp, ner_cache):
    """Get named entities using spaCy NER with caching."""
    if text in ner_cache:
        return ner_cache[text]

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    ner_cache[text] = entities
    return entities


def should_remove_element(idx, item, abstract_idx, ack_range, nlp, ner_cache):
    """Determine if an element should be removed (v4 logic)."""
    label = item.get('label', 'text')
    text = item.get('text', '').strip()

    # Keep title (first element)
    if idx == 0:
        return False

    # REMOVE acknowledgements section
    if ack_range:
        start, end = ack_range
        if start <= idx < end:
            return True

    # Front matter: everything before Abstract
    if abstract_idx is not None and idx < abstract_idx:
        # Keep the Abstract header itself
        if label == 'section_header' and text.lower() == 'abstract':
            return False
        # Remove everything else before abstract
        return True

    # Email detection in any early label (footnotes, headers, etc)
    if abstract_idx is not None and idx < abstract_idx + 30:
        if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text):
            return True

    # Early content check (within 20 elements after abstract)
    if abstract_idx is not None and idx < abstract_idx + 20:
        # Check for author/affiliation patterns
        if label == 'text' and len(text) < 100:
            # Email pattern
            if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text):
                return True

            # Affiliation indicators
            affiliation_keywords = ['university', 'institute', 'department', 'college',
                                   'laboratory', 'research', 'school of']
            if any(keyword in text.lower() for keyword in affiliation_keywords):
                return True

            # Footnote symbols (affiliations)
            if re.search(r'^[0-9∗†‡§¶]+\s*[A-Z]', text):
                return True

    # Email detection in main text (also catches some missed cases)
    if label in ['text', 'footnote', 'page_footer']:
        if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text):
            # Only remove if email is substantial or is the whole element
            if len(text) < 200 or len(re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)) > 0:
                return True

    # NER-based detection for person names
    if label == 'text':
        entities = get_ner_entities(text, nlp, ner_cache)
        for ent_text, ent_label in entities:
            if ent_label == 'PERSON':
                # Only remove if it's a substantial part of the text
                if len(ent_text) / max(len(text), 1) > 0.3:
                    return True

    # Author attribution patterns
    if abstract_idx is not None and idx < abstract_idx + 30:
        attr_keywords = ['correspondence', 'corresponding author', 'author contributions',
                        'author notes', 'author information', 'contact:', 'email:']
        if any(kw in text.lower() for kw in attr_keywords):
            return True

    # Pattern-based fallback for isolated names (RELATIVE indexing)
    if label == 'text' and abstract_idx is not None and idx < abstract_idx + 20:
        if len(text) < 50:
            # Name pattern: "Firstname Lastname" or "Firstname Middlename Lastname"
            # Possibly followed by affiliation numbers/symbols
            name_pattern = r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}(?:\s+[0-9∗†‡,]+)?$'
            if re.match(name_pattern, text.strip()):
                # Filter false positives
                false_positives = [
                    'find photo', 'training stage', 'sequence length',
                    'diagnostic tools', 'memory phase', 'original prompt',
                    'capture model', 'prior work', 'feature extraction'
                ]
                if text.lower() not in false_positives:
                    return True

    return False


def anonymize_paper(paper_id, input_dir, output_dir):
    """Anonymize a single parsed PDF."""
    input_path = os.path.join(input_dir, f"{paper_id}.pdf.json")
    output_path = os.path.join(output_dir, f"{paper_id}.pdf.json")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        texts = doc.get('texts', [])
        if not texts:
            log.warning(f"{paper_id}: No texts found")
            return False

        # Find Abstract position
        abstract_idx = None
        for i, item in enumerate(texts):
            if item.get('label') == 'section_header':
                if item.get('text', '').lower().strip() == 'abstract':
                    abstract_idx = i
                    break

        # Find Acknowledgements range
        ack_range = find_acknowledgements_range(texts)

        # Filter texts using v4 logic
        ner_cache = {}
        filtered_texts = []
        removed_count = 0

        for idx, item in enumerate(texts):
            if should_remove_element(idx, item, abstract_idx, ack_range, nlp, ner_cache):
                removed_count += 1
            else:
                filtered_texts.append(item)

        # Update document
        doc['texts'] = filtered_texts

        # Write anonymized version
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        log.info(f"{paper_id}: Removed {removed_count}/{len(texts)} elements")
        return True

    except Exception as e:
        log.error(f"{paper_id}: Failed - {e}")
        return False


def process_paper(args):
    """Wrapper for multiprocessing."""
    paper_id, input_dir, output_dir = args
    return anonymize_paper(paper_id, input_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Batch anonymize parsed PDFs")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing parsed PDF JSONs")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for anonymized outputs")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of parallel workers")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip papers that are already anonymized")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of papers to process
    paper_files = sorted([f for f in os.listdir(args.input_dir)
                         if f.endswith('.pdf.json')])
    paper_ids = [f.replace('.pdf.json', '') for f in paper_files]

    # Filter out already processed if requested
    if args.skip_existing:
        paper_ids = [pid for pid in paper_ids
                    if not os.path.exists(os.path.join(args.output_dir, f"{pid}.pdf.json"))]

    log.info(f"Found {len(paper_ids)} papers to anonymize")
    log.info(f"Using {args.workers} workers")

    # Process in parallel
    process_args = [(pid, args.input_dir, args.output_dir) for pid in paper_ids]

    with Pool(args.workers) as pool:
        results = pool.map(process_paper, process_args)

    success_count = sum(1 for r in results if r)
    fail_count = len(results) - success_count

    log.info("=" * 60)
    log.info(f"Anonymization complete!")
    log.info(f"  Success: {success_count}")
    log.info(f"  Failed: {fail_count}")
    log.info(f"  Output: {args.output_dir}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
