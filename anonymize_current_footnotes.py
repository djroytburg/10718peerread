#!/usr/bin/env python3
"""
Remove author emails and attribution from footnotes in already-anonymized PDFs.
Lightweight fix for email leaks in footnote sections.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def should_remove_footnote(text):
    """Check if a footnote contains author/email info that should be removed."""
    if not text:
        return False

    # Email pattern
    if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text):
        return True

    # Author attribution keywords
    attribution_keywords = [
        'correspondence', 'corresponding author', 'author contributions',
        'author notes', 'author information', 'contact:', 'email:',
        'equal contribution', 'contributed equally'
    ]
    if any(kw in text.lower() for kw in attribution_keywords):
        return True

    # Author name patterns in early footnotes
    # "* Authors listed" or "† Emails:" etc
    if re.match(r'^[*†‡§¶]+\s+(authors|emails?|correspondence)', text, re.IGNORECASE):
        return True

    return False


def fix_footnotes(json_path):
    """Remove author-identifying footnotes from an anonymized PDF."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        texts = doc.get('texts', [])
        if not texts:
            return True

        # Find Abstract to know what's "early"
        abstract_idx = None
        for i, item in enumerate(texts):
            if item.get('label') == 'section_header' and 'abstract' in item.get('text', '').lower():
                abstract_idx = i
                break

        # Filter out problematic footnotes
        removed_count = 0
        filtered_texts = []

        for idx, item in enumerate(texts):
            # Remove footnotes with author info
            if item.get('label') == 'footnote':
                # Aggressive removal in early section
                if abstract_idx is None or idx < (abstract_idx + 50):
                    if should_remove_footnote(item.get('text', '')):
                        removed_count += 1
                        continue
                # Moderate removal of email patterns everywhere
                elif re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', item.get('text', '')):
                    removed_count += 1
                    continue

            # Also check page footers for emails
            if item.get('label') == 'page_footer':
                if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', item.get('text', '')):
                    removed_count += 1
                    continue

            filtered_texts.append(item)

        # Update document
        doc['texts'] = filtered_texts

        # Write back
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        if removed_count > 0:
            log.info(f"{Path(json_path).stem}: Removed {removed_count} author/email footnotes")

        return True

    except Exception as e:
        log.error(f"{Path(json_path).stem}: Failed - {e}")
        return False


def process_file(args):
    """Wrapper for multiprocessing."""
    return fix_footnotes(args)


def main():
    parser = argparse.ArgumentParser(
        description="Remove author emails from footnotes in anonymized PDFs"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing anonymized_pdfs/")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    anon_dir = os.path.join(args.input_dir, "anonymized_pdfs")

    if not os.path.isdir(anon_dir):
        log.error(f"Directory not found: {anon_dir}")
        return 1

    # Get list of JSON files
    json_files = sorted([
        os.path.join(anon_dir, f)
        for f in os.listdir(anon_dir)
        if f.endswith('.json')
    ])

    if not json_files:
        log.warning(f"No JSON files found in {anon_dir}")
        return 0

    log.info(f"Processing {len(json_files)} anonymized PDFs...")

    # Process in parallel
    with Pool(args.workers) as pool:
        results = pool.map(process_file, json_files)

    success = sum(results)
    log.info(f"Fixed {success}/{len(json_files)} files")
    return 0 if success == len(json_files) else 1


if __name__ == "__main__":
    exit(main())
