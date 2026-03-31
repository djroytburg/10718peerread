#!/usr/bin/env python3
"""Validate anonymization by checking for author-related leaks."""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def check_for_leaks(doc):
    """Check a document for potential author information leaks.

    Returns:
        dict with leak types and examples
    """
    leaks = defaultdict(list)
    texts = doc.get('texts', [])

    # Only check first 30 elements (after title/abstract)
    # Authors should be completely gone from this region
    check_region = texts[1:30] if len(texts) > 30 else texts[1:]

    for i, item in enumerate(check_region, 1):
        text = item.get('text', '').strip()
        label = item.get('label', '')

        if not text:
            continue

        # Check for emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            leaks['email'].append((i, text[:100]))

        # Check for institutions
        institution_keywords = [
            'university', 'institute', 'laboratory', 'college',
            'department', 'lab inc.', 'corporation'
        ]
        if any(kw in text.lower() for kw in institution_keywords):
            # Filter out common false positives in abstract/intro
            if len(text) < 100 and label != 'text':
                leaks['institution'].append((i, text[:100]))

        # Check for "author" related text
        author_keywords = ['corresponding author', 'equal contribution',
                          'work was done', 'these authors']
        if any(kw in text.lower() for kw in author_keywords):
            leaks['author_mention'].append((i, text[:100]))

        # Check for person names (simple heuristic)
        # Pattern: Two capitalized words possibly with numbers/symbols
        # But ignore section headers like "1 Introduction"
        name_pattern = r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[\s0-9∗†‡,]*$'
        if re.match(name_pattern, text.strip()):
            # Additional filter: not a section header
            if not any(text.lower().startswith(word) for word in
                      ['abstract', 'introduction', 'related', 'methods',
                       'results', 'conclusion', 'experiments', 'background']):
                if label != 'section_header':
                    # Filter out common false positives (figure labels, technical terms)
                    false_positive_keywords = [
                        'phase', 'prompt', 'capture', 'model', 'prior',
                        'loss', 'layer', 'attention', 'task', 'mode',
                        'policy', 'agent', 'state', 'action', 'reward'
                    ]
                    if not any(kw in text.lower() for kw in false_positive_keywords):
                        # Only flag if it's in a very early position (1-10)
                        # and doesn't look like a figure label
                        if i <= 10:
                            leaks['possible_name'].append((i, text[:100]))

        # Check for affiliation footnotes
        if label == 'footnote':
            if any(kw in text.lower() for kw in
                  ['university', 'institute', 'equal', 'corresponding', 'contribution']):
                leaks['affiliation_footnote'].append((i, text[:100]))

    return dict(leaks)


def main():
    parser = argparse.ArgumentParser(
        description="Validate anonymization by checking for leaks"
    )
    parser.add_argument("--input-dir", required=True,
                       help="Directory with anonymized .pdf.json files")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed leak examples")
    parser.add_argument("--limit", type=int,
                       help="Limit number of files to check")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("*.pdf.json"))

    if args.limit:
        json_files = json_files[:args.limit]

    print(f"Checking {len(json_files)} files for anonymization leaks...\n")

    files_with_leaks = 0
    leak_stats = defaultdict(int)
    leak_examples = defaultdict(list)

    for i, filepath in enumerate(json_files, 1):
        with open(filepath) as f:
            doc = json.load(f)

        leaks = check_for_leaks(doc)

        if leaks:
            files_with_leaks += 1

            if args.verbose:
                print(f"\n[{i}] {filepath.name} - LEAKS FOUND:")
                for leak_type, examples in leaks.items():
                    print(f"  {leak_type}: {len(examples)} instances")
                    for idx, text in examples[:2]:  # Show first 2
                        print(f"    [{idx}] {text}")

            for leak_type, examples in leaks.items():
                leak_stats[leak_type] += len(examples)
                if len(leak_examples[leak_type]) < 10:  # Keep first 10 examples
                    leak_examples[leak_type].extend([
                        (filepath.name, idx, text)
                        for idx, text in examples[:2]
                    ])

        if i % 50 == 0:
            print(f"Processed {i}/{len(json_files)} files...")

    # Summary
    print(f"\n{'='*60}")
    print("ANONYMIZATION VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files checked: {len(json_files)}")
    print(f"Files with leaks: {files_with_leaks} ({100*files_with_leaks/len(json_files):.1f}%)")
    print(f"Clean files: {len(json_files) - files_with_leaks} ({100*(len(json_files)-files_with_leaks)/len(json_files):.1f}%)")

    if leak_stats:
        print(f"\nLeak breakdown:")
        for leak_type, count in sorted(leak_stats.items()):
            print(f"  {leak_type}: {count} instances")

        if not args.verbose and leak_examples:
            print(f"\nSample leaks (use --verbose for full details):")
            for leak_type, examples in leak_examples.items():
                print(f"\n  {leak_type}:")
                for filename, idx, text in examples[:3]:
                    print(f"    {filename} [{idx}]: {text[:80]}")
    else:
        print("\n✅ No leaks detected!")

    print(f"\n{'='*60}")

    # Return exit code based on leak rate
    leak_rate = files_with_leaks / len(json_files)
    if leak_rate > 0.1:  # More than 10%
        print(f"⚠️  HIGH leak rate: {leak_rate*100:.1f}%")
        return 1
    elif leak_rate > 0:
        print(f"✓ Acceptable leak rate: {leak_rate*100:.1f}%")
        return 0
    else:
        print("✅ Perfect anonymization!")
        return 0


if __name__ == "__main__":
    exit(main())
