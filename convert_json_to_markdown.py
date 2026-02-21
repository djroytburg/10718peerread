#!/usr/bin/env python3
"""Convert anonymized JSON files to clean, readable markdown.

This makes it easy to validate anonymization by reading the markdown output.
"""

import argparse
import json
from pathlib import Path


def json_to_markdown(doc):
    """Convert a parsed PDF JSON document to markdown."""
    lines = []

    texts = doc.get('texts', [])

    for item in texts:
        text = item.get('text', '').strip()
        label = item.get('label', 'text')

        if not text:
            continue

        # Format based on label
        if label == 'section_header':
            # Determine heading level
            if text.startswith(('1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ')):
                lines.append(f"\n## {text}\n")
            else:
                # Main sections like Abstract
                lines.append(f"\n# {text}\n")

        elif label == 'list_item':
            lines.append(f"- {text}")

        elif label == 'caption':
            lines.append(f"\n*{text}*\n")

        elif label == 'formula':
            lines.append(f"\n```\n{text}\n```\n")

        elif label == 'footnote':
            lines.append(f"\n> {text}\n")

        elif label == 'page_footer':
            # Skip page footers in markdown
            continue

        else:  # Regular text
            lines.append(text)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert anonymized JSON files to readable markdown"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .pdf.json files")
    parser.add_argument("--output-dir",
                        help="Output directory (default: input-dir + '_markdown')")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to convert (for testing)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(str(input_dir) + '_markdown')

    json_files = sorted(input_dir.glob("*.pdf.json"))

    if args.limit:
        json_files = json_files[:args.limit]

    print(f"Found {len(json_files)} JSON files")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    for i, filepath in enumerate(json_files, 1):
        with open(filepath, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        markdown = json_to_markdown(doc)

        # Save as .md file (use same base name as JSON)
        output_path = output_dir / filepath.name.replace('.pdf.json', '.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        if i % 25 == 0:
            print(f"Converted {i}/{len(json_files)} files")

    print(f"\nConverted {len(json_files)} files to markdown")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
