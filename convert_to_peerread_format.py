#!/usr/bin/env python3
"""Convert NeurIPS 2025 data to PeerRead format.

This script:
1. Converts Docling JSON format to PeerRead ScienceParse format
2. Creates train/dev/test splits
3. Copies review files to appropriate locations
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


def docling_to_scienceparse(docling_json, paper_id, title, abstract):
    """Convert Docling JSON format to PeerRead ScienceParse format.

    Docling format has:
        - texts: [{text, label}, ...]

    PeerRead ScienceParse format has:
        - name, metadata: {source, title, authors, emails, sections: [{heading, text}]}
    """
    texts = docling_json.get('texts', [])

    # Since PDFs are anonymized, we don't have real authors/emails
    # Use placeholders (get_num_authors() will return 0 anyway for anonymized PDFs)
    authors = []
    emails = []

    # Extract sections from texts
    sections = []
    current_section = None
    current_text = []

    for item in texts:
        label = item.get('label', 'text')
        text = item.get('text', '').strip()

        if not text:
            continue

        if label == 'section_header':
            # Save previous section if exists
            if current_section is not None and current_text:
                sections.append({
                    'heading': current_section,
                    'text': ' '.join(current_text)
                })
            # Start new section
            current_section = text
            current_text = []
        elif label in ('text', 'list_item', 'caption', 'page_header', 'page_footer'):
            # Add to current section
            if current_section is not None:
                current_text.append(text)

    # Save last section
    if current_section is not None and current_text:
        sections.append({
            'heading': current_section,
            'text': ' '.join(current_text)
        })

    # Build ScienceParse format
    scienceparse = {
        'name': f'{paper_id}.pdf',
        'metadata': {
            'source': 'Docling',  # Was 'CRF' in original PeerRead
            'title': title,
            'authors': authors,  # Empty for anonymized PDFs
            'emails': emails,    # Empty for anonymized PDFs
            'sections': sections,
            'references': [],    # Docling doesn't extract structured references
            'referenceMentions': []  # Docling doesn't extract reference mentions
        }
    }

    return scienceparse


def load_reviews_and_pdfs(reviews_dir, pdfs_dir):
    """Load review JSONs and match with parsed PDFs."""
    reviews_dir = Path(reviews_dir)
    pdfs_dir = Path(pdfs_dir)

    papers = []

    # Load all review files
    review_files = sorted(reviews_dir.glob('*.json'))
    print(f"Found {len(review_files)} review files")

    matched = 0
    missing_pdfs = []

    for review_file in review_files:
        # Load review JSON
        with open(review_file) as f:
            review_data = json.load(f)

        paper_id = review_data['id']

        # Find corresponding PDF
        print(pdfs_dir)
        import os
        print(os.listdir(pdfs_dir))
        pdf_file = pdfs_dir / f'{paper_id}.pdf.json'

        if not pdf_file.exists():
            missing_pdfs.append(paper_id)
            continue

        # Load PDF JSON
        with open(pdf_file) as f:
            pdf_data = json.load(f)

        # Convert to ScienceParse format
        scienceparse = docling_to_scienceparse(
            pdf_data,
            paper_id,
            review_data['title'],
            review_data['abstract']
        )

        papers.append({
            'id': paper_id,
            'review': review_data,
            'scienceparse': scienceparse,
            'review_file': review_file,
            'pdf_file': pdf_file
        })
        matched += 1

    print(f"Matched {matched} papers with PDFs")
    if missing_pdfs:
        print(f"Missing PDFs for {len(missing_pdfs)} papers: {missing_pdfs[:5]}...")

    return papers


def create_splits(papers, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42):
    """Split papers into train/dev/test sets."""
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    random.shuffle(papers)

    n = len(papers)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    splits = {
        'train': papers[:train_end],
        'dev': papers[train_end:dev_end],
        'test': papers[dev_end:]
    }

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])} papers")
    print(f"  Dev:   {len(splits['dev'])} papers")
    print(f"  Test:  {len(splits['test'])} papers")

    return splits


def write_peerread_data(splits, output_dir):
    """Write data in PeerRead format."""
    output_dir = Path(output_dir)

    for split_name, papers in splits.items():
        split_dir = output_dir / split_name
        reviews_dir = split_dir / 'reviews'
        pdfs_dir = split_dir / 'parsed_pdfs'

        # Create directories
        reviews_dir.mkdir(parents=True, exist_ok=True)
        pdfs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting {split_name} split...")

        for paper in papers:
            # Write review JSON (single line, no indentation - PeerRead format)
            review_out = reviews_dir / f"{paper['id']}.json"
            with open(review_out, 'w') as f:
                json.dump(paper['review'], f, ensure_ascii=False)

            # Write ScienceParse JSON (pretty-printed is OK for parsed_pdfs)
            pdf_out = pdfs_dir / f"{paper['id']}.pdf.json"
            with open(pdf_out, 'w') as f:
                json.dump(paper['scienceparse'], f, indent=2, ensure_ascii=False)

        print(f"  Wrote {len(papers)} files to {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeurIPS 2025 data to PeerRead format"
    )
    parser.add_argument("--reviews-dir", required=True,
                       help="Directory with review JSONs")
    parser.add_argument("--pdfs-dir", required=True,
                       help="Directory with anonymized parsed PDF JSONs")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for PeerRead-format data")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Train split ratio (default: 0.8)")
    parser.add_argument("--dev-ratio", type=float, default=0.1,
                       help="Dev split ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for splitting (default: 42)")
    args = parser.parse_args()

    # Load and match data
    print("Loading reviews and PDFs...")
    papers = load_reviews_and_pdfs(args.reviews_dir, args.pdfs_dir)

    if not papers:
        print("ERROR: No papers matched!")
        return 1

    # Create splits
    splits = create_splits(
        papers,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Write PeerRead format
    write_peerread_data(splits, args.output_dir)

    print(f"\n✅ Conversion complete!")
    print(f"Data written to: {args.output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {args.output_dir}/train/reviews/")
    print(f"  {args.output_dir}/train/parsed_pdfs/")
    print(f"  {args.output_dir}/dev/reviews/")
    print(f"  {args.output_dir}/dev/parsed_pdfs/")
    print(f"  {args.output_dir}/test/reviews/")
    print(f"  {args.output_dir}/test/parsed_pdfs/")

    return 0


if __name__ == '__main__':
    exit(main())
