#!/usr/bin/env python3
"""Create balanced train/test split from anonymized papers."""

import json
import random
import shutil
from pathlib import Path

# Paths
SOURCE_DIR = Path("PeerRead/data/neurips_2025_anonymized")
OUTPUT_DIR = Path("PeerRead/data/neurips_2025_balanced_anon")

# Load all papers and check accept/reject status
accepts = []
rejects = []

for split in ['train', 'dev', 'test']:
    reviews_dir = SOURCE_DIR / split / 'reviews'
    if not reviews_dir.exists():
        continue

    for review_file in reviews_dir.glob('*.json'):
        with open(review_file) as f:
            data = json.load(f)

        paper_id = data['id']
        accepted = data.get('accepted')

        if accepted is True:
            accepts.append(paper_id)
        elif accepted is False:
            rejects.append(paper_id)

print(f"Total papers: {len(accepts) + len(rejects)}")
print(f"  Accepts: {len(accepts)}")
print(f"  Rejects: {len(rejects)}")
print()

# Balance classes
n_samples = min(len(accepts), len(rejects))
print(f"Creating balanced dataset with {n_samples} accepts and {n_samples} rejects")

random.seed(42)
random.shuffle(accepts)
random.shuffle(rejects)

balanced_accepts = accepts[:n_samples]
balanced_rejects = rejects[:n_samples]

# Combine and split
all_papers = balanced_accepts + balanced_rejects
random.shuffle(all_papers)

# 70/30 split for train/test
n_train = int(len(all_papers) * 0.7)
train_papers = all_papers[:n_train]
test_papers = all_papers[n_train:]

print(f"\nSplit sizes:")
print(f"  Train: {len(train_papers)} papers")
print(f"  Test:  {len(test_papers)} papers")

# Check balance
train_accepts = sum(1 for p in train_papers if p in balanced_accepts)
test_accepts = sum(1 for p in test_papers if p in balanced_accepts)

print(f"\nClass distribution:")
print(f"  Train: {train_accepts} accepts, {len(train_papers) - train_accepts} rejects")
print(f"  Test:  {test_accepts} accepts, {len(test_papers) - test_accepts} rejects")

# Create output directories
for split in ['train', 'test']:
    (OUTPUT_DIR / split / 'reviews').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / 'parsed_pdfs').mkdir(parents=True, exist_ok=True)

# Copy files
def copy_paper_files(paper_id, split):
    """Copy review and parsed PDF for a paper."""
    # Find source files (check all source splits)
    for source_split in ['train', 'dev', 'test']:
        review_src = SOURCE_DIR / source_split / 'reviews' / f'{paper_id}.json'
        pdf_src = SOURCE_DIR / source_split / 'parsed_pdfs' / f'{paper_id}.pdf.json'

        if review_src.exists() and pdf_src.exists():
            review_dst = OUTPUT_DIR / split / 'reviews' / f'{paper_id}.json'
            pdf_dst = OUTPUT_DIR / split / 'parsed_pdfs' / f'{paper_id}.pdf.json'

            shutil.copy2(review_src, review_dst)
            shutil.copy2(pdf_src, pdf_dst)
            return True

    print(f"Warning: Could not find files for {paper_id}")
    return False

print("\nCopying files...")
for paper_id in train_papers:
    copy_paper_files(paper_id, 'train')

for paper_id in test_papers:
    copy_paper_files(paper_id, 'test')

print(f"\n✅ Balanced dataset created at {OUTPUT_DIR}")
print(f"   {len(train_papers)} train + {len(test_papers)} test = {len(all_papers)} total")
