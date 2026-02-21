#!/usr/bin/env python3
"""Parse PDFs in parallel using two GPUs.

Splits PDFs into two batches and runs Docling on each GPU simultaneously.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process

# Set CUDA device before importing docling
def parse_batch(pdf_files, input_dir, gpu_id, process_id):
    """Parse a batch of PDFs on a specific GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import after setting CUDA device
    from docling.document_converter import DocumentConverter

    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU{gpu_id}] %(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    pdfs_dir = os.path.join(input_dir, "pdfs")
    parsed_dir = os.path.join(input_dir, "parsed_pdfs")
    os.makedirs(parsed_dir, exist_ok=True)

    log.info(f"Process {process_id} starting with {len(pdf_files)} PDFs on GPU {gpu_id}")

    # Initialize converter
    log.info("Initializing Docling converter...")
    converter = DocumentConverter()

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        paper_id = pdf_file.replace('.pdf', '')
        output_path = os.path.join(parsed_dir, f"{paper_id}.pdf.json")
        md_path = os.path.join(parsed_dir, f"{paper_id}.pdf.md")

        # Skip if already exists
        if os.path.exists(output_path):
            log.debug(f"Skipping {pdf_file} (already parsed)")
            skip_count += 1
            continue

        log.info(f"[{i}/{len(pdf_files)}] Parsing {pdf_file}...")

        start = time.time()
        try:
            result = converter.convert(pdf_path)

            # Export to markdown
            markdown = result.document.export_to_markdown()
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            # Export to JSON
            doc_dict = result.document.export_to_dict()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start
            success_count += 1

            json_size = os.path.getsize(output_path) / 1024
            md_size = os.path.getsize(md_path) / 1024
            log.info(f"  → Success ({elapsed:.1f}s, JSON: {json_size:.1f}KB, MD: {md_size:.1f}KB)")
        except Exception as e:
            log.error(f"Failed to parse {pdf_file}: {e}")
            fail_count += 1

        # Progress report every 10 papers
        if i % 10 == 0:
            log.info(f"Progress: {i}/{len(pdf_files)} ({success_count} success, {fail_count} failed, {skip_count} skipped)")

    log.info("=" * 60)
    log.info(f"Process {process_id} complete!")
    log.info(f"  Total PDFs: {len(pdf_files)}")
    log.info(f"  Successfully parsed: {success_count}")
    log.info(f"  Failed: {fail_count}")
    log.info(f"  Skipped: {skip_count}")


def main():
    parser = argparse.ArgumentParser(description="Parse PDFs in parallel using two GPUs")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing pdfs/ subdirectory")
    args = parser.parse_args()

    pdfs_dir = os.path.join(args.input_dir, "pdfs")

    if not os.path.isdir(pdfs_dir):
        print(f"Error: PDFs directory not found: {pdfs_dir}", file=sys.stderr)
        return 1

    # Get all PDFs
    pdf_files = sorted([f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')])
    print(f"Found {len(pdf_files)} PDFs to parse")

    # Split into two batches
    mid = len(pdf_files) // 2
    batch1 = pdf_files[:mid]
    batch2 = pdf_files[mid:]

    print(f"Batch 1 (GPU 0): {len(batch1)} PDFs")
    print(f"Batch 2 (GPU 1): {len(batch2)} PDFs")

    # Start two processes
    p1 = Process(target=parse_batch, args=(batch1, args.input_dir, 0, 1))
    p2 = Process(target=parse_batch, args=(batch2, args.input_dir, 1, 2))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("All parsing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
