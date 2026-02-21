#!/usr/bin/env python3
"""Parse PDFs using Docling to extract structured content.

Usage:
    python parse_pdfs_docling.py --input-dir output/neurips_2025
    python parse_pdfs_docling.py --input-dir output/neurips_2025 --limit 10
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_pdf(pdf_path, output_path, converter):
    """Parse a single PDF and save markdown + JSON.

    Returns True on success, False on failure.
    """
    try:
        result = converter.convert(pdf_path)

        # Export to markdown
        markdown = result.document.export_to_markdown()
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        # Export to JSON (full structured document)
        doc_dict = result.document.export_to_dict()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        log.error("Failed to parse %s: %s", pdf_path, e)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Parse PDFs with Docling to extract structured content"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing pdfs/ subdirectory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of PDFs to parse (for testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip PDFs that already have parsed output")
    args = parser.parse_args()

    pdfs_dir = os.path.join(args.input_dir, "pdfs")
    parsed_dir = os.path.join(args.input_dir, "parsed_pdfs")

    if not os.path.isdir(pdfs_dir):
        log.error("PDFs directory not found: %s", pdfs_dir)
        return 1

    os.makedirs(parsed_dir, exist_ok=True)

    # Get list of PDFs to parse
    pdf_files = sorted([f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')])

    if args.limit:
        pdf_files = pdf_files[:args.limit]

    log.info("Found %d PDFs to parse in %s", len(pdf_files), pdfs_dir)

    # Initialize converter once (reuse for all PDFs)
    log.info("Initializing Docling converter...")
    converter = DocumentConverter()

    # Parse PDFs
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        paper_id = pdf_file.replace('.pdf', '')
        output_path = os.path.join(parsed_dir, f"{paper_id}.pdf.json")
        md_path = os.path.join(parsed_dir, f"{paper_id}.pdf.md")

        # Skip if already exists
        if args.skip_existing and os.path.exists(output_path):
            log.debug("Skipping %s (already parsed)", pdf_file)
            skip_count += 1
            continue

        log.info("[%d/%d] Parsing %s...", i, len(pdf_files), pdf_file)

        start = time.time()
        if parse_pdf(pdf_path, output_path, converter):
            elapsed = time.time() - start
            success_count += 1

            # Log file sizes
            json_size = os.path.getsize(output_path) / 1024  # KB
            md_size = os.path.getsize(md_path) / 1024  # KB
            log.info("  → Success (%.1fs, JSON: %.1fKB, MD: %.1fKB)",
                     elapsed, json_size, md_size)
        else:
            fail_count += 1

        # Progress report every 10 papers
        if i % 10 == 0:
            log.info("Progress: %d/%d parsed (%d success, %d failed, %d skipped)",
                     i, len(pdf_files), success_count, fail_count, skip_count)

    # Summary
    log.info("=" * 60)
    log.info("Parsing complete!")
    log.info("  Total PDFs: %d", len(pdf_files))
    log.info("  Successfully parsed: %d", success_count)
    log.info("  Failed: %d", fail_count)
    log.info("  Skipped: %d", skip_count)
    log.info("Output directory: %s", parsed_dir)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
