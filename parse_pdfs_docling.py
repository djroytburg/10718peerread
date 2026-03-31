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
import signal
import subprocess
import sys
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("PDF parsing exceeded timeout")

def parse_pdf(pdf_path, output_path, converter, timeout_secs=120):
    """Parse a single PDF and save markdown + JSON.

    Returns True on success, False on failure.
    Gracefully handles corrupted PDFs and C-level crashes.
    """
    # Set alarm for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_secs)

    try:
        # Check file integrity first
        if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) == 0:
            log.warning("PDF file is missing or empty: %s", pdf_path)
            signal.alarm(0)  # Cancel alarm
            return False

        # Skip PDFs over 10MB (known to cause infinite hangs)
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > 10:
            log.warning("Skipping %s: file too large (%.1f MB)", pdf_path, file_size_mb)
            signal.alarm(0)
            return False

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

        signal.alarm(0)  # Cancel alarm
        return True
    except (KeyboardInterrupt, SystemExit):
        signal.alarm(0)
        raise  # Re-raise interrupts/exits
    except TimeoutError as e:
        log.error("Timeout parsing %s (exceeded %d seconds)", pdf_path, timeout_secs)
        signal.alarm(0)
        return False
    except Exception as e:
        # Catches Python exceptions; C-level segfaults will still crash the process
        # but --skip-existing will resume from where it left off on restart
        log.error("Failed to parse %s: %s (likely corrupted PDF)", pdf_path, e)
        signal.alarm(0)
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
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per PDF in seconds (default: 300)")
    args = parser.parse_args()

    pdfs_dir = os.path.join(args.input_dir, "pdfs")
    parsed_dir = os.path.join(args.input_dir, "parsed_pdfs")
    failed_file = os.path.join(parsed_dir, ".failed_pdfs.txt")  # Track problematic PDFs

    if not os.path.isdir(pdfs_dir):
        log.error("PDFs directory not found: %s", pdfs_dir)
        return 1

    os.makedirs(parsed_dir, exist_ok=True)

    # Load list of previously failed PDFs (to skip them)
    failed_pdfs = set()
    if os.path.exists(failed_file):
        with open(failed_file, 'r') as f:
            failed_pdfs = set(line.strip() for line in f if line.strip())
        if failed_pdfs:
            log.warning("Skipping %d previously failed PDFs", len(failed_pdfs))

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
    newly_failed = []

    for i, pdf_file in enumerate(pdf_files, 1):
        paper_id = pdf_file.replace('.pdf', '')

        # Skip if in failed list (already tried and crashed)
        if paper_id in failed_pdfs:
            log.debug("Skipping %s (previously failed)", pdf_file)
            skip_count += 1
            continue

        pdf_path = os.path.join(pdfs_dir, pdf_file)
        output_path = os.path.join(parsed_dir, f"{paper_id}.pdf.json")
        md_path = os.path.join(parsed_dir, f"{paper_id}.pdf.md")

        # Skip if already exists
        if args.skip_existing and os.path.exists(output_path):
            log.debug("Skipping %s (already parsed)", pdf_file)
            skip_count += 1
            continue

        log.info("[%d/%d] Parsing %s...", i, len(pdf_files), pdf_file)

        start = time.time()
        if parse_pdf(pdf_path, output_path, converter, args.timeout):
            elapsed = time.time() - start
            success_count += 1

            # Log file sizes
            json_size = os.path.getsize(output_path) / 1024  # KB
            md_size = os.path.getsize(md_path) / 1024  # KB
            log.info("  → Success (%.1fs, JSON: %.1fKB, MD: %.1fKB)",
                     elapsed, json_size, md_size)
        else:
            fail_count += 1
            newly_failed.append(paper_id)
            log.warning("Marked %s as failed (will skip on next run)", paper_id)

        # Progress report every 10 papers
        if i % 10 == 0:
            log.info("Progress: %d/%d parsed (%d success, %d failed, %d skipped)",
                     i, len(pdf_files), success_count, fail_count, skip_count)

    # Write newly failed PDFs to tracking file
    if newly_failed:
        with open(failed_file, 'a') as f:
            for pid in newly_failed:
                f.write(pid + '\n')
        log.warning("Recorded %d failed PDFs in %s", len(newly_failed), failed_file)

    # Summary
    log.info("=" * 60)
    log.info("Parsing complete!")
    log.info("  Total PDFs: %d", len(pdf_files))
    log.info("  Successfully parsed: %d", success_count)
    log.info("  Failed: %d", fail_count)
    log.info("  Skipped: %d", skip_count)
    log.info("  Total failures on record: %d", len(failed_pdfs) + len(newly_failed))
    log.info("Output directory: %s", parsed_dir)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
