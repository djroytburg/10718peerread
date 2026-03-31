#!/usr/bin/env python3
"""Scrape NeurIPS papers and reviews from OpenReview API v2 into PeerRead format.

Usage:
    python scrape_neurips.py --limit 200 --output-dir output/neurips_2025
    python scrape_neurips.py --limit 200 --output-dir output/neurips_2025 --validate
    python scrape_neurips.py --limit 200 --output-dir output/neurips_2025 --no-pdf

After scraping, parse PDFs with Docling:
    python parse_pdfs_docling.py --input-dir output/neurips_2025
"""

import argparse
import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import openreview

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_V2_URL = "https://api2.openreview.net"
OPENREVIEW_BASE = "https://openreview.net"
BATCH_SIZE = 250
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds
BATCH_SLEEP = 1  # seconds between pagination batches
PDF_DOWNLOAD_DELAY = 0.5  # seconds between PDF downloads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_value(field):
    """Unwrap OpenReview API v2 content fields: {'value': x} -> x."""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def parse_int(val):
    """Try to extract an integer from a value that may be a string like '7: ...'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        # Handle formats like "7: Good paper" -> 7
        stripped = val.strip().split(":")[0].split(" ")[0]
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def format_timestamp(ts):
    """Convert epoch-ms timestamp to YYYY-MM-DD string."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, ValueError, TypeError):
        return None


def get_pdf_url(note):
    """Extract the full PDF URL from a submission Note."""
    content = note.content or {}
    pdf_path = extract_value(content.get("pdf"))
    if not pdf_path:
        return None
    if pdf_path.startswith("http"):
        return pdf_path
    return OPENREVIEW_BASE + pdf_path


def download_pdf(url, dest_path):
    """Download a PDF with retry logic. Returns True on success."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PeerRead-Scraper/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 1000:
                log.warning("PDF suspiciously small (%d bytes): %s", len(data), url)
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            if attempt == MAX_RETRIES:
                log.warning("Failed to download PDF after %d attempts: %s — %s", MAX_RETRIES, url, e)
                return False
            delay = RETRY_BASE_DELAY ** attempt
            log.debug("PDF download attempt %d failed: %s — retrying in %ds", attempt, e, delay)
            time.sleep(delay)
    return False


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def create_client():
    """Create an unauthenticated OpenReview API v2 client."""
    return openreview.api.OpenReviewClient(baseurl=API_V2_URL)


def fetch_submissions(client, venue_id, limit):
    """Fetch submissions with directReplies in paginated batches.

    Returns a list of Note objects.
    """
    invitation = f"{venue_id}/-/Submission"
    all_notes = []
    offset = 0

    while True:
        batch_limit = min(BATCH_SIZE, limit - len(all_notes))
        if batch_limit <= 0:
            break

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.info(
                    "Fetching submissions offset=%d limit=%d (attempt %d)",
                    offset, batch_limit, attempt,
                )
                notes = client.get_notes(
                    invitation=invitation,
                    limit=batch_limit,
                    offset=offset,
                    details="directReplies",
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY ** attempt
                log.warning("API error (attempt %d): %s — retrying in %ds", attempt, e, delay)
                time.sleep(delay)

        if not notes:
            break

        all_notes.extend(notes)
        log.info("Fetched %d submissions so far", len(all_notes))

        if len(notes) < batch_limit:
            break  # no more results

        offset += len(notes)

        if len(all_notes) < limit:
            time.sleep(BATCH_SLEEP)

    return all_notes[:limit]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_paper(note, venue_id):
    """Extract paper metadata from a submission Note.

    Returns a dict with paper fields, or None if the paper should be skipped.
    """
    content = note.content or {}

    title = extract_value(content.get("title"))
    abstract = extract_value(content.get("abstract"))

    if not title or not abstract:
        log.warning("Skipping %s: missing title or abstract", note.id)
        return None

    # Determine acceptance status
    venueid = extract_value(content.get("venueid"))
    if venueid and "Reject" in str(venueid):
        accepted = False
    elif venueid == venue_id:
        accepted = True
    else:
        accepted = None  # still under review or unknown

    authors = extract_value(content.get("authors")) or []
    keywords = extract_value(content.get("keywords")) or []

    return {
        "title": title,
        "abstract": abstract,
        "id": note.id,
        "authors": authors,
        "accepted": accepted,
        "keywords": keywords,
        "conference": venue_id.replace(".cc/", " ").replace("/Conference", "").replace(".", ""),
    }


def classify_reply(reply):
    """Classify a directReply by its invitation string.

    Returns 'review', 'decision', or None (skip).
    """
    invitations = reply.invitations if hasattr(reply, "invitations") else []
    # Also check singular invitation field
    if not invitations and hasattr(reply, "invitation"):
        invitations = [reply.invitation] if reply.invitation else []

    for inv in invitations:
        if "Official_Review" in inv:
            return "review"
        if "Decision" in inv:
            return "decision"
    return None


def build_comments(content):
    """Concatenate review text sections into a single comments string."""
    sections = []

    # Map of possible field names to section headers
    field_map = [
        ("summary", "SUMMARY"),
        ("strengths_and_weaknesses", "STRENGTHS AND WEAKNESSES"),
        ("strengths", "STRENGTHS"),
        ("weaknesses", "WEAKNESSES"),
        ("questions", "QUESTIONS"),
        ("limitations", "LIMITATIONS"),
        ("feedback", "FEEDBACK"),
        ("review", "REVIEW"),
    ]

    for field_name, header in field_map:
        val = extract_value(content.get(field_name))
        if val and isinstance(val, str) and val.strip():
            sections.append(f"{header}:\n{val.strip()}")

    if not sections:
        # Fallback: use any text content
        for key in sorted(content.keys()):
            val = extract_value(content[key])
            if isinstance(val, str) and len(val) > 50:
                sections.append(f"{key.upper()}:\n{val.strip()}")

    return "\n\n".join(sections) if sections else "No review text available."


def parse_review(reply):
    """Parse an Official_Review reply into a PeerRead review dict."""
    content = reply.content or {}

    comments = build_comments(content)

    rating = parse_int(extract_value(content.get("rating")))
    confidence = parse_int(extract_value(content.get("confidence")))
    soundness = parse_int(extract_value(content.get("soundness")))
    clarity = parse_int(extract_value(content.get("clarity")))
    significance = parse_int(extract_value(content.get("significance")))
    originality = parse_int(extract_value(content.get("originality")))

    title = extract_value(content.get("title")) or "Official Review"
    date = format_timestamp(reply.cdate or reply.tcdate)

    review = {
        "IS_META_REVIEW": False,
        "comments": comments,
        "TITLE": title,
    }

    if rating is not None:
        review["RECOMMENDATION"] = rating
    if confidence is not None:
        review["REVIEWER_CONFIDENCE"] = confidence
    if soundness is not None:
        review["SOUNDNESS_CORRECTNESS"] = soundness
    if clarity is not None:
        review["CLARITY"] = clarity
    if significance is not None:
        review["IMPACT"] = significance
    if originality is not None:
        review["ORIGINALITY"] = originality
    if date is not None:
        review["DATE"] = date

    return review


def parse_decision(reply):
    """Parse a Decision reply into a PeerRead meta-review dict."""
    content = reply.content or {}

    decision_text = extract_value(content.get("decision")) or ""
    comment = extract_value(content.get("comment")) or ""

    comments_parts = []
    if decision_text:
        comments_parts.append(f"DECISION:\n{decision_text}")
    if comment:
        comments_parts.append(f"COMMENT:\n{comment}")

    return {
        "IS_META_REVIEW": True,
        "comments": "\n\n".join(comments_parts) if comments_parts else "No decision text.",
        "RECOMMENDATION": decision_text,
        "TITLE": "Decision",
        "DATE": format_timestamp(reply.cdate or reply.tcdate),
    }


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def build_peerread_record(note, venue_id):
    """Build a complete PeerRead-compatible dict from a submission Note.

    Returns (record_dict, stats_dict) or (None, None) if paper should be skipped.
    """
    paper = parse_paper(note, venue_id)
    if paper is None:
        return None, None

    reviews = []
    replies = []

    # directReplies can be in note.details dict
    if hasattr(note, "details") and isinstance(note.details, dict):
        replies = note.details.get("directReplies", [])

    for reply_data in replies:
        # replies may be dicts or Note objects
        if isinstance(reply_data, dict):
            # Convert to a simple namespace for uniform access
            reply = _dict_to_namespace(reply_data)
        else:
            reply = reply_data

        kind = classify_reply(reply)
        if kind == "review":
            reviews.append(parse_review(reply))
        elif kind == "decision":
            reviews.append(parse_decision(reply))

    # Filter out papers with no decision and no reviews (still under review)
    if paper["accepted"] is None and len(reviews) == 0:
        log.debug("Skipping %s: no decision and no reviews", paper["id"])
        return None, None

    record = {
        "title": paper["title"],
        "abstract": paper["abstract"],
        "id": paper["id"],
        "authors": paper["authors"],
        "accepted": paper["accepted"],
        "keywords": paper["keywords"],
        "conference": paper["conference"],
        "reviews": reviews,
        "histories": [],
    }

    stats = {
        "accepted": paper["accepted"],
        "num_reviews": len([r for r in reviews if not r.get("IS_META_REVIEW")]),
        "ratings": [r["RECOMMENDATION"] for r in reviews
                    if not r.get("IS_META_REVIEW") and "RECOMMENDATION" in r],
    }

    return record, stats


class _Namespace:
    """Minimal namespace for dict-to-object conversion."""
    pass


def _dict_to_namespace(d):
    """Convert a dict (possibly nested) to a namespace object for uniform attribute access."""
    ns = _Namespace()
    for key, val in d.items():
        if key == "content" and isinstance(val, dict):
            # Keep content as dict for extract_value
            setattr(ns, key, val)
        else:
            setattr(ns, key, val)
    # Ensure required attributes exist
    if not hasattr(ns, "content"):
        ns.content = {}
    if not hasattr(ns, "invitations"):
        ns.invitations = []
    if not hasattr(ns, "invitation"):
        ns.invitation = None
    if not hasattr(ns, "cdate"):
        ns.cdate = None
    if not hasattr(ns, "tcdate"):
        ns.tcdate = None
    return ns


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_record(record, output_dir):
    """Write a single PeerRead record as a JSON line file."""
    reviews_dir = os.path.join(output_dir, "reviews")
    os.makedirs(reviews_dir, exist_ok=True)

    filepath = os.path.join(reviews_dir, f"{record['id']}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

    return filepath


def download_paper_pdf(note, output_dir):
    """Download the PDF for a paper. Returns True on success, False on failure/skip."""
    pdf_url = get_pdf_url(note)
    if not pdf_url:
        log.debug("No PDF URL for %s", note.id)
        return False

    pdfs_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdfs_dir, exist_ok=True)

    dest = os.path.join(pdfs_dir, f"{note.id}.pdf")
    if os.path.exists(dest):
        log.debug("PDF already exists: %s", dest)
        return True

    return download_pdf(pdf_url, dest)


def write_manifest(all_stats, output_dir, pdfs_downloaded=0, pdfs_failed=0):
    """Write manifest.json with scrape summary statistics."""
    total = len(all_stats)
    accepted = sum(1 for s in all_stats if s["accepted"] is True)
    rejected = sum(1 for s in all_stats if s["accepted"] is False)
    unknown = sum(1 for s in all_stats if s["accepted"] is None)
    all_ratings = [r for s in all_stats for r in s["ratings"]]
    review_counts = [s["num_reviews"] for s in all_stats]

    manifest = {
        "scrape_date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_papers": total,
        "accepted": accepted,
        "rejected": rejected,
        "unknown_status": unknown,
        "total_reviews": sum(review_counts),
        "avg_reviews_per_paper": round(sum(review_counts) / max(total, 1), 2),
        "pdfs_downloaded": pdfs_downloaded,
        "pdfs_failed": pdfs_failed,
        "rating_stats": {
            "count": len(all_ratings),
            "mean": round(sum(all_ratings) / max(len(all_ratings), 1), 2),
            "min": min(all_ratings) if all_ratings else None,
            "max": max(all_ratings) if all_ratings else None,
        },
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log.info("Manifest written to %s", manifest_path)
    log.info(
        "Stats: %d papers (%d accepted, %d rejected, %d unknown), "
        "%d reviews, avg %.1f reviews/paper, mean rating %.1f",
        total, accepted, rejected, unknown,
        sum(review_counts),
        manifest["avg_reviews_per_paper"],
        manifest["rating_stats"]["mean"] if all_ratings else 0,
    )

    return manifest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(output_dir):
    """Post-scrape validation of output files."""
    reviews_dir = os.path.join(output_dir, "reviews")
    if not os.path.isdir(reviews_dir):
        log.error("Validation failed: %s does not exist", reviews_dir)
        return False

    files = [f for f in os.listdir(reviews_dir) if f.endswith(".json")]
    log.info("Validating %d files in %s", len(files), reviews_dir)

    errors = 0
    forum_ids = set()
    acceptance_dist = {"True": 0, "False": 0, "None": 0}
    review_counts = []
    all_ratings = []

    for fname in files:
        filepath = os.path.join(reviews_dir, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.loads(f.readline().strip())
        except (json.JSONDecodeError, IOError) as e:
            log.error("  FAIL %s: invalid JSON — %s", fname, e)
            errors += 1
            continue

        # Check required fields
        for field in ("title", "abstract", "reviews"):
            if field not in data:
                log.error("  FAIL %s: missing '%s'", fname, field)
                errors += 1

        # Check reviews have comments
        for i, rev in enumerate(data.get("reviews", [])):
            if "comments" not in rev:
                log.error("  FAIL %s: review[%d] missing 'comments'", fname, i)
                errors += 1

        # Duplicate check
        fid = data.get("id", fname)
        if fid in forum_ids:
            log.error("  FAIL %s: duplicate forum ID %s", fname, fid)
            errors += 1
        forum_ids.add(fid)

        # Stats
        acc_key = str(data.get("accepted"))
        acceptance_dist[acc_key] = acceptance_dist.get(acc_key, 0) + 1

        non_meta = [r for r in data.get("reviews", []) if not r.get("IS_META_REVIEW")]
        review_counts.append(len(non_meta))
        for r in non_meta:
            if "RECOMMENDATION" in r and isinstance(r["RECOMMENDATION"], (int, float)):
                all_ratings.append(r["RECOMMENDATION"])

    # Report
    log.info("--- Validation Report ---")
    log.info("Files checked: %d", len(files))
    log.info("Errors: %d", errors)
    log.info("Acceptance distribution: %s", acceptance_dist)
    log.info(
        "Review count distribution: min=%d, max=%d, mean=%.1f",
        min(review_counts) if review_counts else 0,
        max(review_counts) if review_counts else 0,
        sum(review_counts) / max(len(review_counts), 1),
    )
    if all_ratings:
        log.info(
            "Rating stats: min=%d, max=%d, mean=%.2f (n=%d)",
            min(all_ratings), max(all_ratings),
            sum(all_ratings) / len(all_ratings), len(all_ratings),
        )

    # PDF check
    pdfs_dir = os.path.join(output_dir, "pdfs")
    if os.path.isdir(pdfs_dir):
        pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]
        log.info("PDFs found: %d", len(pdf_files))
        # Check each review file has a corresponding PDF
        review_ids = {f.replace(".json", "") for f in files}
        pdf_ids = {f.replace(".pdf", "") for f in pdf_files}
        missing_pdfs = review_ids - pdf_ids
        if missing_pdfs:
            log.warning("Papers missing PDFs: %d", len(missing_pdfs))
    else:
        log.info("No pdfs/ directory found (PDFs not downloaded)")

    if errors == 0:
        log.info("Validation PASSED")
    else:
        log.error("Validation FAILED with %d errors", errors)

    return errors == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape NeurIPS papers and reviews from OpenReview into PeerRead format."
    )
    parser.add_argument("--limit", type=int, default=200,
                        help="Max number of submissions to fetch (default: 200)")
    parser.add_argument("--output-dir", default="output/neurips_2025",
                        help="Output directory (default: output/neurips_2025)")
    parser.add_argument("--year", type=int, default=2025,
                        help="NeurIPS year (default: 2025)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip downloading PDFs")
    parser.add_argument("--validate", action="store_true",
                        help="Run post-scrape validation")
    args = parser.parse_args()

    venue_id = f"NeurIPS.cc/{args.year}/Conference"
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Scraping %s (limit=%d) → %s", venue_id, args.limit, args.output_dir)

    client = create_client()
    notes = fetch_submissions(client, venue_id, args.limit)
    log.info("Retrieved %d submissions from API", len(notes))

    all_stats = []
    written = 0
    skipped = 0
    pdfs_ok = 0
    pdfs_fail = 0
    download_pdfs = not args.no_pdf

    # Build records first, then download PDFs
    records_and_notes = []
    for note in notes:
        record, stats = build_peerread_record(note, venue_id)
        if record is None:
            skipped += 1
            continue

        write_record(record, args.output_dir)
        all_stats.append(stats)
        records_and_notes.append(note)
        written += 1

    log.info("Wrote %d papers, skipped %d", written, skipped)

    if download_pdfs and records_and_notes:
        log.info("Downloading PDFs for %d papers...", len(records_and_notes))
        for i, note in enumerate(records_and_notes, 1):
            if download_paper_pdf(note, args.output_dir):
                pdfs_ok += 1
            else:
                pdfs_fail += 1
            if i % 25 == 0:
                log.info("  PDF progress: %d/%d downloaded", pdfs_ok, i)
            if i < len(records_and_notes):
                time.sleep(PDF_DOWNLOAD_DELAY)
        log.info("PDFs: %d downloaded, %d failed", pdfs_ok, pdfs_fail)

    if all_stats:
        write_manifest(all_stats, args.output_dir, pdfs_ok, pdfs_fail)

    if args.validate:
        run_validation(args.output_dir)


if __name__ == "__main__":
    main()
