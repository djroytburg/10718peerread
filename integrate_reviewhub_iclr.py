#!/usr/bin/env python3
"""Download and normalize ReviewHub/ICLR into this repo's review JSON layout."""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

from datasets import get_dataset_config_names, load_dataset


def first_present(record, keys, default=None):
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return default


def sanitize_id(raw_id, split, idx):
    if raw_id is None:
        return f"{split}_{idx:07d}"
    text = str(raw_id).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or f"{split}_{idx:07d}"


def as_int(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
    return None


def build_review_obj(review):
    if isinstance(review, str):
        text = review.strip()
        if not text:
            return None
        return {"IS_META_REVIEW": False, "comments": text, "TITLE": "ReviewHub Review"}

    if not isinstance(review, dict):
        return None

    text = first_present(
        review,
        [
            "comments",
            "comment",
            "review",
            "review_text",
            "text",
            "content",
            "body",
            "summary",
        ],
        "",
    )
    if not isinstance(text, str) or not text.strip():
        return None

    out = {"IS_META_REVIEW": False, "comments": text.strip(), "TITLE": "ReviewHub Review"}

    rating = as_int(first_present(review, ["rating", "score", "recommendation"]))
    confidence = as_int(first_present(review, ["confidence", "reviewer_confidence"]))
    if rating is not None:
        out["RECOMMENDATION"] = rating
    if confidence is not None:
        out["REVIEWER_CONFIDENCE"] = confidence
    return out


def extract_reviews_from_paper_json(paper_json_str: str) -> list:
    """Extract peer reviews from the nested paper_json field."""
    if not paper_json_str:
        return []
    try:
        pj = json.loads(paper_json_str)
    except Exception:
        return []
    reviews = []
    for rev in pj.get("reviews", []):
        content = rev.get("content", {})
        if not isinstance(content, dict):
            continue
        text = content.get("review") or content.get("comments") or content.get("comment", "")
        if not isinstance(text, str) or not text.strip():
            continue
        out = {"IS_META_REVIEW": False, "comments": text.strip(), "TITLE": "ICLR Review"}
        rating = as_int(content.get("rating"))
        confidence = as_int(content.get("confidence"))
        if rating is not None:
            out["RECOMMENDATION"] = rating
        if confidence is not None:
            out["REVIEWER_CONFIDENCE"] = confidence
        reviews.append(out)
    return reviews


def normalize_row(row, split, idx, conference_label):
    paper_id = sanitize_id(
        first_present(row, ["file_id", "paper_id", "forum", "id", "submission_id"]),
        split,
        idx,
    )

    title    = first_present(row, ["title", "paper_title"], "")
    abstract = first_present(row, ["abstract", "paper_abstract"], "")
    authors  = first_present(row, ["authors", "author_list"], [])
    keywords = first_present(row, ["keywords", "topics"], [])

    if not isinstance(authors, list):
        authors = [str(authors)] if authors else []
    if not isinstance(keywords, list):
        keywords = [str(keywords)] if keywords else []

    # Accept/reject from decision_bucket (always present in ReviewHub)
    bucket = first_present(row, ["decision_bucket"], None)
    if isinstance(bucket, str):
        accepted = bucket.strip().lower() == "accept"
    else:
        # fallback: infer from decision_text
        decision_text = first_present(row, ["decision_text", "decision"], "") or ""
        lower = decision_text.lower()
        if "accept" in lower:
            accepted = True
        elif "reject" in lower:
            accepted = False
        else:
            accepted = None

    # Real reviews from paper_json
    reviews = extract_reviews_from_paper_json(row.get("paper_json", ""))

    # paper_markdown stored for eval use (no docling needed)
    paper_markdown = row.get("paper_markdown", "")

    record = {
        "title":          title if isinstance(title, str) else str(title),
        "abstract":       abstract if isinstance(abstract, str) else str(abstract),
        "id":             paper_id,
        "authors":        authors,
        "accepted":       accepted,
        "keywords":       keywords,
        "conference":     conference_label,
        "reviews":        reviews,
        "paper_markdown": paper_markdown,
        "histories":      [],
    }
    return paper_id, record


def main():
    parser = argparse.ArgumentParser(description="Integrate ReviewHub/ICLR dataset")
    parser.add_argument(
        "--output-dir",
        default="output/iclr_reviewhub",
        help="Directory to write normalized data (default: output/iclr_reviewhub)",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated ReviewHub configs (default: all available ICLR years)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per split")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    reviews_dir = out_root / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = get_dataset_config_names("ReviewHub/ICLR")

    split_counts = {}
    review_count = 0
    accepted_counter = Counter()
    seen_ids = set()

    for config in configs:
        ds = load_dataset("ReviewHub/ICLR", config)
        conference_label = f"ICLR {config.replace('iclr', '')} (ReviewHub)"

        for split_name, split_ds in ds.items():
            split_key = f"{config}:{split_name}"
            n = len(split_ds) if args.limit is None else min(len(split_ds), args.limit)
            split_counts[split_key] = n

            for idx in range(n):
                row = split_ds[idx]
                paper_id, record = normalize_row(row, split_key, idx, conference_label)

                if paper_id in seen_ids:
                    paper_id = f"{paper_id}_{config}_{split_name}_{idx}"
                    record["id"] = paper_id
                seen_ids.add(paper_id)

                review_count += len(record["reviews"])
                accepted_counter[str(record.get("accepted"))] += 1

                with open(reviews_dir / f"{paper_id}.json", "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False)

    manifest = {
        "source": "huggingface://ReviewHub/ICLR",
        "output_dir": str(out_root),
        "split_counts": split_counts,
        "total_papers": sum(split_counts.values()),
        "total_reviews": review_count,
        "accepted_breakdown": dict(accepted_counter),
    }

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
