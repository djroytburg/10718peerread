#!/usr/bin/env python3
"""Download and normalize ReviewHub/ICLR data from Hugging Face.

Example:
    python scrape_iclr_hf.py \
      --dataset ReviewHub/ICLR \
      --config iclr2018 \
      --split all \
      --output-dir output/iclr_2018_hf

This script writes:
- <output-dir>/papers/<paper_id>.json
- <output-dir>/reviews/<paper_id>.json
- <output-dir>/manifest.json

Notes:
- The user mentioned Kaggle, but the provided URL points to Hugging Face.
- The dataset schema may evolve, so this script uses robust field matching.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            yield text
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def _deep_get(obj: Dict[str, Any], candidates: List[str]) -> Any:
    """Get the first non-empty value for candidate dotted paths."""
    for path in candidates:
        cur: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if not ok:
            continue
        if cur is None:
            continue
        if isinstance(cur, str) and not cur.strip():
            continue
        return cur
    return None


def _coerce_accept_label(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if not v:
            return None

        # Common explicit buckets from review datasets.
        if "accept" in v and "reject" not in v:
            return True
        if "reject" in v or "desk" in v:
            return False

        if v in {"accept", "accepted", "true", "yes", "y", "1"}:
            return True
        if v in {"reject", "rejected", "false", "no", "n", "0"}:
            return False
    return None


def _format_review_list(raw_reviews: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_reviews, str):
        try:
            raw_reviews = json.loads(raw_reviews)
        except json.JSONDecodeError:
            raw_reviews = [raw_reviews]

    if isinstance(raw_reviews, dict):
        raw_reviews = [raw_reviews]

    if not isinstance(raw_reviews, list):
        return []

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_reviews, 1):
        if isinstance(item, dict):
            title = _normalize_text(
                item.get("title")
                or item.get("TITLE")
                or item.get("reviewer_title")
                or f"Review {idx}"
            )
            comments = _normalize_text(
                item.get("comments")
                or item.get("review")
                or item.get("text")
                or item.get("content")
            )
            if not comments:
                comments = "\n\n".join(_iter_strings(item))
            out.append(
                {
                    "TITLE": title or f"Review {idx}",
                    "comments": comments,
                    "IS_META_REVIEW": bool(
                        item.get("is_meta_review") or item.get("IS_META_REVIEW")
                    ),
                }
            )
        else:
            text = _normalize_text(item)
            if text:
                out.append(
                    {
                        "TITLE": f"Review {idx}",
                        "comments": text,
                        "IS_META_REVIEW": False,
                    }
                )
    return out


def _extract_record(record: Dict[str, Any], idx: int, split_name: str) -> Tuple[Optional[bool], Dict[str, Any], Dict[str, Any]]:
    paper_id_raw = _deep_get(
        record,
        [
            "paper_id",
            "id",
            "submission_id",
            "openreview_id",
            "forum",
            "paper.id",
        ],
    )
    paper_id = _normalize_text(paper_id_raw)
    if not paper_id:
        paper_id = f"{split_name}_{idx:07d}"

    # Sanitize to filesystem-safe ID.
    paper_id = re.sub(r"[^a-zA-Z0-9._-]", "_", paper_id)

    markdown = _deep_get(
        record,
        [
            # Explicitly prioritize the provided schema field.
            "paper_markdown",
            "markdown",
            "paper.markdown",
            "paper_md",
            "document.markdown",
            "content.markdown",
            "paper_text",
            "full_text",
            "text",
            "paper.abstract",
            "abstract",
        ],
    )
    markdown_text = _normalize_text(markdown)

    title = _normalize_text(
        _deep_get(
            record,
            ["title", "paper.title", "submission.title", "name"],
        )
    )

    accepted_raw = _deep_get(
        record,
        [
            # Explicitly prioritize the provided schema fields.
            "decision_bucket",
            "decision_text",
            "accepted",
            "is_accepted",
            "label",
            "decision",
            "final_decision",
            "verdict",
            "meta_review.decision",
        ],
    )
    accepted = _coerce_accept_label(accepted_raw)

    reviews_raw = _deep_get(
        record,
        [
            "reviewer_scores_json",
            "reviews",
            "paper.reviews",
            "peer_reviews",
            "all_reviews",
            "messages",
            "generated_review",
        ],
    )
    reviews = _format_review_list(reviews_raw)

    paper_doc = {
        "paper_id": paper_id,
        "title": title,
        "markdown": markdown_text,
        "source_split": split_name,
    }

    review_doc = {
        "paper_id": paper_id,
        "accepted": accepted,
        "accepted_raw": accepted_raw,
        "reviews": reviews,
        "source_split": split_name,
    }

    return accepted, paper_doc, review_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ICLR dataset from Hugging Face")
    parser.add_argument("--dataset", default="ReviewHub/ICLR", help="HF dataset name")
    parser.add_argument("--config", default="iclr2025", help="HF dataset config")
    parser.add_argument(
        "--split",
        default="all",
        help="Dataset split (e.g., train/validation/test) or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        default="output/iclr_2025_hf",
        help="Output directory",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap on records processed (0 means no cap)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    output_dir = Path(args.output_dir)
    papers_dir = output_dir / "papers"
    reviews_dir = output_dir / "reviews"
    papers_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        ds_obj = load_dataset(args.dataset, args.config)
        if not hasattr(ds_obj, "keys"):
            raise SystemExit("Expected a DatasetDict when --split all is used")
        split_map = {split_name: ds_obj[split_name] for split_name in ds_obj.keys()}
    else:
        split_map = {args.split: load_dataset(args.dataset, args.config, split=args.split)}

    processed = 0
    skipped_no_markdown = 0
    skipped_no_label = 0

    manifest_entries: List[Dict[str, Any]] = []

    for split_name, dataset_split in split_map.items():
        for idx, record in enumerate(dataset_split):
            accepted, paper_doc, review_doc = _extract_record(record, idx, split_name)

            # Keep only rows that can be evaluated.
            if not paper_doc["markdown"]:
                skipped_no_markdown += 1
                continue
            if accepted is None:
                skipped_no_label += 1
                continue

            paper_id = paper_doc["paper_id"]
            with (papers_dir / f"{paper_id}.json").open("w", encoding="utf-8") as f:
                json.dump(paper_doc, f, ensure_ascii=False)
            with (reviews_dir / f"{paper_id}.json").open("w", encoding="utf-8") as f:
                json.dump(review_doc, f, ensure_ascii=False)

            manifest_entries.append(
                {
                    "paper_id": paper_id,
                    "split": split_name,
                    "accepted": bool(accepted),
                }
            )
            processed += 1

            if args.max_records > 0 and processed >= args.max_records:
                break
        if args.max_records > 0 and processed >= args.max_records:
            break

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "processed": processed,
        "skipped_no_markdown": skipped_no_markdown,
        "skipped_no_label": skipped_no_label,
        "papers_dir": str(papers_dir),
        "reviews_dir": str(reviews_dir),
        "entries": manifest_entries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Finished scraping ICLR dataset from Hugging Face")
    print(f"Output dir: {output_dir}")
    print(f"Processed: {processed}")
    print(f"Skipped (no markdown): {skipped_no_markdown}")
    print(f"Skipped (no label): {skipped_no_label}")


if __name__ == "__main__":
    main()
