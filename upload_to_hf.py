#!/usr/bin/env python3
"""
Build and upload NeurIPS 2023-2025 peer-review dataset to HuggingFace.

Creates one JSONL file per year under data/ then uploads the whole folder
in a single commit using upload_large_folder (avoids per-file rate limits).

Each row contains:
  paper_id, year, conference, accepted, title, abstract, keywords,
  pdf_url, paper_text (reconstructed from anonymized PDF),
  reviews (list of review dicts, non-meta only)
"""

import json
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID   = "djroytburg/NeurIPS-2023-2025"
REPO_TYPE = "dataset"
REPO      = Path(__file__).resolve().parent
OUT_DIR   = REPO / "hf_data"

CORPORA = {
    "2023": REPO / "output/neurips_2023",
    "2024": REPO / "output/neurips_2024",
    "2025": REPO / "output/neurips_2025_full",
}

TEXT_LABELS = {"section_header", "text", "title", "list_item"}


def extract_text(anon_path: Path):
    """Reconstruct plain text from a DocLing anonymized JSON."""
    if not anon_path.exists():
        return None
    try:
        doc = json.loads(anon_path.read_text())
        parts = [
            t["text"].strip()
            for t in doc.get("texts", [])
            if t.get("label") in TEXT_LABELS and t.get("text", "").strip()
        ]
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None


def build_year(year, base):
    reviews_dir = base / "reviews"
    anon_dir    = base / "anonymized_pdfs"
    out_file    = OUT_DIR / f"{year}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(reviews_dir.glob("*.json"))
    print(f"{year}: building from {len(files)} review files...", flush=True)
    expected_conf = f"NeurIPS {year}"

    written = 0
    with out_file.open("w") as fout:
        for f in files:
            d = json.loads(f.read_text())
            # skip any files that ended up in the wrong year directory
            if d.get("conference") and d["conference"] != expected_conf:
                continue
            pid = d.get("id") or f.stem

            # Reconstruct paper text from anonymized PDF
            anon_path = anon_dir / f"{pid}.pdf.json"
            paper_text = extract_text(anon_path)

            reviews = [
                r for r in d.get("reviews", [])
                if not r.get("IS_META_REVIEW")
            ]

            row = {
                "paper_id":   pid,
                "year":       int(year),
                "conference": d.get("conference", f"NeurIPS {year}"),
                "accepted":   d.get("accepted"),
                "title":      d.get("title"),
                "abstract":   d.get("abstract"),
                "keywords":   d.get("keywords", []),
                "pdf_url":    f"https://openreview.net/pdf?id={pid}",
                "paper_text": paper_text,
                "reviews":    reviews,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"  wrote {written} rows -> {out_file}", flush=True)
    return out_file


# ---------------------------------------------------------------------------
# Build all years
# ---------------------------------------------------------------------------

for year, base in CORPORA.items():
    build_year(year, base)

print("\nAll JSONL files built. Uploading via upload_large_folder...", flush=True)

api = HfApi()
api.upload_large_folder(
    folder_path=str(OUT_DIR),
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print(f"\nDone. https://huggingface.co/datasets/{REPO_ID}")
