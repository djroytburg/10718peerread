"""
Few-shot 3R/2A evaluation on the canonical 46-paper balanced test set.

One model plays the judge for all papers. Examples and test papers are
truncated to ~max_paragraphs (default 40, roughly 2-3K tokens each).
Confidence is elicited alongside the binary prediction.
No reviews given for the paper under evaluation.

Usage:
  python run_few_shot_3r2a.py --model llama
  python run_few_shot_3r2a.py --model deepseek
  python run_few_shot_3r2a.py --model gemma

  # Default: in-distribution examples from neurips_2025_full (eval papers excluded)
  python run_few_shot_3r2a.py --model deepseek --few-shot-source output/neurips_2025_full

  # Out-of-distribution examples from another year/venue
  python run_few_shot_3r2a.py --model deepseek --few-shot-source output/neurips_2023
  python run_few_shot_3r2a.py --model deepseek --few-shot-source output/neurips_2024
  python run_few_shot_3r2a.py --model deepseek --few-shot-source output/iclr_reviewhub

  python run_few_shot_3r2a.py --model deepseek --max-paragraphs 60
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from convert_json_to_markdown import json_to_markdown
from utils.bedrock import MODEL_PRICING, Usage, converse_text, get_bedrock_client
from utils.judge_rigs import make_few_shot_examples, _truncate_paper_md

DATASET     = "neurips_2025_full"
ANON_PDFS   = BASE_DIR / "output" / DATASET / "anonymized_pdfs"
REVIEWS_DIR = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR = BASE_DIR / "results"
REFERENCE   = RESULTS_DIR / "llama3.3_70b_instruct_balanced.json"
DEFAULT_FEW_SHOT_SRC = BASE_DIR / "output" / "neurips_2025_full"

MODELS = {
    "llama":    "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek": "deepseek.v3.2",
    "gemma":    "google.gemma-3-12b-it",
}
DEFAULT_MAX_PARAGRAPHS = 30


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _paper_block(paper_md: str) -> str:
    return (
        "---------------- BEGIN PAPER ----------------\n"
        f"{paper_md}\n"
        "----------------- END PAPER -----------------\n"
    )


def _reviews_block(reviews_text: str) -> str:
    return (
        "Official Reviews (meta-reviews excluded):\n"
        "---------------- BEGIN REVIEWS ----------------\n"
        f"{reviews_text}\n"
        "----------------- END REVIEWS -----------------\n"
    )


def format_reviews(review_json: dict) -> str:
    """Format non-meta reviews into a single string."""
    parts = []
    for i, r in enumerate(review_json.get("reviews", []), 1):
        if r.get("IS_META_REVIEW"):
            continue
        text = r.get("comments", "").strip()
        if not text:
            continue
        rec  = r.get("RECOMMENDATION")
        conf = r.get("REVIEWER_CONFIDENCE")
        header = f"--- Review {i}"
        if rec is not None:
            header += f"  rating={rec}"
        if conf is not None:
            header += f"  confidence={conf}"
        header += " ---"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_prompt(paper_md: str, examples: list[dict]) -> str:
    task = (
        "You are an expert NeurIPS 2025 area chair.\n\n"
        "Task:\n"
        "- Read the following anonymized NeurIPS 2025 paper in Markdown form.\n"
        "- Based ONLY on the paper content and quality, predict whether it "
        "was accepted to NeurIPS 2025.\n"
        "- The labeled examples below include their official peer reviews "
        "to help you calibrate what accepted vs rejected papers look like.\n"
        "- You do not know the true decision for the paper you are evaluating; "
        "you must make your best judgment.\n\n"
        "Output format — respond with ONLY a JSON object, nothing else:\n"
        '  {"prediction": "ACCEPT", "confidence": 0.85}\n'
        "or\n"
        '  {"prediction": "REJECT", "confidence": 0.72}\n\n'
        "confidence is your certainty (0.0 = completely uncertain, 1.0 = certain).\n\n"
    )
    if examples:
        task += "Here are labeled example papers (with their reviews) to guide your judgment:\n\n"
        for i, ex in enumerate(examples, 1):
            task += f"=== EXAMPLE {i} ===\n"
            task += "Paper Markdown:\n"
            task += _paper_block(ex["paper_md"])
            if ex.get("reviews_text"):
                task += _reviews_block(ex["reviews_text"])
            task += f'\nCorrect answer: {{"prediction": "{ex["label"]}"}}\n\n'
        task += "=== PAPER TO EVALUATE (no reviews provided) ===\n"
    task += "Paper Markdown:\n"
    task += _paper_block(paper_md)
    return task


def parse_response(text: str) -> tuple[Optional[str], float]:
    m = re.search(
        r'\{\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*,\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)\s*\}',
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), min(1.0, max(0.0, float(m.group(2))))
    m2 = re.search(
        r'\{\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)\s*,\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*\}',
        text, re.IGNORECASE,
    )
    if m2:
        return m2.group(2).upper(), min(1.0, max(0.0, float(m2.group(1))))
    upper = text.upper()
    if "ACCEPT" in upper and "REJECT" not in upper:
        return "ACCEPT", 0.5
    if "REJECT" in upper and "ACCEPT" not in upper:
        return "REJECT", 0.5
    return None, 0.5


# ---------------------------------------------------------------------------
# Paper selection
# ---------------------------------------------------------------------------

def get_paper_ids() -> list[str]:
    ref = json.loads(REFERENCE.read_text())
    accepts = [r["paper_id"] for r in ref["results"] if r["ground_truth"] == "ACCEPT"][:23]
    rejects = [r["paper_id"] for r in ref["results"] if r["ground_truth"] == "REJECT"]
    return accepts + rejects


def get_ground_truth(review_json: dict) -> str:
    val = review_json.get("accepted")
    if isinstance(val, bool):
        return "ACCEPT" if val else "REJECT"
    return "ACCEPT" if str(val).lower() in ("true", "yes", "1", "accept") else "REJECT"


def results_file(model_key: str, max_paragraphs: int, source_tag: str) -> Path:
    return RESULTS_DIR / f"few_shot_3r2a_{model_key}_{source_tag}_trunc{max_paragraphs}_46papers_neurips_2025_full.json"


def print_metrics(results: list[dict], model_id: str, usage: Usage) -> None:
    ev  = [r for r in results if r["prediction"] is not None]
    if not ev:
        return
    tp  = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp  = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn  = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn  = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n   = len(ev)
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1  = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else float("nan")
    print(f"  n={n}  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Acc={(tp+tn)/n:.3f}  FPR={fpr:.3f}  TPR={tpr:.3f}  F1={f1:.3f}")
    p = MODEL_PRICING.get(model_id)
    if p:
        cost = usage.input_tokens / 1000 * p["input"] + usage.output_tokens / 1000 * p["output"]
        print(f"  Cost: ${cost:.3f}  (${cost/n:.4f}/paper)")


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def load_examples(few_shot_src: Path, max_paragraphs: int,
                  exclude_ids: set) -> list[dict]:
    """
    Load a fixed 3R/2A exemplar set from few_shot_src.

    Rejects: the last 3 (by sorted ID) from the pool not in exclude_ids.
    Accepts: the first 2 (by sorted ID) from the pool not in exclude_ids.
    Same result for every model — deterministic, no random sampling.
    """
    pdf_dir = few_shot_src / "anonymized_pdfs"
    rev_dir = few_shot_src / "reviews"
    rng     = random.Random(10718)   # used only for paragraph truncation
    has_pdfs = pdf_dir.exists()

    rejects, accepts = [], []
    for rp in sorted(rev_dir.glob("*.json")):
        pid = rp.stem
        if pid in exclude_ids:
            continue
        rv  = json.loads(rp.read_text())
        if rv.get("accepted") is None:
            continue
        # require either a separate PDF json OR inline paper_markdown
        pdf = pdf_dir / f"{pid}.pdf.json" if has_pdfs else None
        if has_pdfs and not pdf.exists():
            continue
        if not has_pdfs and not rv.get("paper_markdown", "").strip():
            continue
        label = "ACCEPT" if rv.get("accepted") else "REJECT"
        (accepts if label == "ACCEPT" else rejects).append((pid, pdf))

    # take last 3 rejects and first 2 accepts (sorted order = deterministic)
    chosen_r = rejects[-3:]
    chosen_a = accepts[:2]
    if len(chosen_r) < 3 or len(chosen_a) < 2:
        raise ValueError(
            f"Not enough examples in {few_shot_src} after exclusions: "
            f"{len(rejects)} rejects, {len(accepts)} accepts available"
        )

    examples = []
    for pid, pdf, label in [(p, f, "REJECT") for p, f in chosen_r] + \
                            [(p, f, "ACCEPT") for p, f in chosen_a]:
        rev_path = rev_dir / f"{pid}.json"
        rev_json = json.loads(rev_path.read_text()) if rev_path.exists() else {}

        if pdf is not None and pdf.exists():
            # standard path: separate anonymized PDF json
            doc      = json.loads(pdf.read_text())
            paper_md = json_to_markdown(doc)
        elif rev_json.get("paper_markdown"):
            # reviewhub path: markdown stored directly in review json
            paper_md = rev_json["paper_markdown"]
        else:
            continue

        if max_paragraphs:
            paper_md = _truncate_paper_md(paper_md, max_paragraphs, rng)
        reviews_text = format_reviews(rev_json) if rev_json else None
        examples.append({"paper_id": pid, "paper_md": paper_md,
                          "label": label, "reviews_text": reviews_text})
    rng.shuffle(examples)
    return examples


def run(model_key: str, max_paragraphs: int, few_shot_src: Path) -> None:
    model_id   = MODELS[model_key]
    paper_ids  = get_paper_ids()
    eval_ids   = set(paper_ids)
    source_tag = few_shot_src.name          # e.g. "neurips_2025_full" or "neurips_2023"
    out_file   = results_file(model_key, max_paragraphs, source_tag)

    in_dist = few_shot_src.resolve() == (BASE_DIR / "output" / DATASET).resolve()
    print(f"Loading 3R/2A examples from {few_shot_src.name} "
          f"({'in-distribution' if in_dist else 'out-of-distribution'}, "
          f"max_paragraphs={max_paragraphs})...")
    examples = load_examples(few_shot_src, max_paragraphs,
                             exclude_ids=eval_ids if in_dist else set())
    print(f"  {sum(1 for e in examples if e['label']=='REJECT')}R "
          f"{sum(1 for e in examples if e['label']=='ACCEPT')}A examples loaded")

    if out_file.exists():
        existing    = json.loads(out_file.read_text())
        results     = existing.get("results", [])
        total_usage = Usage(**existing.get("usage", {}))
        done        = {r["paper_id"] for r in results}
        print(f"Resuming {out_file.name} — {len(done)}/{len(paper_ids)} done.")
    else:
        results, total_usage, done = [], Usage(), set()

    client = get_bedrock_client()
    n      = len(paper_ids)
    trunc_rng = random.Random(10718)

    for i, paper_id in enumerate(paper_ids):
        if paper_id in done:
            continue

        pdf_path    = ANON_PDFS / f"{paper_id}.pdf.json"
        review_path = REVIEWS_DIR / f"{paper_id}.json"
        if not pdf_path.exists() or not review_path.exists():
            print(f"[{i+1}/{n}] {paper_id}: missing files, skipping")
            continue

        raw_md       = json_to_markdown(json.loads(pdf_path.read_text()))
        paper_md     = _truncate_paper_md(raw_md, max_paragraphs, trunc_rng)
        ground_truth = get_ground_truth(json.loads(review_path.read_text()))

        prompt = build_prompt(paper_md, examples)
        text, usage = converse_text(client, prompt, model_id=model_id, max_tokens=64)
        prediction, confidence = parse_response(text or "")

        total_usage += usage
        correct = prediction == ground_truth if prediction is not None else None
        results.append({
            "paper_id":     paper_id,
            "prediction":   prediction,
            "confidence":   confidence,
            "ground_truth": ground_truth,
            "correct":      correct,
        })
        print(
            f"[{i+1}/{n}] {paper_id}: pred={prediction}({confidence:.2f}) "
            f"gt={ground_truth} ✓={correct} | "
            f"in={usage.input_tokens:,} out={usage.output_tokens}"
        )

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps({
            "model":           model_id,
            "model_key":       model_key,
            "few_shot":        "3R2A",
            "few_shot_source": str(few_shot_src),
            "max_paragraphs":  max_paragraphs,
            "with_reviews":    False,
            "usage":           dataclasses.asdict(total_usage),
            "results":         results,
        }, indent=2))

    print(f"\n=== {out_file.name} ===")
    print_metrics(results, model_id, total_usage)
    return out_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--max-paragraphs", type=int, default=DEFAULT_MAX_PARAGRAPHS,
                        help=f"Truncate papers to this many paragraphs "
                             f"(default: {DEFAULT_MAX_PARAGRAPHS}, ~2-3K tokens each)")
    parser.add_argument("--few-shot-source", type=Path, default=DEFAULT_FEW_SHOT_SRC,
                        help="Path to dataset directory to draw examples from "
                             f"(default: {DEFAULT_FEW_SHOT_SRC}, in-distribution). "
                             "Examples: output/neurips_2023  output/neurips_2024  output/iclr_reviewhub")
    args = parser.parse_args()
    run(args.model, args.max_paragraphs, args.few_shot_source)


if __name__ == "__main__":
    main()
