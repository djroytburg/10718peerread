"""
Few-shot AcceptanceRig eval on 10 accept + 10 reject from the canonical test set.

Two variants:
  --mode balanced   : 5 accept + 5 reject examples from neurips_2023
  --mode reject-heavy: 8 reject + 2 accept examples from neurips_2023

Model: deepseek.v3.2 (matches the baseline we're comparing against)

Usage:
  python run_few_shot_eval.py --mode balanced
  python run_few_shot_eval.py --mode reject-heavy
"""

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from convert_json_to_markdown import json_to_markdown
from utils.bedrock import Usage, get_bedrock_client
from utils.judge_rigs import AcceptanceRig, format_reviews, make_few_shot_examples

DATASET      = "neurips_2025_full"
ANON_PDFS    = BASE_DIR / "output" / DATASET / "anonymized_pdfs"
REVIEWS_DIR  = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR  = BASE_DIR / "results"
REFERENCE    = RESULTS_DIR / "llama3.3_70b_instruct_balanced.json"
FEW_SHOT_SRC = BASE_DIR / "output" / "neurips_2023"  # different dataset for examples

DEEPSEEK_V3 = "deepseek.v3.2"
TEST_SEED   = 10718
N_TEST      = 10  # per class


def get_test_papers() -> list[dict]:
    """10 accept + 10 reject from the canonical test set."""
    ref = json.loads(REFERENCE.read_text())
    rng = random.Random(TEST_SEED)
    accepts = [r for r in ref["results"] if r["ground_truth"] == "ACCEPT"]
    rejects = [r for r in ref["results"] if r["ground_truth"] == "REJECT"]
    return rng.sample(accepts, N_TEST) + rng.sample(rejects, min(N_TEST, len(rejects)))


def get_ground_truth(review_json: dict) -> str:
    val = review_json.get("accepted")
    if isinstance(val, bool):
        return "ACCEPT" if val else "REJECT"
    return "ACCEPT" if str(val).lower() in ("true", "accept", "yes", "1") else "REJECT"


def print_metrics(results: list[dict]) -> None:
    evaluated = [r for r in results if r["prediction"] is not None]
    if not evaluated:
        print("  No results."); return
    tp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n = len(evaluated)
    print(f"  Evaluated: {n}/{len(results)}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy={(tp+tn)/n:.3f}  FPR={fp/(fp+tn) if fp+tn else float('nan'):.3f}  TPR={tp/(tp+fn) if tp+fn else float('nan'):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["balanced", "reject-heavy"], required=True)
    args = parser.parse_args()

    if args.mode == "balanced":
        n_examples, n_reject_ex, n_accept_ex = 10, 5, 5
    else:
        n_examples, n_reject_ex, n_accept_ex = 10, 8, 2

    out_file = RESULTS_DIR / f"few_shot_{args.mode}_dsv3_neurips_2025_full.json"

    print(f"Loading {n_examples} few-shot examples from neurips_2023 "
          f"({n_reject_ex} reject, {n_accept_ex} accept)...")

    # Build custom balanced/reject-heavy example set from neurips_2023
    all_examples_balanced = make_few_shot_examples(
        FEW_SHOT_SRC, n=20, balanced=True, seed=42
    )
    rejects_ex = [e for e in all_examples_balanced if e["label"] == "REJECT"]
    accepts_ex = [e for e in all_examples_balanced if e["label"] == "ACCEPT"]

    rng = random.Random(42)
    few_shot = (rng.sample(rejects_ex, min(n_reject_ex, len(rejects_ex))) +
                rng.sample(accepts_ex, min(n_accept_ex, len(accepts_ex))))
    rng.shuffle(few_shot)
    print(f"  Got {sum(1 for e in few_shot if e['label']=='REJECT')} reject, "
          f"{sum(1 for e in few_shot if e['label']=='ACCEPT')} accept examples")

    rig = AcceptanceRig(with_reviews=False, conference="NeurIPS 2025",
                        max_output_tokens=64, few_shot_examples=few_shot)

    test_papers = get_test_papers()
    print(f"Test papers: {len(test_papers)} "
          f"({sum(1 for r in test_papers if r['ground_truth']=='ACCEPT')} accept, "
          f"{sum(1 for r in test_papers if r['ground_truth']=='REJECT')} reject)")

    # Cost estimate
    sample_pdf = ANON_PDFS / f"{test_papers[0]['paper_id']}.pdf.json"
    sample_md  = json_to_markdown(json.loads(sample_pdf.read_text()))
    _, est = rig(None, sample_md, estimate_only=True)
    print(f"Estimated cost/paper: ${est.cost(DEEPSEEK_V3):.4f}  total: ${est.cost(DEEPSEEK_V3)*len(test_papers):.3f}")

    if out_file.exists():
        existing = json.loads(out_file.read_text())
        results = existing.get("results", [])
        total_usage = Usage(**existing.get("usage", {}))
        done = {r["paper_id"] for r in results}
        print(f"Resuming — {len(done)}/{len(test_papers)} done.")
    else:
        results, total_usage, done = [], Usage(), set()

    client = get_bedrock_client()

    for i, ref_entry in enumerate(test_papers):
        paper_id = ref_entry["paper_id"]
        if paper_id in done:
            continue

        pdf_path    = ANON_PDFS / f"{paper_id}.pdf.json"
        review_path = REVIEWS_DIR / f"{paper_id}.json"
        if not pdf_path.exists() or not review_path.exists():
            print(f"[{i+1}/{len(test_papers)}] {paper_id}: missing, skipping")
            continue

        paper_md     = json_to_markdown(json.loads(pdf_path.read_text()))
        ground_truth = get_ground_truth(json.loads(review_path.read_text()))

        pred, usage = rig(client, paper_md, model_id=DEEPSEEK_V3,
                          cache_config={"dataset": DATASET, "paper_id": paper_id})
        total_usage += usage
        correct = pred == ground_truth if pred is not None else None
        results.append({"paper_id": paper_id, "prediction": pred,
                        "ground_truth": ground_truth, "correct": correct})

        print(f"[{i+1}/{len(test_papers)}] {paper_id}: pred={pred} gt={ground_truth} ✓={correct}")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps({
            "mode": args.mode,
            "model": DEEPSEEK_V3,
            "n_few_shot": len(few_shot),
            "few_shot_source": "neurips_2023",
            "usage": dataclasses.asdict(total_usage),
            "results": results,
        }, indent=2))

    print(f"\n=== Few-shot ({args.mode}) ===")
    print_metrics(results)
    print(f"Total usage: {total_usage.summary(DEEPSEEK_V3)}")


if __name__ == "__main__":
    main()
