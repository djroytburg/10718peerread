"""
Few-shot composition ablation: sweep R/A balance from 5R/0A to 0R/5A.

Pool:    5 reject + 5 accept from neurips_2023 (full text, seed=42)
Test:    configurable per-class sample size (default 20A + 20R)
Setting: no reviews, deepseek.v3.2

Compositions tested: (5,0) (4,1) (3,2) (2,3) (1,4) (0,5)

Usage:
  python run_few_shot_ablation.py
  python run_few_shot_ablation.py --comp 3 2
  python run_few_shot_ablation.py --n-test-per-class 20 --sampling confidence-stratified
"""

import argparse
import dataclasses
import json
import random
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from convert_json_to_markdown import json_to_markdown
from utils.bedrock import Usage, get_bedrock_client
from utils.judge_rigs import AcceptanceRig, make_few_shot_examples

DATASET      = "neurips_2025_full"
ANON_PDFS    = BASE_DIR / "output" / DATASET / "anonymized_pdfs"
REVIEWS_DIR  = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR  = BASE_DIR / "results"
REFERENCE    = RESULTS_DIR / "llama3.3_70b_instruct_balanced.json"
FEW_SHOT_SRC = BASE_DIR / "output" / "neurips_2023"

DEEPSEEK_V3 = "deepseek.v3.2"
TEST_SEED   = 10718
N_TEST_PER_CLASS = 20
POOL_SIZE   = 5   # per class in the few-shot pool
DEFAULT_CONFIDENCE_SOURCE = "debate_jury_with_reviews_46papers_neurips_2025_full.json"

COMPOSITIONS = [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)]


def _parse_comp_key(key: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)R(\d+)A", key)
    if not match:
        raise ValueError(f"Invalid composition key: {key}")
    return int(match.group(1)), int(match.group(2))


def _resolve_results_path(path_or_name: str) -> Path:
    p = Path(path_or_name)
    return p if p.is_absolute() else RESULTS_DIR / p


def _stratified_confidence_sample(rows: list[dict], n: int) -> list[dict]:
    """
    Confidence-stratified sampling:
    sort by confidence and pick one representative from each equal-width rank bin.
    """
    if n > len(rows):
        raise ValueError(f"Requested n={n} but only {len(rows)} rows available")
    ordered = sorted(rows, key=lambda r: (float(r["confidence"]), r["paper_id"]))
    picked: list[dict] = []
    m = len(ordered)
    for i in range(n):
        start = int(i * m / n)
        end = int((i + 1) * m / n)
        idx = (start + max(start, end - 1)) // 2
        picked.append(ordered[idx])
    return picked


def get_test_papers(
    n_per_class: int,
    sampling: str,
    confidence_source: str,
) -> list[dict]:
    ref = json.loads(REFERENCE.read_text())
    rng = random.Random(TEST_SEED)
    accepts = [r for r in ref["results"] if r["ground_truth"] == "ACCEPT"]
    rejects = [r for r in ref["results"] if r["ground_truth"] == "REJECT"]

    if sampling == "random":
        return (
            rng.sample(accepts, min(n_per_class, len(accepts))) +
            rng.sample(rejects, min(n_per_class, len(rejects)))
        )

    conf_path = _resolve_results_path(confidence_source)
    conf_data = json.loads(conf_path.read_text())
    conf_rows = [
        {
            "paper_id": r["paper_id"],
            "ground_truth": r["ground_truth"],
            "confidence": float(r["confidence"]),
        }
        for r in conf_data.get("results", [])
        if r.get("prediction") in ("ACCEPT", "REJECT")
        and r.get("ground_truth") in ("ACCEPT", "REJECT")
        and isinstance(r.get("confidence"), (int, float))
    ]

    conf_accepts = [r for r in conf_rows if r["ground_truth"] == "ACCEPT"]
    conf_rejects = [r for r in conf_rows if r["ground_truth"] == "REJECT"]
    if len(conf_accepts) < n_per_class or len(conf_rejects) < n_per_class:
        raise ValueError(
            f"Not enough confidence-scored papers in {conf_path.name}: "
            f"{len(conf_accepts)} accepts, {len(conf_rejects)} rejects, "
            f"requested {n_per_class} per class."
        )

    sampled = _stratified_confidence_sample(conf_accepts, n_per_class)
    sampled += _stratified_confidence_sample(conf_rejects, n_per_class)
    return sampled


def get_ground_truth(review_json: dict) -> str:
    val = review_json.get("accepted")
    if isinstance(val, bool):
        return "ACCEPT" if val else "REJECT"
    return "ACCEPT" if str(val).lower() in ("true", "accept", "yes", "1") else "REJECT"


def comp_key(n_r: int, n_a: int) -> str:
    return f"{n_r}R{n_a}A"


def print_metrics(label: str, results: list[dict]) -> None:
    evaluated = [r for r in results if r["prediction"] is not None]
    if not evaluated:
        print(f"  [{label}] No results.")
        return
    tp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n = len(evaluated)
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    acc = (tp + tn) / n
    print(f"  [{label}] n={n}  TP={tp} FP={fp} TN={tn} FN={fn}  "
          f"Acc={acc:.3f}  FPR={fpr:.3f}  TPR={tpr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", nargs=2, type=int, metavar=("N_REJECT", "N_ACCEPT"),
                        help="Run a single composition, e.g. --comp 3 2")
    parser.add_argument("--n-test-per-class", type=int, default=N_TEST_PER_CLASS,
                        help=f"Number of ACCEPT and REJECT test papers each (default: {N_TEST_PER_CLASS})")
    parser.add_argument("--sampling", choices=["random", "confidence-stratified"],
                        default="confidence-stratified",
                        help="How to choose test papers")
    parser.add_argument("--confidence-source", type=str, default=DEFAULT_CONFIDENCE_SOURCE,
                        help="Results file containing per-paper confidence for confidence-stratified sampling")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default derived from n and sampling)")
    args = parser.parse_args()

    compositions = [tuple(args.comp)] if args.comp else COMPOSITIONS

    if args.output:
        out_file = _resolve_results_path(args.output)
    else:
        out_file = RESULTS_DIR / (
            f"few_shot_ablation_dsv3_neurips_2025_full_"
            f"n{args.n_test_per_class * 2}_{args.sampling}.json"
        )

    # Load existing results
    if out_file.exists():
        saved = json.loads(out_file.read_text())
        all_results = saved.get("compositions", {})
        total_usage = Usage(**saved.get("usage", {}))
        print(f"Resuming from {out_file.name} — "
              f"completed: {list(all_results.keys())}")
    else:
        all_results, total_usage = {}, Usage()

    # Build pool: exactly POOL_SIZE reject + POOL_SIZE accept (full text, no truncation)
    print(f"\nLoading few-shot pool ({POOL_SIZE}R + {POOL_SIZE}A) from neurips_2023...")
    pool_examples = make_few_shot_examples(
        FEW_SHOT_SRC, n=POOL_SIZE * 2, balanced=True, seed=42, max_paragraphs=None
    )
    reject_pool = [e for e in pool_examples if e["label"] == "REJECT"]
    accept_pool = [e for e in pool_examples if e["label"] == "ACCEPT"]
    print(f"  Pool: {len(reject_pool)}R  {len(accept_pool)}A")

    # Test set
    test_papers = get_test_papers(
        n_per_class=args.n_test_per_class,
        sampling=args.sampling,
        confidence_source=args.confidence_source,
    )
    print(f"Test set: {len(test_papers)} papers  "
          f"({sum(1 for r in test_papers if r['ground_truth']=='ACCEPT')}A  "
          f"{sum(1 for r in test_papers if r['ground_truth']=='REJECT')}R)")

    client = get_bedrock_client()

    for n_r, n_a in compositions:
        key = comp_key(n_r, n_a)
        if key in all_results and len(all_results[key]) == len(test_papers):
            print(f"\n[{key}] already complete ({len(all_results[key])} results), skipping.")
            print_metrics(key, all_results[key])
            continue

        print(f"\n{'='*50}")
        print(f"Composition {key}: {n_r} reject + {n_a} accept examples")

        # Build few-shot list: take first n_r from reject pool, n_a from accept pool
        # Shuffle with a fixed seed so order is consistent but not trivially sorted
        few_shot: list[dict] = []
        if n_r > 0:
            few_shot += reject_pool[:n_r]
        if n_a > 0:
            few_shot += accept_pool[:n_a]
        rng = random.Random(42)
        rng.shuffle(few_shot)

        rig = AcceptanceRig(
            with_reviews=False,
            conference="NeurIPS 2025",
            max_output_tokens=64,
            few_shot_examples=few_shot,
            few_shot_label=key,  # ensures each composition gets a unique cache hash
        )

        # Cost estimate from first test paper
        sample_id = test_papers[0]["paper_id"]
        sample_pdf = ANON_PDFS / f"{sample_id}.pdf.json"
        sample_md  = json_to_markdown(json.loads(sample_pdf.read_text()))
        _, est = rig(None, sample_md, estimate_only=True)
        print(f"  Est. tokens/paper: {est.input_tokens:,}  "
              f"cost: ${est.cost(DEEPSEEK_V3):.4f}/paper  "
              f"total: ${est.cost(DEEPSEEK_V3)*len(test_papers):.3f}")

        done_ids = {r["paper_id"] for r in all_results.get(key, [])}
        results = list(all_results.get(key, []))

        for i, ref_entry in enumerate(test_papers):
            paper_id = ref_entry["paper_id"]
            if paper_id in done_ids:
                continue

            pdf_path    = ANON_PDFS / f"{paper_id}.pdf.json"
            review_path = REVIEWS_DIR / f"{paper_id}.json"
            if not pdf_path.exists() or not review_path.exists():
                print(f"  [{i+1}/{len(test_papers)}] {paper_id}: missing, skipping")
                continue

            paper_md     = json_to_markdown(json.loads(pdf_path.read_text()))
            ground_truth = get_ground_truth(json.loads(review_path.read_text()))

            pred, usage = rig(client, paper_md, model_id=DEEPSEEK_V3,
                              cache_config={"dataset": DATASET, "paper_id": paper_id})
            total_usage += usage
            correct = pred == ground_truth if pred is not None else None
            results.append({"paper_id": paper_id, "prediction": pred,
                            "ground_truth": ground_truth, "correct": correct})

            print(f"  [{i+1}/{len(test_papers)}] {paper_id}: "
                  f"pred={pred} gt={ground_truth} ✓={correct}")

            all_results[key] = results
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps({
                "model": DEEPSEEK_V3,
                "pool_size": POOL_SIZE,
                "few_shot_source": "neurips_2023",
                "test_sampling": args.sampling,
                "confidence_source": args.confidence_source if args.sampling == "confidence-stratified" else None,
                "test_n": len(test_papers),
                "test_n_per_class": args.n_test_per_class,
                "usage": dataclasses.asdict(total_usage),
                "compositions": all_results,
            }, indent=2))

        print_metrics(key, results)

    # Final summary table
    print(f"\n{'='*60}")
    print(f"{'Comp':<8} {'n_ex':<6} {'Acc':>6} {'FPR':>6} {'TPR':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print("-" * 60)
    for n_r, n_a in COMPOSITIONS:
        key = comp_key(n_r, n_a)
        if key not in all_results:
            continue
        res = [r for r in all_results[key] if r["prediction"] is not None]
        if not res:
            continue
        tp = sum(1 for r in res if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
        fp = sum(1 for r in res if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
        tn = sum(1 for r in res if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
        fn = sum(1 for r in res if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
        n = len(res)
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{key:<8} {n_r+n_a:<6} {(tp+tn)/n:>6.3f} {fpr:>6.3f} {tpr:>6.3f} "
              f"{tp:>4} {fp:>4} {tn:>4} {fn:>4}")

    print(f"\nTotal cost: {total_usage.summary(DEEPSEEK_V3)}")


if __name__ == "__main__":
    main()
