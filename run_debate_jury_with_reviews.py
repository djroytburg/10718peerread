"""
Debate jury WITH real reviews seeding round-1 reviewer arguments.
Same 46 papers, same reviewer/AC models, but each reviewer gets
a randomly sampled real review as their opening-argument seed.

Usage:
  python run_debate_jury_with_reviews.py --shard 0 --n-shards 4
  python run_debate_jury_with_reviews.py --merge
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
from utils.judge_rigs import DebateRig

DATASET      = "neurips_2025_full"
ANON_PDFS    = BASE_DIR / "output" / DATASET / "anonymized_pdfs"
REVIEWS_DIR  = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "debate_jury_with_reviews_46papers_neurips_2025_full.json"
REFERENCE    = RESULTS_DIR / "llama3.3_70b_instruct_balanced.json"

MAGISTRAL   = "mistral.magistral-small-2509"
GEMMA_12B   = "google.gemma-3-12b-it"
LLAMA_70B   = "us.meta.llama3-3-70b-instruct-v1:0"
DEEPSEEK_V3 = "deepseek.v3.2"
REVIEW_SEED = 10718


def make_rig(reviews: list[dict]) -> DebateRig:
    return DebateRig(
        n_reviewers=3,
        reviewer_models=[MAGISTRAL, GEMMA_12B, LLAMA_70B],
        ac_model=DEEPSEEK_V3,
        n_rounds=2,
        ac_sees="both",
        conference="NeurIPS 2025",
        persona_seed=10718,
        reviews=reviews,
    )


def get_paper_ids() -> list[str]:
    ref = json.loads(REFERENCE.read_text())
    accepts = [r["paper_id"] for r in ref["results"] if r["ground_truth"] == "ACCEPT"][:23]
    rejects = [r["paper_id"] for r in ref["results"] if r["ground_truth"] == "REJECT"]
    return accepts + rejects


def get_ground_truth(review_json: dict) -> str:
    val = review_json.get("accepted")
    if isinstance(val, bool):
        return "ACCEPT" if val else "REJECT"
    if isinstance(val, str):
        return "ACCEPT" if val.lower() in ("true", "accept", "yes", "1") else "REJECT"
    return "ACCEPT" if val else "REJECT"


def sample_reviews(review_json: dict, n: int = 3, seed: int = REVIEW_SEED) -> list[dict]:
    """Sample n non-meta reviews; fewer if not enough available."""
    non_meta = [r for r in review_json.get("reviews", []) if not r.get("IS_META_REVIEW")]
    rng = random.Random(seed)
    return rng.sample(non_meta, min(n, len(non_meta)))


def print_metrics(results: list[dict]) -> None:
    evaluated = [r for r in results if r["prediction"] is not None]
    if not evaluated:
        print("  No evaluated results yet.")
        return
    tp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n = len(evaluated)
    print(f"  Evaluated: {n}/{len(results)}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy={(tp+tn)/n:.3f}  FPR={fp/(fp+tn) if fp+tn else float('nan'):.3f}  TPR={tp/(tp+fn) if tp+fn else float('nan'):.3f}")


def shard_file(idx: int) -> Path:
    return RESULTS_DIR / f"debate_jury_with_reviews_46papers_shard{idx}.json"


def save(path: Path, results: list[dict], usage: Usage, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rig": "DebateRig+reviews",
        "config": config,
        "usage": dataclasses.asdict(usage),
        "results": results,
    }, indent=2))


def run(paper_ids: list[str], out_file: Path) -> None:
    if out_file.exists():
        existing = json.loads(out_file.read_text())
        results = existing.get("results", [])
        total_usage = Usage(**existing.get("usage", {}))
        done = {r["paper_id"] for r in results}
        print(f"Resuming {out_file.name} — {len(done)}/{len(paper_ids)} done.")
    else:
        results, total_usage, done = [], Usage(), set()

    client = get_bedrock_client()
    n = len(paper_ids)
    config = {"reviewers": [MAGISTRAL, GEMMA_12B, LLAMA_70B],
              "ac_model": DEEPSEEK_V3, "n_rounds": 2, "with_reviews": True, "review_seed": REVIEW_SEED}

    for i, paper_id in enumerate(paper_ids):
        if paper_id in done:
            continue

        pdf_path    = ANON_PDFS / f"{paper_id}.pdf.json"
        review_path = REVIEWS_DIR / f"{paper_id}.json"
        if not pdf_path.exists() or not review_path.exists():
            print(f"[{i+1}/{n}] {paper_id}: missing files, skipping")
            continue

        review_json  = json.loads(review_path.read_text())
        paper_md     = json_to_markdown(json.loads(pdf_path.read_text()))
        ground_truth = get_ground_truth(review_json)
        sampled_reviews = sample_reviews(review_json, n=3)

        rig = make_rig(sampled_reviews)
        pred, conf, _, usage = rig(
            client, paper_md,
            cache_config={"dataset": DATASET, "paper_id": paper_id},
        )
        total_usage += usage
        correct = pred == ground_truth if pred is not None else None
        results.append({"paper_id": paper_id, "prediction": pred,
                        "confidence": conf, "ground_truth": ground_truth, "correct": correct})

        print(f"[{i+1}/{n}] {paper_id}: pred={pred}({conf:.2f}) gt={ground_truth} ✓={correct} | "
              f"in={usage.input_tokens:,} out={usage.output_tokens:,}")
        save(out_file, results, total_usage, config)

    print(f"\n=== Summary: {out_file.name} ===")
    print_metrics(results)
    print(f"Tokens — in={total_usage.input_tokens:,}  out={total_usage.output_tokens:,}  calls={total_usage.calls}")


def merge() -> None:
    shard_files = sorted(RESULTS_DIR.glob("debate_jury_with_reviews_46papers_shard*.json"))
    all_results, total_usage = [], Usage()
    for sf in shard_files:
        d = json.loads(sf.read_text())
        all_results.extend(d.get("results", []))
        total_usage += Usage(**d.get("usage", {}))
        print(f"  Merged {sf.name}: {len(d.get('results',[]))} papers")
    seen, deduped = set(), []
    for r in all_results:
        if r["paper_id"] not in seen:
            deduped.append(r); seen.add(r["paper_id"])
    config = {"reviewers": [MAGISTRAL, GEMMA_12B, LLAMA_70B], "ac_model": DEEPSEEK_V3,
              "n_rounds": 2, "with_reviews": True, "review_seed": REVIEW_SEED}
    save(RESULTS_FILE, deduped, total_usage, config)
    print(f"\n=== Merged {len(deduped)} papers -> {RESULTS_FILE.name} ===")
    print_metrics(deduped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge(); return

    all_ids = get_paper_ids()
    if args.shard is not None:
        paper_ids = all_ids[args.shard::args.n_shards]
        out_file = shard_file(args.shard)
        print(f"Shard {args.shard}/{args.n_shards}: {len(paper_ids)} papers")
    else:
        paper_ids, out_file = all_ids, RESULTS_FILE
        print(f"Papers: {len(paper_ids)}")

    run(paper_ids, out_file)


if __name__ == "__main__":
    main()
