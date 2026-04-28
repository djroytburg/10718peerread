"""
Self-debate (single-model) ablation on the canonical 46-paper test set.

One model plays all three reviewer roles AND the AC. Persona assignments
are still deterministic (persona_seed=10718), so the same model is prompted
with [critical, neutral, critical] personas across its three reviewer slots.

Usage:
  python run_self_debate.py --model llama
  python run_self_debate.py --model deepseek
  python run_self_debate.py --model llama    --with-reviews
  python run_self_debate.py --model deepseek --with-reviews
  python run_self_debate.py --model llama    --shard 0 --n-shards 4
  python run_self_debate.py --model llama    --merge
  python run_self_debate.py --model llama    --with-reviews --merge
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
from utils.bedrock import MODEL_PRICING, Usage, get_bedrock_client
from utils.judge_rigs import DebateRig

DATASET     = "neurips_2025_full"
ANON_PDFS   = BASE_DIR / "output" / DATASET / "anonymized_pdfs"
REVIEWS_DIR = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR = BASE_DIR / "results"
REFERENCE   = RESULTS_DIR / "llama3.3_70b_instruct_balanced.json"

MODELS = {
    "llama":    "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek": "deepseek.v3.2",
    "gemma":    "google.gemma-3-12b-it",
}
REVIEW_SEED = 10718


def make_rig(model_id: str, reviews: list = None) -> DebateRig:
    return DebateRig(
        n_reviewers=3,
        reviewer_models=[model_id],   # cycled → all 3 slots use same model
        ac_model=model_id,
        n_rounds=2,
        ac_sees="both",
        conference="NeurIPS 2025",
        persona_seed=10718,
        reviews=reviews,
    )


def sample_reviews(review_json: dict, n: int = 3, seed: int = REVIEW_SEED) -> list:
    """Sample up to n non-meta reviews."""
    non_meta = [r for r in review_json.get("reviews", []) if not r.get("IS_META_REVIEW")]
    rng = random.Random(seed)
    return rng.sample(non_meta, min(n, len(non_meta)))


def _tag(with_reviews: bool) -> str:
    return "with_reviews" if with_reviews else "no_reviews"


def results_file(model_key: str, with_reviews: bool) -> Path:
    return RESULTS_DIR / f"self_debate_{model_key}_{_tag(with_reviews)}_46papers_neurips_2025_full.json"


def shard_file(model_key: str, with_reviews: bool, shard_idx: int) -> Path:
    return RESULTS_DIR / f"self_debate_{model_key}_{_tag(with_reviews)}_46papers_shard{shard_idx}.json"


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


def print_metrics(results: list[dict], model_id: str, usage: Usage) -> None:
    evaluated = [r for r in results if r["prediction"] is not None]
    if not evaluated:
        print("  No evaluated results yet.")
        return
    tp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp = sum(1 for r in evaluated if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn = sum(1 for r in evaluated if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n  = len(evaluated)
    acc = (tp + tn) / n
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    print(f"  n={n}  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy={acc:.3f}  FPR={fpr:.3f}  TPR={tpr:.3f}")

    pricing = MODEL_PRICING.get(model_id)
    if pricing:
        cost = (usage.input_tokens / 1000 * pricing["input"] +
                usage.output_tokens / 1000 * pricing["output"])
        print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out  "
              f"({usage.calls} calls)")
        print(f"  Total cost: ${cost:.3f}   Per paper: ${cost/n:.4f}")
    else:
        print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out  "
              f"(pricing unknown for {model_id})")


def save(path: Path, model_key: str, model_id: str, rig: DebateRig,
         results: list, usage: Usage, with_reviews: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rig":     "DebateRig",
        "variant": f"self_debate_{model_key}",
        "config": {
            "model":        model_id,
            "reviewers":    [model_id, model_id, model_id],
            "ac_model":     model_id,
            "n_rounds":     rig.n_rounds,
            "ac_sees":      rig.ac_sees,
            "personas":     rig._personas,
            "with_reviews": with_reviews,
            "review_seed":  REVIEW_SEED if with_reviews else None,
        },
        "usage":   dataclasses.asdict(usage),
        "results": results,
    }, indent=2))


def run(model_key: str, paper_ids: list, out_file: Path, with_reviews: bool) -> None:
    model_id = MODELS[model_key]
    _rig_no_reviews = make_rig(model_id)  # used only for save() metadata

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

        if with_reviews:
            reviews = sample_reviews(review_json)
            rig = make_rig(model_id, reviews=reviews)
        else:
            rig = make_rig(model_id)

        pred, conf, _, usage = rig(
            client, paper_md,
            cache_config={"dataset": DATASET, "paper_id": paper_id},
        )
        total_usage += usage
        correct = pred == ground_truth if pred is not None else None
        results.append({
            "paper_id":     paper_id,
            "prediction":   pred,
            "confidence":   conf,
            "ground_truth": ground_truth,
            "correct":      correct,
        })

        print(
            f"[{i+1}/{n}] {paper_id}: pred={pred}({conf:.2f}) gt={ground_truth} "
            f"✓={correct} | in={usage.input_tokens:,} out={usage.output_tokens} "
            f"calls={usage.calls}"
        )
        save(out_file, model_key, model_id, rig, results, total_usage, with_reviews)

    print(f"\n=== {out_file.name} ===")
    print_metrics(results, model_id, total_usage)


def merge(model_key: str, with_reviews: bool) -> None:
    tag = _tag(with_reviews)
    shard_files = sorted(RESULTS_DIR.glob(f"self_debate_{model_key}_{tag}_46papers_shard*.json"))
    if not shard_files:
        print(f"No shard files found for model={model_key} {tag}.")
        return
    all_results, total_usage = [], Usage()
    for sf in shard_files:
        d = json.loads(sf.read_text())
        all_results.extend(d.get("results", []))
        total_usage += Usage(**d.get("usage", {}))
        print(f"  Merged {sf.name}: {len(d['results'])} papers")
    seen, deduped = set(), []
    for r in all_results:
        if r["paper_id"] not in seen:
            deduped.append(r)
            seen.add(r["paper_id"])
    model_id = MODELS[model_key]
    rig = make_rig(model_id)
    out = results_file(model_key, with_reviews)
    save(out, model_key, model_id, rig, deduped, total_usage, with_reviews)
    print(f"\n=== Merged {len(deduped)} papers -> {out.name} ===")
    print_metrics(deduped, model_id, total_usage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True,
                        help="Model for all roles: llama | deepseek")
    parser.add_argument("--with-reviews", action="store_true",
                        help="Seed reviewer round-1 prompts with real reviews (meta-reviews stripped)")
    parser.add_argument("--shard",    type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--merge",    action="store_true")
    args = parser.parse_args()
    wr = args.with_reviews

    if args.merge:
        merge(args.model, wr)
        return

    all_ids = get_paper_ids()

    if args.shard is not None:
        slices    = [all_ids[i::args.n_shards] for i in range(args.n_shards)]
        paper_ids = slices[args.shard]
        out_file  = shard_file(args.model, wr, args.shard)
        print(f"Shard {args.shard}/{args.n_shards}: {len(paper_ids)} papers  "
              f"model={args.model}  with_reviews={wr}")
    else:
        paper_ids = all_ids
        out_file  = results_file(args.model, wr)
        print(f"Papers: {len(paper_ids)}  model={args.model}  with_reviews={wr}")

    run(args.model, paper_ids, out_file, wr)


if __name__ == "__main__":
    main()
