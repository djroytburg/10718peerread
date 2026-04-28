"""
Zero-shot acceptance prediction on ICLR ReviewHub data.

Mirrors run_zero_shot_eval.py but reads from output/iclr_reviewhub/reviews/
where paper_markdown is pre-stored (no docling needed).

Usage:
  python run_zero_shot_eval_iclr.py --model llama
  python run_zero_shot_eval_iclr.py --model deepseek
  python run_zero_shot_eval_iclr.py --model gemma
  python run_zero_shot_eval_iclr.py --model llama --with-reviews
  python run_zero_shot_eval_iclr.py --year 2023   # restrict to one year
  python run_zero_shot_eval_iclr.py --plot         # regenerate plots only

Requires integrate_reviewhub_iclr.py to have been run first.
Re-run it to fix accepted labels and reviews:
  python integrate_reviewhub_iclr.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils.bedrock import MODEL_PRICING, Usage, converse_text, get_bedrock_client
from run_zero_shot_eval import (
    build_prompt, parse_response, sample_reviews,
    compute_metrics, has_real_confidence,
    roc_from_confidence, auc_from_confidence, accept_scores,
    bootstrap_ci, _fpr, _tpr, _f1,
    MODELS, MODEL_SIZE_ORDER, FONT_DIR, VOLKHOV, UBUNTU_MONO, UBUNTU_MONO_B,
)

DATASET      = "iclr_reviewhub"
REVIEWS_DIR  = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR  = BASE_DIR / "results"
REVIEW_SEED  = 10718

N_PER_CLASS  = 113   # balanced per class; mirrors NeurIPS eval size
FEW_SHOT_HOLDOUT = 3


# ---------------------------------------------------------------------------
# Paper selection
# ---------------------------------------------------------------------------

def load_paper_pool(year: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    """Return (accepts, rejects) as lists of {paper_id, path} dicts."""
    accepts, rejects = [], []
    for p in sorted(REVIEWS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        if year and str(year) not in d.get("conference", ""):
            continue
        if not d.get("paper_markdown", "").strip():
            continue
        acc = d.get("accepted")
        if acc is True:
            accepts.append({"paper_id": p.stem, "path": p})
        elif acc is False:
            rejects.append({"paper_id": p.stem, "path": p})
    return accepts, rejects


def get_paper_entries(year: Optional[str] = None) -> list[dict]:
    """Return up to N_PER_CLASS accepts + N_PER_CLASS rejects, seeded."""
    accepts, rejects = load_paper_pool(year)
    holdouts = set(sorted(r["paper_id"] for r in rejects)[-FEW_SHOT_HOLDOUT:])
    rejects  = [r for r in rejects if r["paper_id"] not in holdouts]

    rng = random.Random(10718)
    rng.shuffle(accepts)
    rng.shuffle(rejects)
    n = min(N_PER_CLASS, len(accepts), len(rejects))
    chosen = accepts[:n] + rejects[:n]
    print(f"ICLR pool: {len(accepts)} accepts, {len(rejects)} rejects "
          f"(year={year or 'all'}) → using {n}A + {n}R = {2*n} papers")
    return chosen


def get_ground_truth(review_json: dict) -> str:
    return "ACCEPT" if review_json.get("accepted") else "REJECT"


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------

def results_file(model_key: str, with_reviews: bool, year: Optional[str] = None) -> Path:
    tag  = "with_reviews" if with_reviews else "no_reviews"
    yr   = f"_{year}" if year else ""
    return RESULTS_DIR / f"zero_shot_iclr{yr}_{model_key}_{tag}.json"


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
    print(f"  n={n}  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Acc={(tp+tn)/n:.3f}  FPR={fpr:.3f}  TPR={tpr:.3f}")
    p = MODEL_PRICING.get(model_id)
    if p:
        cost = usage.input_tokens/1000*p["input"] + usage.output_tokens/1000*p["output"]
        print(f"  Cost: ${cost:.3f}  (${cost/n:.4f}/paper)")


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run(model_key: str, with_reviews: bool, year: Optional[str] = None) -> None:
    model_id = MODELS[model_key]
    entries  = get_paper_entries(year)
    out_file = results_file(model_key, with_reviews, year)

    if out_file.exists():
        existing    = json.loads(out_file.read_text())
        results     = existing.get("results", [])
        total_usage = Usage(**existing.get("usage", {}))
        done        = {r["paper_id"] for r in results}
        print(f"Resuming {out_file.name} — {len(done)}/{len(entries)} done.")
    else:
        results, total_usage, done = [], Usage(), set()

    client = get_bedrock_client()
    n = len(entries)

    for i, entry in enumerate(entries):
        paper_id = entry["paper_id"]
        if paper_id in done:
            continue

        review_json  = json.loads(entry["path"].read_text())
        paper_md     = review_json.get("paper_markdown", "")
        ground_truth = get_ground_truth(review_json)
        reviews_text = sample_reviews(review_json) if with_reviews else None

        if not paper_md.strip():
            print(f"[{i+1}/{n}] {paper_id}: no markdown, skipping")
            continue

        prompt = build_prompt(paper_md, reviews_text)
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
            "model":        model_id,
            "model_key":    model_key,
            "dataset":      DATASET,
            "year":         year,
            "with_reviews": with_reviews,
            "usage":        dataclasses.asdict(total_usage),
            "results":      results,
        }, indent=2))

    print(f"\n=== {out_file.name} ===")
    print_metrics(results, model_id, total_usage)


# ---------------------------------------------------------------------------
# Plotting (reuses NeurIPS plot structure)
# ---------------------------------------------------------------------------

def make_plots(year: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    from matplotlib.patches import Patch

    for ttf in Path(FONT_DIR).rglob("*.ttf"):
        fm.fontManager.addfont(str(ttf))
    volkhov = fm.FontProperties(fname=VOLKHOV)
    mono    = fm.FontProperties(fname=UBUNTU_MONO)
    mono_b  = fm.FontProperties(fname=UBUNTU_MONO_B)

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
        "axes.edgecolor": "#cccccc", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "white", "grid.linewidth": 1.0,
    })

    yr_tag   = f"_{year}" if year else ""
    yr_label = f" {year}" if year else " (all years)"
    fig_dir  = RESULTS_DIR / "figures" / "baselines"
    fig_dir.mkdir(parents=True, exist_ok=True)

    MODEL_META = {
        "gemma":    ("Gemma 3 12B",   "#55A868"),
        "llama":    ("Llama 3.3 70B", "#4C72B0"),
        "deepseek": ("DeepSeek V3.2", "#DD8452"),
    }

    no_rev, with_rev = [], []
    for key in MODEL_SIZE_ORDER:
        label, color = MODEL_META[key]
        p_no = results_file(key, False, year)
        p_wr = results_file(key, True,  year)
        if p_no.exists():
            no_rev.append((label, color, json.loads(p_no.read_text()).get("results", [])))
        if p_wr.exists():
            with_rev.append((label, color, json.loads(p_wr.read_text()).get("results", [])))

    if not no_rev:
        print("No ICLR zero-shot results found yet.")
        return

    # ROC plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1, ls="--", zorder=1)
    for label, color, results in no_rev:
        if not results:
            continue
        m       = compute_metrics(results)
        f1_lo, f1_hi   = bootstrap_ci(results, _f1)
        fpr_c, tpr_c   = roc_from_confidence(results)
        ax.plot(fpr_c, tpr_c, color=color, lw=2, zorder=3,
                label=f"{label}  F1={m['f1']:.2f} [{f1_lo:.2f}-{f1_hi:.2f}]")
        flo, fhi = bootstrap_ci(results, _fpr)
        tlo, thi = bootstrap_ci(results, _tpr)
        ax.errorbar(m["fpr"], m["tpr"],
                    xerr=[[m["fpr"]-flo],[fhi-m["fpr"]]],
                    yerr=[[m["tpr"]-tlo],[thi-m["tpr"]]],
                    fmt="o", color=color, ms=7, capsize=4, lw=1.5, zorder=4)
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontproperties=mono, fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontproperties=mono, fontsize=11)
    ax.set_title(f"Zero-Shot — ICLR{yr_label}\nNo Reviews",
                 fontproperties=volkhov, fontsize=13, pad=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(mono); lbl.set_fontsize(9)
    ax.legend(prop=mono, fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, zorder=0)
    fig.tight_layout()
    out = fig_dir / f"roc_iclr{yr_tag}_no_reviews.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Bar chart (condition split)
    wr_by_label = {lbl: res for lbl, _, res in with_rev if res}
    entries_no  = [(lbl, col, res) for lbl, col, res in no_rev if res]
    n_mod = len(entries_no)
    x, w  = np.arange(n_mod), 0.32

    conditions = [("No Reviews", entries_no)]
    if wr_by_label:
        conditions.append(("With Reviews",
                           [(lbl, col, wr_by_label[lbl])
                            for lbl, col, _ in entries_no if lbl in wr_by_label]))

    fig, axes = plt.subplots(1, len(conditions),
                              figsize=(3.2*n_mod*len(conditions)+1.5, 5.5),
                              sharey=True, gridspec_kw={"wspace": 0.06})
    if len(conditions) == 1:
        axes = [axes]

    for ax, (cond_title, entries) in zip(axes, conditions):
        for i, (label, _, results) in enumerate(entries):
            for xoff, fn, color, ec in [
                (-w/2, _fpr, "#C44E52", "#7a1a1d"),
                ( w/2, _tpr, "#4C72B0", "#1a3a6e"),
            ]:
                val = fn(results)
                lo, hi = bootstrap_ci(results, fn)
                ax.bar(x[i]+xoff, val, w, color=color, alpha=0.85, zorder=3)
                ax.errorbar(x[i]+xoff, val, yerr=[[val-lo],[hi-val]],
                            fmt="none", color="black", capsize=4, lw=1.5, zorder=4)
                ax.text(x[i]+xoff, hi+0.03, f"{val:.2f}",
                        ha="center", fontproperties=mono, fontsize=8.5, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl,_,_ in entries], fontproperties=mono, fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_title(cond_title, fontproperties=mono, fontsize=11, pad=8)
        ax.grid(axis="y", zorder=0)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(mono); tick.set_fontsize(9)
        ax.legend(handles=[
            Patch(color="#C44E52", alpha=0.85, label="FPR (lower is better)"),
            Patch(color="#4C72B0", alpha=0.85, label="TPR (higher is better)"),
        ], prop=mono, fontsize=8.5, framealpha=0.9, edgecolor="#cccccc")

    axes[0].set_ylabel("Rate  [95% bootstrap CI]", fontproperties=mono, fontsize=11)
    n_label = f"n={len(entries_no[0][2])}" if entries_no else ""
    fig.suptitle(f"Zero-Shot — ICLR{yr_label}  ·  {n_label}  ·  balanced A/R",
                 fontproperties=volkhov, fontsize=13, y=1.01)
    fig.tight_layout()
    out = fig_dir / f"fpr_tpr_iclr{yr_tag}_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------
    table_dir = RESULTS_DIR / "tables" / "baselines"
    table_dir.mkdir(parents=True, exist_ok=True)

    all_entries = [(lbl, "no",  res) for lbl, _, res in no_rev   if res] + \
                  [(lbl, "yes", res) for lbl, _, res in with_rev if res]
    rows = []
    for label, wr, results in all_entries:
        m        = compute_metrics(results)
        flo, fhi = bootstrap_ci(results, _fpr)
        tlo, thi = bootstrap_ci(results, _tpr)
        alo, ahi = bootstrap_ci(results, _f1)
        rows.append((label, wr, m["n"], m["acc"], m["fpr"], flo, fhi,
                     m["tpr"], tlo, thi, m["f1"], alo, ahi))
    rows.sort(key=lambda r: r[4])

    yr_heading = f"ICLR{yr_label}"
    lines = [
        f"# Zero-Shot Baseline Comparison — {yr_heading}, n={rows[0][2] if rows else '?'} (balanced A/R)\n",
        "> 95% bootstrap CIs (2000 resamples, seed=10718)\n",
        "| Model | Reviews | n | Acc | FPR (lower) | 95% CI | TPR (higher) | 95% CI | F1 | 95% CI |",
        "|---|:---:|---|---|---|---|---|---|---|---|",
    ]
    for (label, wr, n, acc, fpr, flo, fhi, tpr, tlo, thi, f1, alo, ahi) in rows:
        lines.append(
            f"| {label} | {wr} | {n} | {acc:.3f} "
            f"| {fpr:.3f} | [{flo:.3f}, {fhi:.3f}] "
            f"| {tpr:.3f} | [{tlo:.3f}, {thi:.3f}] "
            f"| {f1:.3f} | [{alo:.3f}, {ahi:.3f}] |"
        )
    table_path = table_dir / f"baselines_iclr{yr_tag}.md"
    table_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {table_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--with-reviews", action="store_true")
    parser.add_argument("--year",  default=None, help="e.g. 2023 — restrict to one ICLR year")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    if args.plot:
        make_plots(args.year)
        return

    if not args.model:
        parser.error("--model is required unless --plot is passed")

    run(args.model, args.with_reviews, args.year)
    make_plots(args.year)


if __name__ == "__main__":
    main()
