"""
Zero-shot acceptance prediction on the canonical 46-paper balanced test set.
Elicits a confidence score alongside the binary prediction.
Runs for Llama 3.3 70B, DeepSeek V3.2, and Gemma 3 12B, with and without reviews.
Meta-reviews (IS_META_REVIEW=True) are always stripped when reviews are used.

After all six result files exist, generates:
  results/figures/baselines/  — ROC-space scatter + FPR/TPR bar chart
  results/tables/baselines/   — markdown comparison table

Usage:
  python run_zero_shot_eval.py --model llama
  python run_zero_shot_eval.py --model deepseek
  python run_zero_shot_eval.py --model gemma
  python run_zero_shot_eval.py --model llama --with-reviews
  python run_zero_shot_eval.py --plot          # regenerate plots/tables only
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
from utils.judge_rigs import STRATEGIES

DATASET     = "neurips_2025_full"
ANON_PDFS   = BASE_DIR / "output" / DATASET / "anonymized_pdfs"
REVIEWS_DIR = BASE_DIR / "output" / DATASET / "reviews"
RESULTS_DIR = BASE_DIR / "results"
REFERENCE   = RESULTS_DIR / "llama3.3_70b_instruct_balanced.json"
REVIEW_SEED = 10718

MODELS = {
    "llama":    "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek": "deepseek.v3.2",
    "gemma":    "google.gemma-3-12b-it",
}

# Model display order: smallest to largest (for bar chart x-axis)
MODEL_SIZE_ORDER = ["gemma", "llama", "deepseek"]

N_PER_CLASS = 113   # max rejects with parsed PDFs after holdout (116 total - 3 holdouts)
                    # increase to 150 after running docling on remaining reject PDFs
FEW_SHOT_HOLDOUT = 3  # rejects reserved for few-shot examples, excluded from eval
MIN_N = 20          # skip partial runs in plots/tables

FONT_DIR     = Path("~/fonts").expanduser()
VOLKHOV      = str(FONT_DIR / "Volkhov" / "Volkhov-Bold.ttf")
UBUNTU_MONO  = str(FONT_DIR / "Ubuntu_Mono" / "UbuntuMono-Regular.ttf")
UBUNTU_MONO_B= str(FONT_DIR / "Ubuntu_Mono" / "UbuntuMono-Bold.ttf")

# STRATEGIES dict moved to utils/judge_rigs.py (imported above)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(paper_md: str, reviews_text: Optional[str] = None,
                 strategy: str = "neutral") -> str:
    task = (
        "You are an expert NeurIPS 2025 area chair.\n\n"
        "Task:\n"
        "- Read the following anonymized NeurIPS 2025 paper in Markdown form.\n"
    )
    if reviews_text:
        task += (
            "- You are also given the official peer reviews for this submission "
            "(meta-reviews and author responses have been excluded).\n"
            "- Based on the paper AND the reviews, predict whether it was accepted.\n"
        )
    else:
        task += (
            "- Based ONLY on the paper content and quality, predict whether it "
            "was accepted to NeurIPS 2025.\n"
        )
    task += "- You do not know the true decision; you must make your best judgment.\n\n"
    policy = STRATEGIES.get(strategy, "")
    if policy:
        task += policy
    task += (
        "Output format — respond with ONLY a JSON object, nothing else:\n"
        '  {"prediction": "ACCEPT", "confidence": 0.85}\n'
        "or\n"
        '  {"prediction": "REJECT", "confidence": 0.72}\n\n'
        "confidence is your certainty (0.0 = completely uncertain, 1.0 = certain).\n\n"
        "Paper:\n"
        "---------------- BEGIN PAPER ----------------\n"
        f"{paper_md}\n"
        "----------------- END PAPER -----------------\n"
    )
    if reviews_text:
        task += (
            "\nPeer Reviews:\n"
            "---------------- BEGIN REVIEWS ----------------\n"
            f"{reviews_text}\n"
            "----------------- END REVIEWS -----------------\n"
        )
    return task


def parse_response(text: str) -> tuple[Optional[str], float]:
    """Return (prediction, confidence) from model output."""
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
    # fallback: prediction only
    upper = text.upper()
    if "ACCEPT" in upper and "REJECT" not in upper:
        return "ACCEPT", 0.5
    if "REJECT" in upper and "ACCEPT" not in upper:
        return "REJECT", 0.5
    return None, 0.5


def sample_reviews(review_json: dict, n: int = 3) -> str:
    """Return formatted non-meta reviews (up to n), empty string if none."""
    non_meta = [r for r in review_json.get("reviews", []) if not r.get("IS_META_REVIEW")]
    rng = random.Random(REVIEW_SEED)
    chosen = rng.sample(non_meta, min(n, len(non_meta)))
    if not chosen:
        return ""
    parts = []
    for i, r in enumerate(chosen, 1):
        parts.append(f"--- Review {i} ---")
        parts.append(r.get("comments", "").strip())
        parts.append("")
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def get_paper_ids() -> list[str]:
    """
    Return up to N_PER_CLASS accepts + N_PER_CLASS rejects from neurips_2025_full,
    seeded deterministically. The canonical 46-paper set (23A/23R) is always
    included so existing predictions are never overwritten.
    """
    pdf_ids = {p.stem.replace(".pdf", "")
               for p in (BASE_DIR / "output" / DATASET / "anonymized_pdfs").glob("*.pdf.json")}

    accepts_pool, rejects_pool = [], []
    for rp in sorted(REVIEWS_DIR.glob("*.json")):
        pid = rp.stem
        if pid not in pdf_ids:
            continue
        rv = json.loads(rp.read_text())
        if rv.get("accepted"):
            accepts_pool.append(pid)
        else:
            rejects_pool.append(pid)

    # Canonical 46: first 23 accepts + all 23 rejects from the reference file
    ref = json.loads(REFERENCE.read_text())
    canonical_accepts = [r["paper_id"] for r in ref["results"] if r["ground_truth"] == "ACCEPT"][:23]
    canonical_rejects = [r["paper_id"] for r in ref["results"] if r["ground_truth"] == "REJECT"]
    canonical = set(canonical_accepts + canonical_rejects)

    # Exclude few-shot holdouts from rejects (last FEW_SHOT_HOLDOUT by sorted order)
    holdouts = set(sorted(rejects_pool)[-FEW_SHOT_HOLDOUT:])
    rejects_pool = [p for p in rejects_pool if p not in holdouts]

    rng = random.Random(10718)

    # Fill up to N_PER_CLASS, always keeping canonical papers
    extra_accepts = [p for p in accepts_pool if p not in canonical]
    extra_rejects = [p for p in rejects_pool if p not in canonical]
    rng.shuffle(extra_accepts)
    rng.shuffle(extra_rejects)

    n_extra_a = max(0, N_PER_CLASS - len(canonical_accepts))
    n_extra_r = max(0, N_PER_CLASS - len(canonical_rejects))

    final_accepts = canonical_accepts + extra_accepts[:n_extra_a]
    final_rejects = canonical_rejects + extra_rejects[:n_extra_r]

    return final_accepts + final_rejects


def results_file(model_key: str, with_reviews: bool, strategy: str = "neutral") -> Path:
    tag      = "with_reviews" if with_reviews else "no_reviews"
    strat    = "" if strategy == "neutral" else f"_{strategy}"
    return RESULTS_DIR / f"zero_shot_{model_key}_{tag}{strat}_neurips_2025_full.json"


def get_ground_truth(review_json: dict) -> str:
    val = review_json.get("accepted")
    if isinstance(val, bool):
        return "ACCEPT" if val else "REJECT"
    return "ACCEPT" if str(val).lower() in ("true", "yes", "1", "accept") else "REJECT"


def print_metrics(results: list[dict], model_id: str, usage: Usage) -> None:
    ev = [r for r in results if r["prediction"] is not None]
    if not ev:
        return
    tp = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n  = len(ev)
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"  n={n}  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Acc={(tp+tn)/n:.3f}  FPR={fpr:.3f}  TPR={tpr:.3f}")
    p = MODEL_PRICING.get(model_id)
    if p:
        cost = usage.input_tokens / 1000 * p["input"] + usage.output_tokens / 1000 * p["output"]
        print(f"  Cost: ${cost:.3f}  (${cost/n:.4f}/paper)")


def run(model_key: str, with_reviews: bool, strategy: str = "neutral") -> None:
    model_id  = MODELS[model_key]
    paper_ids = get_paper_ids()
    out_file  = results_file(model_key, with_reviews, strategy)

    # Check for legacy 46-paper result file to migrate from (neutral only)
    tag = "with_reviews" if with_reviews else "no_reviews"
    legacy_file = RESULTS_DIR / f"zero_shot_{model_key}_{tag}_46papers_neurips_2025_full.json"

    if out_file.exists():
        existing    = json.loads(out_file.read_text())
        results     = existing.get("results", [])
        total_usage = Usage(**existing.get("usage", {}))
        done        = {r["paper_id"] for r in results}
        print(f"Resuming {out_file.name} — {len(done)}/{len(paper_ids)} done.")
    elif strategy == "neutral" and legacy_file.exists():
        print(f"Migrating results from {legacy_file.name} ...")
        existing    = json.loads(legacy_file.read_text())
        results     = existing.get("results", [])
        total_usage = Usage(**existing.get("usage", {}))
        done        = {r["paper_id"] for r in results}
        print(f"  Loaded {len(done)} existing predictions — will skip these.")
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
        reviews_text = sample_reviews(review_json) if with_reviews else None

        prompt = build_prompt(paper_md, reviews_text, strategy=strategy)
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
            "with_reviews": with_reviews,
            "strategy":     strategy,
            "usage":        dataclasses.asdict(total_usage),
            "results":      results,
        }, indent=2))

    print(f"\n=== {out_file.name} ===")
    print_metrics(results, model_id, total_usage)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    ev  = [r for r in results if r["prediction"] is not None]
    tp  = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fp  = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn  = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    fn  = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    n   = len(ev)
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    # F1 = harmonic mean of precision and recall
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else float("nan")
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, n=n, fpr=fpr, tpr=tpr, acc=(tp+tn)/n, f1=f1)


def has_real_confidence(results: list[dict]) -> bool:
    """True if results contain model-elicited confidence (not all fallback 0.5)."""
    conf_vals = [r["confidence"] for r in results
                 if r.get("prediction") is not None and r.get("confidence") is not None]
    if not conf_vals:
        return False
    return any(c != 0.5 for c in conf_vals)


def accept_scores(results: list[dict]) -> list[float]:
    """Convert (prediction, confidence) pairs to a single accept-probability score."""
    scores = []
    for r in results:
        if r.get("prediction") is None:
            continue
        c = r.get("confidence", 0.5)
        scores.append(c if r["prediction"] == "ACCEPT" else 1.0 - c)
    return scores


def roc_from_confidence(results: list[dict]) -> tuple[list, list]:
    from sklearn.metrics import roc_curve
    y_true = [1 if r["ground_truth"] == "ACCEPT" else 0
              for r in results if r.get("prediction") is not None]
    fpr_arr, tpr_arr, _ = roc_curve(y_true, accept_scores(results))
    return fpr_arr.tolist(), tpr_arr.tolist()


def auc_from_confidence(results: list[dict]) -> float:
    from sklearn.metrics import roc_auc_score
    import warnings
    ev = [r for r in results if r.get("prediction") is not None]
    y_true = [1 if r["ground_truth"] == "ACCEPT" else 0 for r in ev]
    if len(set(y_true)) < 2:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return roc_auc_score(y_true, accept_scores(ev))
    except Exception:
        return float("nan")


def bootstrap_ci(results: list[dict], stat_fn, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Return (lo, hi) bootstrap CI for a scalar statistic over results."""
    import numpy as np
    rng = np.random.default_rng(10718)
    arr = np.array(results, dtype=object)
    stats = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, len(arr), len(arr))].tolist()
        try:
            stats.append(stat_fn(sample))
        except Exception:
            pass
    if not stats:
        return float("nan"), float("nan")
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def _fpr(results):
    ev = [r for r in results if r["prediction"] is not None]
    fp = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "REJECT")
    tn = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "REJECT")
    return fp / (fp + tn) if (fp + tn) else float("nan")


def _tpr(results):
    ev = [r for r in results if r["prediction"] is not None]
    tp = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    fn = sum(1 for r in ev if r["prediction"] == "REJECT" and r["ground_truth"] == "ACCEPT")
    return tp / (tp + fn) if (tp + fn) else float("nan")


def _f1(results):
    return compute_metrics(results)["f1"]


def make_plots() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np

    for ttf in Path(FONT_DIR).rglob("*.ttf"):
        fm.fontManager.addfont(str(ttf))

    volkhov  = fm.FontProperties(fname=VOLKHOV)
    mono     = fm.FontProperties(fname=UBUNTU_MONO)
    mono_b   = fm.FontProperties(fname=UBUNTU_MONO_B)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f8f8",
        "axes.edgecolor":   "#cccccc",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "grid.color":       "white",
        "grid.linewidth":   1.0,
    })

    fig_dir   = RESULTS_DIR / "figures" / "baselines"
    table_dir = RESULTS_DIR / "tables"  / "baselines"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    def load_results(path: Path) -> list[dict]:
        return json.loads(path.read_text()).get("results", [])

    # Zero-shot only — no few-shot, no debate
    # Model order: smallest to largest
    MODEL_META = {
        "gemma":    ("Gemma 3 12B",    "#55A868"),
        "llama":    ("Llama 3.3 70B",  "#4C72B0"),
        "deepseek": ("DeepSeek V3.2",  "#DD8452"),
    }

    no_rev, with_rev = [], []
    for key in MODEL_SIZE_ORDER:
        label, color = MODEL_META[key]
        for container, with_reviews in [(no_rev, False), (with_rev, True)]:
            p = results_file(key, with_reviews)
            if not p.exists():
                continue
            res = load_results(p)
            if len(res) < MIN_N:
                print(f"Skipping {p.name}: only {len(res)} results (< {MIN_N})")
                continue
            container.append((label, color, res))

    if not no_rev:
        print("No zero-shot results found yet — skipping plots.")
        return

    # ======================================================================
    # Plot 1: ROC curves with 95% bootstrap CI bands — no reviews
    # ======================================================================
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1, ls="--", zorder=1)
    ax.fill_between([0, 1], [0, 1], alpha=0)   # invisible — just for spacing

    for label, color, results in no_rev:
        if not results:
            continue
        m            = compute_metrics(results)
        f1_lo, f1_hi = bootstrap_ci(results, _f1)

        # full ROC curve from confidence scores
        fpr_c, tpr_c = roc_from_confidence(results)
        ax.plot(fpr_c, tpr_c, color=color, lw=2, zorder=3,
                label=f"{label}  F1={m['f1']:.2f} [{f1_lo:.2f}–{f1_hi:.2f}]")

        # operating point with FPR/TPR CIs as cross-hairs
        fpr_lo, fpr_hi = bootstrap_ci(results, _fpr)
        tpr_lo, tpr_hi = bootstrap_ci(results, _tpr)
        ax.errorbar(m["fpr"], m["tpr"],
                    xerr=[[m["fpr"] - fpr_lo], [fpr_hi - m["fpr"]]],
                    yerr=[[m["tpr"] - tpr_lo], [tpr_hi - m["tpr"]]],
                    fmt="o", color=color, ms=7, capsize=4, lw=1.5, zorder=4)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontproperties=mono, fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontproperties=mono, fontsize=11)
    ax.set_title("Zero-Shot Acceptance Prediction\nNeurIPS 2025  ·  n=46  ·  No Reviews",
                 fontproperties=volkhov, fontsize=13, pad=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(mono)
        lbl.set_fontsize(9)
    ax.legend(prop=mono, fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, zorder=0)
    fig.tight_layout()
    out = fig_dir / "roc_no_reviews.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ======================================================================
    # Plot 2: FPR and TPR bars — split by condition (No Reviews | With Reviews)
    # Within each subplot: one cluster per model (FPR red, TPR blue), sorted small->large
    # ======================================================================
    from matplotlib.patches import Patch

    wr_by_label = {lbl: res for lbl, _, res in with_rev if res}
    entries_no  = [(lbl, col, res) for lbl, col, res in no_rev if res]
    n_mod = len(entries_no)
    x     = np.arange(n_mod)
    w     = 0.32

    conditions = [("No Reviews", entries_no)]
    if wr_by_label:
        entries_wr = [(lbl, col, wr_by_label[lbl])
                      for lbl, col, _ in entries_no if lbl in wr_by_label]
        conditions.append(("With Reviews", entries_wr))

    n_cond = len(conditions)
    fig, axes = plt.subplots(
        1, n_cond,
        figsize=(3.2 * n_mod * n_cond + 1.5, 5.5),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    if n_cond == 1:
        axes = [axes]

    for ax, (cond_title, entries) in zip(axes, conditions):
        for i, (label, model_color, results) in enumerate(entries):
            fpr_val = _fpr(results)
            tpr_val = _tpr(results)
            flo, fhi = bootstrap_ci(results, _fpr)
            tlo, thi = bootstrap_ci(results, _tpr)

            # FPR bar
            ax.bar(x[i] - w/2, fpr_val, w, color="#C44E52", alpha=0.85, zorder=3)
            ax.errorbar(x[i] - w/2, fpr_val,
                        yerr=[[fpr_val - flo], [fhi - fpr_val]],
                        fmt="none", color="black", capsize=4, lw=1.5, zorder=4)
            ax.text(x[i] - w/2, fhi + 0.03, f"{fpr_val:.2f}",
                    ha="center", fontproperties=mono, fontsize=8.5, color="#C44E52")

            # TPR bar
            ax.bar(x[i] + w/2, tpr_val, w, color="#4C72B0", alpha=0.85, zorder=3)
            ax.errorbar(x[i] + w/2, tpr_val,
                        yerr=[[tpr_val - tlo], [thi - tpr_val]],
                        fmt="none", color="black", capsize=4, lw=1.5, zorder=4)
            ax.text(x[i] + w/2, thi + 0.03, f"{tpr_val:.2f}",
                    ha="center", fontproperties=mono, fontsize=8.5, color="#4C72B0")

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _, _ in entries],
                           fontproperties=mono, fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_title(cond_title, fontproperties=mono, fontsize=11, pad=8)
        ax.grid(axis="y", zorder=0)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(mono)
            tick.set_fontsize(9)
        ax.legend(
            handles=[
                Patch(color="#C44E52", alpha=0.85, label="FPR  (lower is better)"),
                Patch(color="#4C72B0", alpha=0.85, label="TPR  (higher is better)"),
            ],
            prop=mono, fontsize=8.5, framealpha=0.9, edgecolor="#cccccc",
        )

    axes[0].set_ylabel("Rate  [95% bootstrap CI]", fontproperties=mono, fontsize=11)
    n_label = f"n={len(entries_no[0][2])}" if entries_no else "n=46"
    fig.suptitle(
        f"Zero-Shot Acceptance Prediction — FPR & TPR by Model\n"
        f"NeurIPS 2025  ·  {n_label}  ·  balanced A/R",
        fontproperties=volkhov, fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = fig_dir / "fpr_tpr_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ======================================================================
    # Plot 3: ROC — no reviews vs with reviews (paired, if data available)
    # ======================================================================
    if with_rev:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1, ls="--", zorder=1)

        for (label, color, res_no), (_, _, res_wr) in zip(no_rev, with_rev):
            for res, ls, suffix in [(res_no, "--", ""), (res_wr, "-", " + reviews")]:
                if not res:
                    continue
                m     = compute_metrics(res)
                fpr_c, tpr_c = roc_from_confidence(res)
                ax.plot(fpr_c, tpr_c, color=color, lw=2, ls=ls, zorder=3,
                        label=f"{label}{suffix}  F1={m['f1']:.2f}")
                fpr_lo, fpr_hi = bootstrap_ci(res, _fpr)
                tpr_lo, tpr_hi = bootstrap_ci(res, _tpr)
                ax.errorbar(m["fpr"], m["tpr"],
                            xerr=[[m["fpr"]-fpr_lo],[fpr_hi-m["fpr"]]],
                            yerr=[[m["tpr"]-tpr_lo],[tpr_hi-m["tpr"]]],
                            fmt="o" if suffix else "s", color=color,
                            ms=7, capsize=4, lw=1.5, zorder=4)

        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("False Positive Rate", fontproperties=mono, fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontproperties=mono, fontsize=11)
        ax.set_title("Zero-Shot: No Reviews (dashed) vs With Reviews (solid)\nNeurIPS 2025  ·  n=46",
                     fontproperties=volkhov, fontsize=13, pad=12)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(mono)
            lbl.set_fontsize(9)
        ax.legend(prop=mono, fontsize=8.5, loc="lower right",
                  framealpha=0.9, edgecolor="#cccccc")
        ax.grid(True, zorder=0)
        fig.tight_layout()
        out = fig_dir / "roc_with_vs_without_reviews.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # ======================================================================
    # Markdown table — with bootstrap CIs
    # ======================================================================
    all_entries = [(lbl, "✗", res) for lbl, _, res in no_rev if res] + \
                  [(lbl, "✓", res) for lbl, _, res in with_rev if res]

    rows = []
    for label, wr, results in all_entries:
        m              = compute_metrics(results)
        fpr_lo, fpr_hi = bootstrap_ci(results, _fpr)
        tpr_lo, tpr_hi = bootstrap_ci(results, _tpr)
        f1_lo,  f1_hi  = bootstrap_ci(results, _f1)
        rows.append((label, wr, m["n"], m["acc"], m["fpr"], fpr_lo, fpr_hi,
                     m["tpr"], tpr_lo, tpr_hi, m["f1"], f1_lo, f1_hi))

    rows.sort(key=lambda r: r[4])

    lines = [
        "# Zero-Shot Baseline Comparison — NeurIPS 2025\n",
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

    table_path = table_dir / "baselines.md"
    table_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {table_path}")


def make_conservative_table() -> None:
    """Compare neutral vs conservative vs severe_conservative across all three models."""
    table_dir = RESULTS_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    MODEL_LABELS = {
        "gemma":    "Gemma 3 12B",
        "llama":    "Llama 3.3 70B",
        "deepseek": "DeepSeek V3.2",
    }

    rows = []
    for model_key in MODEL_SIZE_ORDER:
        label = MODEL_LABELS[model_key]
        for strategy in ("neutral", "conservative", "severe_conservative"):
            p = results_file(model_key, False, strategy)
            if not p.exists():
                continue
            results = json.loads(p.read_text()).get("results", [])
            if len(results) < MIN_N:
                continue
            m              = compute_metrics(results)
            fpr_lo, fpr_hi = bootstrap_ci(results, _fpr)
            tpr_lo, tpr_hi = bootstrap_ci(results, _tpr)
            f1_lo,  f1_hi  = bootstrap_ci(results, _f1)
            rows.append({
                "model":    label,
                "strategy": strategy,
                "n":        m["n"],
                "acc":      m["acc"],
                "fpr":      m["fpr"], "fpr_lo": fpr_lo, "fpr_hi": fpr_hi,
                "tpr":      m["tpr"], "tpr_lo": tpr_lo, "tpr_hi": tpr_hi,
                "f1":       m["f1"],  "f1_lo":  f1_lo,  "f1_hi":  f1_hi,
            })

    lines = [
        "# Prompting Strategy Comparison — No Reviews, NeurIPS 2025\n",
        "> 95% bootstrap CIs (2000 resamples, seed=10718).\n"
        "> Models sorted smallest to largest; strategies: neutral / conservative / severe_conservative.\n",
        "| Model | Strategy | n | Acc | FPR (lower) | 95% CI | TPR (higher) | 95% CI | F1 | 95% CI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['strategy']} | {r['n']} | {r['acc']:.3f} "
            f"| {r['fpr']:.3f} | [{r['fpr_lo']:.3f}, {r['fpr_hi']:.3f}] "
            f"| {r['tpr']:.3f} | [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}] "
            f"| {r['f1']:.3f} | [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}] |"
        )

    out = table_dir / "conservative_prompting.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS),
                        help="Model to evaluate")
    parser.add_argument("--with-reviews", action="store_true",
                        help="Include real peer reviews (meta-reviews stripped)")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="neutral",
                        help="Prompting strategy (default: neutral)")
    parser.add_argument("--plot", action="store_true",
                        help="Regenerate plots and tables from existing results")
    args = parser.parse_args()

    if args.plot:
        make_plots()
        make_conservative_table()
        return

    if not args.model:
        parser.error("--model is required unless --plot is passed")

    run(args.model, args.with_reviews, args.strategy)
    make_plots()
    make_conservative_table()


if __name__ == "__main__":
    main()
