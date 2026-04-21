"""
Distribution shift analysis: ICLR 2023-2025 vs NeurIPS 2023-2025.

Outputs:
  results/figures/distribution/   -- score distributions, separation, heuristics
  results/tables/distribution/    -- markdown summary tables

Run:
  python analyze_iclr_neurips_shift.py

Notes:
  - ICLR 2023/2024 loaded via parquet (huggingface_hub) due to datasets schema mismatch.
  - ICLR 2025 loaded via datasets library (works fine).
  - NeurIPS loaded from local output/ directory.
  - All scores normalized to [0,1] within venue using empirical max scale.
"""
from __future__ import annotations
import ast, json
from pathlib import Path
import numpy as np
from scipy import stats

RESULTS_DIR   = Path("results")
FIG_DIR       = RESULTS_DIR / "figures" / "distribution"
TABLE_DIR     = RESULTS_DIR / "tables"  / "distribution"
FONT_DIR      = Path("~/fonts").expanduser()
VOLKHOV       = str(FONT_DIR / "Volkhov"     / "Volkhov-Bold.ttf")
UBUNTU_MONO   = str(FONT_DIR / "Ubuntu_Mono" / "UbuntuMono-Regular.ttf")

# Score scale per venue (empirically confirmed)
NEURIPS_MAX = {2023: 10, 2024: 10, 2025: 6}
ICLR_MAX    = {2023: 10, 2024: 10, 2025: 10}

NEURIPS_DIRS = {
    2023: Path("output/neurips_2023/reviews"),
    2024: Path("output/neurips_2024/reviews"),
    2025: Path("output/neurips_2025_full/reviews"),
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_neurips(year: int) -> dict:
    acc, rej = [], []
    mx = NEURIPS_MAX[year]
    for p in sorted(NEURIPS_DIRS[year].glob("*.json")):
        d        = json.loads(p.read_text())
        non_meta = [r for r in d.get("reviews", []) if not r.get("IS_META_REVIEW")]
        scores   = []
        for r in non_meta:
            raw = r.get("RECOMMENDATION")
            if raw is None:
                continue
            try:
                scores.append(float(str(raw).split(":")[0].strip()))
            except (ValueError, AttributeError):
                pass
        entry = {
            "paper_id":       p.stem,
            "accepted":       bool(d.get("accepted")),
            "scores":         scores,
            "n_reviews":      len(non_meta),
            "abstract_words": len(str(d.get("abstract", "")).split()),
            "venue":          f"NeurIPS {year}",
            "year":           year,
            "score_max":      mx,
        }
        (acc if entry["accepted"] else rej).append(entry)
    print(f"NeurIPS {year}: {len(acc)} accept, {len(rej)} reject  (scale 1-{mx})")
    return {"accept": acc, "reject": rej}


def _load_iclr_parquet(year: int) -> dict:
    """Load ICLR 2023/2024 via parquet, bypassing datasets schema mismatch."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    acc, rej = [], []
    mx = ICLR_MAX[year]
    yr_str = f"iclr{year}"

    for split in ("accept", "reject"):
        frames = []
        for shard in ["00000-of-00001", "00000-of-00002", "00001-of-00002"]:
            try:
                path = hf_hub_download(
                    repo_id="ReviewHub/ICLR", repo_type="dataset",
                    filename=f"{yr_str}/{split}-{shard}.parquet",
                )
                frames.append(pd.read_parquet(path))
            except Exception:
                pass
        if not frames:
            print(f"  ICLR {year}/{split}: no parquet shards found")
            continue
        df = pd.concat(frames, ignore_index=True)
        for _, row in df.iterrows():
            raw = row.get("reviewer_scores_json", [])
            try:
                sc = [float(s) for s in
                      (ast.literal_eval(raw) if isinstance(raw, str) else raw)
                      if s is not None]
            except Exception:
                sc = []
            ab = str(row.get("abstract", "") or "")
            entry = {
                "paper_id":       str(row.get("file_id", "")),
                "accepted":       split == "accept",
                "scores":         sc,
                "n_reviews":      len(sc),
                "abstract_words": len(ab.split()),
                "venue":          f"ICLR {year}",
                "year":           year,
                "score_max":      mx,
            }
            (acc if split == "accept" else rej).append(entry)
    print(f"ICLR {year}: {len(acc)} accept, {len(rej)} reject  (scale 1-{mx})")
    return {"accept": acc, "reject": rej}


def _load_iclr_datasets(year: int) -> dict:
    """Load ICLR 2025 via datasets library (works fine for this year)."""
    from datasets import load_dataset
    acc, rej = [], []
    mx = ICLR_MAX[year]
    yr_str = f"iclr{year}"
    for split in ("accept", "reject"):
        try:
            ds = load_dataset("ReviewHub/ICLR", yr_str, split=split)
        except Exception as e:
            print(f"  ICLR {year}/{split}: {e}")
            continue
        for i in range(len(ds)):
            row = ds[i]
            raw = row.get("reviewer_scores_json", [])
            try:
                sc = [float(s) for s in
                      (ast.literal_eval(raw) if isinstance(raw, str) else raw)
                      if s is not None]
            except Exception:
                sc = []
            ab = str(row.get("abstract", "") or "")
            entry = {
                "paper_id":       row.get("file_id", f"{yr_str}_{i}"),
                "accepted":       split == "accept",
                "scores":         sc,
                "n_reviews":      len(sc),
                "abstract_words": len(ab.split()),
                "venue":          f"ICLR {year}",
                "year":           year,
                "score_max":      mx,
            }
            (acc if split == "accept" else rej).append(entry)
    print(f"ICLR {year}: {len(acc)} accept, {len(rej)} reject  (scale 1-{mx})")
    return {"accept": acc, "reject": rej}


def load_iclr(year: int) -> dict:
    if year in (2023, 2024):
        return _load_iclr_parquet(year)
    return _load_iclr_datasets(year)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def norm_scores(papers: list) -> np.ndarray:
    """Normalized mean score per paper (divided by venue max scale)."""
    return np.array([np.mean(p["scores"]) / p["score_max"]
                     for p in papers if p["scores"]])

def within_std(papers: list) -> np.ndarray:
    return np.array([np.std([x / p["score_max"] for x in p["scores"]])
                     for p in papers if len(p["scores"]) > 1])

def abstract_words_arr(papers: list) -> np.ndarray:
    return np.array([p["abstract_words"] for p in papers], dtype=float)

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / s) if s > 0 else float("nan")

def bootstrap_ci(arr: np.ndarray, stat_fn=np.mean, n: int = 2000, seed: int = 10718):
    rng = np.random.default_rng(seed)
    if len(arr) < 2:
        return float("nan"), float("nan")
    samples = [stat_fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

def bootstrap_ci_d(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 10718):
    rng = np.random.default_rng(seed)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    samples = [cohens_d(a[rng.integers(0, len(a), len(a))],
                        b[rng.integers(0, len(b), len(b))]) for _ in range(n)]
    samples = [x for x in samples if not np.isnan(x)]
    if not samples:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for ttf in FONT_DIR.rglob("*.ttf"):
        fm.fontManager.addfont(str(ttf))
    volkhov = fm.FontProperties(fname=VOLKHOV)
    mono    = fm.FontProperties(fname=UBUNTU_MONO)

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
        "axes.edgecolor": "#cccccc", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "white", "grid.linewidth": 1.0,
    })

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load all venues, then sort by year
    raw = {}
    for year in (2023, 2024, 2025):
        raw[f"NeurIPS {year}"] = load_neurips(year)
        raw[f"ICLR {year}"]    = load_iclr(year)

    # Sort venues chronologically: interleave ICLR and NeurIPS by year
    venues = sorted(raw.keys(), key=lambda v: (int(v.split()[-1]), v.split()[0]))
    bins   = np.linspace(0, 1, 21)

    # Palette: one color per year, different markers for ICLR vs NeurIPS
    YEAR_COLOR  = {2023: "#4C72B0", 2024: "#DD8452", 2025: "#55A868"}
    VENUE_STYLE = {"ICLR": "--", "NeurIPS": "-"}
    VENUE_MARKER= {"ICLR": "^",  "NeurIPS": "o"}

    def venue_color(v):
        return YEAR_COLOR[int(v.split()[-1])]

    # =====================================================================
    # Plot 1: Normalized score distributions — accept vs reject, per venue
    # =====================================================================
    n_cols = 3
    n_rows = int(np.ceil(len(venues) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharey=False)
    axes = axes.flatten()

    for i, venue in enumerate(venues):
        ax = axes[i]
        d  = raw[venue]
        ma = norm_scores(d["accept"])
        mr = norm_scores(d["reject"])
        if len(ma):
            ax.hist(ma, bins=bins, alpha=0.6, color="#4C72B0", density=True,
                    label=f"Accept  n={len(d['accept'])}")
            alo, ahi = bootstrap_ci(ma)
            ax.axvline(np.mean(ma), color="#4C72B0", lw=1.5)
            ax.axvspan(alo, ahi, alpha=0.15, color="#4C72B0")
        if len(mr):
            ax.hist(mr, bins=bins, alpha=0.6, color="#C44E52", density=True,
                    label=f"Reject  n={len(d['reject'])}")
            rlo, rhi = bootstrap_ci(mr)
            ax.axvline(np.mean(mr), color="#C44E52", lw=1.5)
            ax.axvspan(rlo, rhi, alpha=0.15, color="#C44E52")
        scale = raw[venue]["accept"][0]["score_max"] if raw[venue]["accept"] else "?"
        ax.set_title(f"{venue}  (scale 1-{scale})", fontproperties=mono, fontsize=9, pad=5)
        ax.set_xlabel("Normalized mean score", fontproperties=mono, fontsize=8)
        ax.set_ylabel("Density",               fontproperties=mono, fontsize=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(mono); lbl.set_fontsize(7)
        ax.legend(prop=mono, fontsize=7)
        ax.grid(True, zorder=0)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Normalized Reviewer Score Distributions — Accept vs Reject\n"
                 "Vertical lines = mean, shaded = 95% bootstrap CI",
                 fontproperties=volkhov, fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "score_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # =====================================================================
    # Plot 2: Cohen's d and mean gap (accept-reject separation), sorted by year
    # =====================================================================
    print("Computing bootstrap CIs...")
    sep = []
    for venue in venues:
        d  = raw[venue]
        ma = norm_scores(d["accept"]); mr = norm_scores(d["reject"])
        if not len(ma) or not len(mr):
            continue
        ks, ksp = stats.ks_2samp(ma, mr)
        gap     = float(np.mean(ma) - np.mean(mr))
        rng     = np.random.default_rng(10718)
        gap_samples = [
            np.mean(ma[rng.integers(0, len(ma), len(ma))]) -
            np.mean(mr[rng.integers(0, len(mr), len(mr))])
            for _ in range(2000)
        ]
        gap_lo, gap_hi       = float(np.percentile(gap_samples, 2.5)), float(np.percentile(gap_samples, 97.5))
        d_val                = cohens_d(ma, mr)
        d_lo, d_hi           = bootstrap_ci_d(ma, mr)
        mean_a_lo, mean_a_hi = bootstrap_ci(ma)
        mean_r_lo, mean_r_hi = bootstrap_ci(mr)
        sep.append({
            "venue": venue, "year": int(venue.split()[-1]),
            "n_a": len(d["accept"]), "n_r": len(d["reject"]),
            "acc_rate": len(d["accept"]) / (len(d["accept"]) + len(d["reject"])),
            "mean_a": float(np.mean(ma)), "mean_a_lo": mean_a_lo, "mean_a_hi": mean_a_hi,
            "mean_r": float(np.mean(mr)), "mean_r_lo": mean_r_lo, "mean_r_hi": mean_r_hi,
            "gap": gap, "gap_lo": gap_lo, "gap_hi": gap_hi,
            "d": d_val, "d_lo": d_lo, "d_hi": d_hi,
            "ks": float(ks), "ksp": float(ksp),
        })

    # Sort by year for bar charts
    sep_by_year = sorted(sep, key=lambda r: (r["year"], r["venue"]))
    lbls = [r["venue"] for r in sep_by_year]
    x    = np.arange(len(lbls))
    bar_colors = [venue_color(v) for v in lbls]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.35})

    gaps   = [r["gap"] for r in sep_by_year]
    g_errs = [[r["gap"] - r["gap_lo"] for r in sep_by_year],
              [r["gap_hi"] - r["gap"] for r in sep_by_year]]
    ax1.barh(x, gaps, color=bar_colors, alpha=0.8, zorder=3)
    ax1.errorbar(gaps, x, xerr=g_errs, fmt="none", color="#333333",
                 capsize=4, lw=1.5, zorder=4)
    ax1.set_yticks(x); ax1.set_yticklabels(lbls, fontproperties=mono, fontsize=9)
    ax1.set_xlabel("Mean score gap, accept minus reject (normalized)\n[95% bootstrap CI]",
                   fontproperties=mono, fontsize=9)
    ax1.set_title("Score Gap (sorted by year)", fontproperties=mono, fontsize=11)
    ax1.grid(axis="x", zorder=0)
    for lbl in ax1.get_xticklabels(): lbl.set_fontproperties(mono); lbl.set_fontsize(8)

    d_vals = [r["d"] for r in sep_by_year]
    d_errs = [[r["d"] - r["d_lo"] for r in sep_by_year],
              [r["d_hi"] - r["d"] for r in sep_by_year]]
    ax2.barh(x, d_vals, color=bar_colors, alpha=0.8, zorder=3)
    ax2.errorbar(d_vals, x, xerr=d_errs, fmt="none", color="#333333",
                 capsize=4, lw=1.5, zorder=4)
    ax2.set_yticks(x); ax2.set_yticklabels(lbls, fontproperties=mono, fontsize=9)
    ax2.set_xlabel("Cohen's d  [95% bootstrap CI]", fontproperties=mono, fontsize=9)
    ax2.set_title("Effect Size — Cohen's d (sorted by year)", fontproperties=mono, fontsize=11)
    for thresh, ls, label in [(0.2, "--", "small"), (0.5, ":", "medium"), (0.8, "-.", "large")]:
        ax2.axvline(thresh, color="grey", ls=ls, lw=1, alpha=0.7)
        ax2.text(thresh + 0.01, len(lbls) - 0.5, label, fontproperties=mono,
                 fontsize=7, color="grey", va="top")
    ax2.grid(axis="x", zorder=0)
    for lbl in ax2.get_xticklabels(): lbl.set_fontproperties(mono); lbl.set_fontsize(8)

    # Color legend by year
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=YEAR_COLOR[yr], label=str(yr)) for yr in (2023, 2024, 2025)],
               prop=mono, fontsize=8, title="Year", title_fontsize=8)

    fig.suptitle("Accept-Reject Score Separation by Venue and Year",
                 fontproperties=volkhov, fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "accept_reject_separation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # =====================================================================
    # Plot 3: Score distribution overlay — ICLR vs NeurIPS per year
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, year in zip(axes, (2023, 2024, 2025)):
        iclr_all    = norm_scores(raw[f"ICLR {year}"]["accept"] + raw[f"ICLR {year}"]["reject"])
        neurips_all = norm_scores(raw[f"NeurIPS {year}"]["accept"] + raw[f"NeurIPS {year}"]["reject"])
        if len(iclr_all) and len(neurips_all):
            ks, p = stats.ks_2samp(iclr_all, neurips_all)
        else:
            ks, p = float("nan"), float("nan")
        if len(neurips_all):
            ax.hist(neurips_all, bins=bins, alpha=0.6, color="#4C72B0", density=True,
                    label=f"NeurIPS {year}  (n={len(neurips_all)})")
        if len(iclr_all):
            ax.hist(iclr_all, bins=bins, alpha=0.6, color="#DD8452", density=True,
                    label=f"ICLR {year}  (n={len(iclr_all)})")
        ax.set_title(f"{year}  KS={ks:.2f}  p={p:.1e}", fontproperties=mono, fontsize=10)
        ax.set_xlabel("Normalized mean score", fontproperties=mono, fontsize=9)
        ax.set_ylabel("Density",               fontproperties=mono, fontsize=9)
        ax.legend(prop=mono, fontsize=8)
        ax.grid(True, zorder=0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(mono); lbl.set_fontsize(8)
    fig.suptitle("Score Distribution Shift: ICLR vs NeurIPS by Year (all papers)",
                 fontproperties=volkhov, fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "venue_score_shift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # =====================================================================
    # Plot 4: Acceptance rate + reviewer disagreement
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    acc_by_year = sorted(sep, key=lambda r: (r["year"], r["venue"]))
    lbls_a = [r["venue"] for r in acc_by_year]
    cols_a = [venue_color(v) for v in lbls_a]
    ax1.barh(range(len(acc_by_year)), [r["acc_rate"] for r in acc_by_year],
             color=cols_a, alpha=0.8)
    ax1.set_yticks(range(len(acc_by_year)))
    ax1.set_yticklabels(lbls_a, fontproperties=mono, fontsize=9)
    ax1.set_xlabel("Acceptance rate", fontproperties=mono, fontsize=9)
    ax1.set_title("Acceptance Rate by Venue (sorted by year)", fontproperties=mono, fontsize=11)
    ax1.axvline(0.5, color="grey", ls="--", lw=1)
    ax1.grid(axis="x", zorder=0)
    for lbl in ax1.get_xticklabels(): lbl.set_fontproperties(mono); lbl.set_fontsize(8)

    data_a, data_r, box_labels, box_colors = [], [], [], []
    for venue in venues:
        d  = raw[venue]
        wa = within_std(d["accept"]); wr = within_std(d["reject"])
        if len(wa) and len(wr):
            data_a.append(wa); data_r.append(wr)
            box_labels.append(venue); box_colors.append(venue_color(venue))

    xs = np.arange(len(box_labels))
    w  = 0.35
    ax2.boxplot([d.tolist() for d in data_a], positions=xs - w/2, widths=w * 0.8,
                patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                medianprops=dict(color="navy"), flierprops=dict(ms=2, alpha=0.3),
                whiskerprops=dict(alpha=0.5), capprops=dict(alpha=0.5))
    ax2.boxplot([d.tolist() for d in data_r], positions=xs + w/2, widths=w * 0.8,
                patch_artist=True, boxprops=dict(facecolor="#C44E52", alpha=0.6),
                medianprops=dict(color="darkred"), flierprops=dict(ms=2, alpha=0.3),
                whiskerprops=dict(alpha=0.5), capprops=dict(alpha=0.5))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(box_labels, fontproperties=mono, fontsize=8, rotation=30, ha="right")
    for lbl in ax2.get_yticklabels(): lbl.set_fontproperties(mono); lbl.set_fontsize(8)
    ax2.set_ylabel("Within-paper score std (normalized)", fontproperties=mono, fontsize=9)
    ax2.set_title("Reviewer Disagreement — Accept (blue) vs Reject (red)",
                  fontproperties=mono, fontsize=10)
    ax2.grid(axis="y", zorder=0)
    fig.suptitle("Acceptance Rate and Reviewer Disagreement by Venue",
                 fontproperties=volkhov, fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "acceptance_and_agreement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # =====================================================================
    # Plot 5: Abstract length scatter — accept vs reject, per venue
    # =====================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    for venue in venues:
        d   = raw[venue]
        wa  = abstract_words_arr(d["accept"]); wr = abstract_words_arr(d["reject"])
        if not len(wa) or not len(wr):
            continue
        col = venue_color(venue)
        mrk = VENUE_MARKER[venue.split()[0]]
        ax.scatter(np.mean(wa), np.mean(wr), s=80, color=col, marker=mrk,
                   zorder=3, label=venue)
        alo, ahi = bootstrap_ci(wa); rlo, rhi = bootstrap_ci(wr)
        ax.errorbar(np.mean(wa), np.mean(wr),
                    xerr=[[np.mean(wa) - alo], [ahi - np.mean(wa)]],
                    yerr=[[np.mean(wr) - rlo], [rhi - np.mean(wr)]],
                    fmt="none", color=col, alpha=0.6, capsize=3)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Mean abstract words — Accept", fontproperties=mono, fontsize=10)
    ax.set_ylabel("Mean abstract words — Reject", fontproperties=mono, fontsize=10)
    ax.set_title("Abstract Length: Accept vs Reject by Venue\n"
                 "(error bars = 95% bootstrap CI, triangle = ICLR, circle = NeurIPS)",
                 fontproperties=volkhov, fontsize=12, pad=10)
    ax.legend(prop=mono, fontsize=8)
    ax.grid(True, zorder=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(mono); lbl.set_fontsize(9)
    fig.tight_layout()
    out = FIG_DIR / "abstract_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # =====================================================================
    # Tables (sorted by year)
    # =====================================================================
    lines = [
        "# Accept-Reject Score Separation by Venue\n",
        "> Scores normalized to [0,1] within venue (divided by max scale).\n"
        "> 95% bootstrap CIs, 2000 resamples, seed=10718.\n"
        "> Cohen's d: 0.2 = small, 0.5 = medium, 0.8 = large.\n",
        "| Venue | n_accept | n_reject | Acc rate | Mean(A) | 95% CI | Mean(R) | 95% CI | "
        "Gap | 95% CI | Cohen's d | 95% CI | KS stat | KS p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sep_by_year:
        lines.append(
            f"| {r['venue']} | {r['n_a']} | {r['n_r']} | {r['acc_rate']:.3f} "
            f"| {r['mean_a']:.3f} | [{r['mean_a_lo']:.3f}, {r['mean_a_hi']:.3f}] "
            f"| {r['mean_r']:.3f} | [{r['mean_r_lo']:.3f}, {r['mean_r_hi']:.3f}] "
            f"| {r['gap']:.3f} | [{r['gap_lo']:.3f}, {r['gap_hi']:.3f}] "
            f"| {r['d']:.2f} | [{r['d_lo']:.2f}, {r['d_hi']:.2f}] "
            f"| {r['ks']:.3f} | {r['ksp']:.2e} |"
        )
    (TABLE_DIR / "separation.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {TABLE_DIR / 'separation.md'}")

    lines = [
        "# Score and Heuristic Summary (sorted by year)\n",
        "| Venue | Split | n | Acc rate | Score mean | Score std | "
        "Within-paper std | Avg reviews | Avg abstract words |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for venue in venues:
        d        = raw[venue]
        acc_rate = len(d["accept"]) / (len(d["accept"]) + len(d["reject"]) + 1e-9)
        for split_label, papers in [("accept", d["accept"]), ("reject", d["reject"])]:
            ms = norm_scores(papers); ws = within_std(papers)
            nr = np.array([p["n_reviews"] for p in papers], dtype=float)
            aw = abstract_words_arr(papers)
            lines.append(
                f"| {venue} | {split_label} | {len(papers)} | {acc_rate:.3f} "
                f"| {np.mean(ms):.3f} | {np.std(ms):.3f} | "
                f"{np.mean(ws):.3f} | {np.mean(nr):.1f} | {np.mean(aw):.0f} |"
            )
    (TABLE_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {TABLE_DIR / 'summary.md'}")


if __name__ == "__main__":
    run()
