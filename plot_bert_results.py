#!/usr/bin/env python3
"""Generate BERT classifier table (LaTeX) and figure (PNG).

CIs are computed as 95% t-intervals across multiple independent training runs
stored in repetition_results/.
"""

import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO = Path(__file__).parent

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONTS_DIR = Path.home() / "fonts"
for ttf in FONTS_DIR.rglob("*.ttf"):
    fm.fontManager.addfont(str(ttf))
VOLKHOV    = "Volkhov"
UBUNTUMONO = "Ubuntu Mono"

# ---------------------------------------------------------------------------
# Load repetition runs
# ---------------------------------------------------------------------------

def load_runs(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]

# t critical value (two-tailed 95%) for small n
def t_crit(n: int) -> float:
    # scipy not guaranteed; use lookup for common n, else ~1.96
    table = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,
             9:2.306,10:2.262,11:2.228,12:2.201,13:2.179,14:2.160,15:2.145,
             16:2.131,17:2.120,18:2.110,19:2.101,20:2.093,21:2.086,22:2.080,
             23:2.074,24:2.069,25:2.064,30:2.042,40:2.021,60:2.000,120:1.980}
    for k in sorted(table):
        if n <= k:
            return table[k]
    return 1.960

def ci_from_runs(runs: list[dict], metric: str) -> dict:
    vals = [r[metric] for r in runs]
    n    = len(vals)
    mu   = sum(vals) / n
    if n < 2:
        return {"point": mu, "lower": mu, "upper": mu}
    var  = sum((v - mu) ** 2 for v in vals) / (n - 1)
    se   = math.sqrt(var / n)
    hw   = t_crit(n) * se
    return {"point": mu, "lower": max(0.0, mu - hw), "upper": min(1.0, mu + hw)}

# Map: (label, repetition file, train corpus label, test corpus label)
CONDITIONS = [
    ("Old->Old", "repetition_results/o_o_repeats.json",  "PeerRead", "PeerRead"),
    ("Old->New", "repetition_results/o_n_repeats.json",  "PeerRead", "NeurIPS"),
    ("New->Old", "repetition_results/n_o_repeats.json",  "NeurIPS",  "PeerRead"),
    ("New->New", "repetition_results/n_n_repeats.jsonl", "NeurIPS",  "NeurIPS"),
]

METRIC_KEYS = {
    "acc":    "accuracy",
    "fpr":    "false_positive_rate",
    "recall": "recall",
    "f1":     "f1",
}

stats = {}
for label, fname, train_c, test_c in CONDITIONS:
    runs = load_runs(REPO / fname)
    n_runs = len(runs)
    n_examples = runs[0]["num_labeled_examples"]
    entry = {"train": train_c, "test": test_c, "n": n_examples, "n_runs": n_runs}
    for key, mkey in METRIC_KEYS.items():
        entry[key] = ci_from_runs(runs, mkey)
    stats[label] = entry
    print(f"  {label} ({n_runs} runs, n={n_examples}): "
          f"acc={entry['acc']['point']:.3f}  fpr={entry['fpr']['point']:.3f}  "
          f"recall={entry['recall']['point']:.3f}  f1={entry['f1']['point']:.3f}")

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def fmt_ci(d, decimals=2):
    hw = (d["upper"] - d["lower"]) / 2
    return rf"{d['point']:.{decimals}f}~($\pm${hw:.{decimals}f})"

LABELS   = [c[0] for c in CONDITIONS]
BOLD_ACC = max(LABELS, key=lambda l: stats[l]["acc"]["point"])
BOLD_REC = max(LABELS, key=lambda l: stats[l]["recall"]["point"])
BOLD_F1  = max(LABELS, key=lambda l: stats[l]["f1"]["point"])
BOLD_FPR = min(LABELS, key=lambda l: stats[l]["fpr"]["point"])
IN_DIST  = {"Old->Old", "New->New"}

rows = []
for label, _, train_c, test_c in CONDITIONS:
    shade = r"\rowcolor{gray!12} " if label in IN_DIST else ""
    def cell(key, best):
        s = fmt_ci(stats[label][key])
        return rf"\textbf{{{s}}}" if label == best else s
    rows.append(
        rf"    {shade}{train_c} & {test_c} & {stats[label]['n']} & "
        rf"{cell('acc', BOLD_ACC)} & {cell('fpr', BOLD_FPR)} & "
        rf"{cell('recall', BOLD_REC)} & {cell('f1', BOLD_F1)} \\"
    )

table = r"""\begin{table}[t]
  \centering
  \caption{\textbf{DistilBERT acceptance classifier under distribution shift.}
  In-distribution conditions (shaded) train and test on the same venue year.
  Results are mean $\pm$ half-width of 95\% $t$-interval across independent
  training runs. \textbf{Bold} = best per column.}
  \label{tab:bert-classifier}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{llccccc}
    \toprule
    Train & Test & $n$ & Acc & \textcolor{red!70!black}{FPR~$\downarrow$} & \textcolor{green!50!black}{Recall~$\uparrow$} & \textcolor{green!50!black}{F1~$\uparrow$} \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""

out_tex = REPO / "paper" / "tables" / "bert_classifier.tex"
out_tex.write_text(table)
print(f"\nWrote {out_tex}")
print(table)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

XLABELS = [
    "Train: PeerRead\nTest:  PeerRead\n(in-dist.)",
    "Train: PeerRead\nTest:  NeurIPS\n(cross)",
    "Train: NeurIPS\nTest:  PeerRead\n(cross)",
    "Train: NeurIPS\nTest:  NeurIPS\n(in-dist.)",
]
METRICS       = ["f1", "recall", "fpr"]
METRIC_LABELS = {"f1": "F1", "recall": "Recall", "fpr": "FPR"}
PALETTE       = {"f1": "#4C72B0", "recall": "#55A868", "fpr": "#D94F3D"}

x     = np.arange(len(LABELS))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, metric in enumerate(METRICS):
    vals   = [stats[l][metric]["point"] for l in LABELS]
    lowers = [stats[l][metric]["point"] - stats[l][metric]["lower"] for l in LABELS]
    uppers = [stats[l][metric]["upper"] - stats[l][metric]["point"] for l in LABELS]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=METRIC_LABELS[metric],
                  color=PALETTE[metric], alpha=0.85,
                  yerr=[lowers, uppers], capsize=4,
                  error_kw={"elinewidth": 1.2})
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                fontfamily=UBUNTUMONO, color=PALETTE[metric])

for x_pos in [0, 3]:
    ax.axvspan(x_pos - 0.45, x_pos + 0.45, color="gray", alpha=0.08, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(XLABELS, fontfamily=UBUNTUMONO, fontsize=9)
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily(UBUNTUMONO)
    lbl.set_fontsize(9)
ax.set_ylabel("Metric value", fontfamily=VOLKHOV, fontsize=12)
ax.set_title("DistilBERT classifier: in-distribution vs. cross-domain",
             fontfamily=VOLKHOV, fontsize=13)
ax.set_ylim(0, 1.18)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(prop={"family": VOLKHOV, "size": 10}, framealpha=0.85, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out_fig = REPO / "paper" / "figures" / "bert_classifier.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Saved {out_fig}")
