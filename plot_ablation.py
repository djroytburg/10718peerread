"""Plot the few-shot R/A composition ablation results with 95% Clopper-Pearson CIs."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.metrics import compute_cis

# ---------------------------------------------------------------------------
# Register fonts
# ---------------------------------------------------------------------------
FONTS_DIR = Path.home() / "fonts"
for ttf in FONTS_DIR.rglob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

VOLKHOV    = "Volkhov"
UBUNTUMONO = "Ubuntu Mono"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/few_shot_ablation_dsv3_neurips_2025_full.json",
        help="Input ablation JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/few_shot_ablation_plot.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Data + CIs
    # -----------------------------------------------------------------------
    data_file = Path(args.input)
    data = json.loads(data_file.read_text())

    compositions = [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)]
    x = [n_r / 5 for n_r, _ in compositions]

    fpr_vals, tpr_vals, acc_vals = [], [], []
    fpr_ci, tpr_ci, acc_ci = [], [], []

    for n_r, n_a in compositions:
        key = f"{n_r}R{n_a}A"
        results = [r for r in data["compositions"][key] if r["prediction"] is not None]
        ci = compute_cis(results)
        fpr_vals.append(ci["fpr"]["point"])
        tpr_vals.append(ci["tpr"]["point"])
        acc_vals.append(ci["acc"]["point"])
        fpr_ci.append((ci["fpr"]["point"] - ci["fpr"]["lower"],
                       ci["fpr"]["upper"] - ci["fpr"]["point"]))
        tpr_ci.append((ci["tpr"]["point"] - ci["tpr"]["lower"],
                       ci["tpr"]["upper"] - ci["tpr"]["point"]))
        acc_ci.append((ci["acc"]["point"] - ci["acc"]["lower"],
                       ci["acc"]["upper"] - ci["acc"]["point"]))

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))

    PALETTE = {
        "FPR": "#D94F3D",
        "TPR": "#4C72B0",
        "Accuracy": "#55A868",
    }

    ax.plot(x, fpr_vals, marker="o", linewidth=2, color=PALETTE["FPR"], label="FPR")
    ax.plot(x, tpr_vals, marker="s", linewidth=2, color=PALETTE["TPR"], label="TPR")
    ax.plot(x, acc_vals, marker="^", linewidth=2, color=PALETTE["Accuracy"], label="Accuracy")

    # Annotate each point
    for xi, fpr, tpr, acc in zip(x, fpr_vals, tpr_vals, acc_vals):
        ax.annotate(f"{fpr:.2f}", (xi, fpr), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=PALETTE["FPR"],
                    fontfamily=UBUNTUMONO)
        ax.annotate(f"{tpr:.2f}", (xi, tpr), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=PALETTE["TPR"],
                    fontfamily=UBUNTUMONO)
        ax.annotate(f"{acc:.2f}", (xi, acc), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=PALETTE["Accuracy"],
                    fontfamily=UBUNTUMONO)

    ax.set_xlabel("Fraction of negative (reject) few-shot examples",
                  fontfamily=VOLKHOV, fontsize=12)
    ax.set_ylabel("Metric value", fontfamily=VOLKHOV, fontsize=12)
    n_eval = data.get("test_n", "unknown")
    ax.set_title(
        f"Few-shot composition ablation\n(5 examples, no reviews, DeepSeek V3.2, n={n_eval})",
        fontfamily=VOLKHOV,
        fontsize=13,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["5/5\n(5R0A)", "4/5\n(4R1A)", "3/5\n(3R2A)", "2/5\n(2R3A)", "1/5\n(1R4A)", "0/5\n(0R5A)"],
        fontfamily=UBUNTUMONO,
        fontsize=9,
    )
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for label in ax.get_yticklabels():
        label.set_fontfamily(UBUNTUMONO)
        label.set_fontsize(9)

    ax.legend(loc="upper right", prop={"family": VOLKHOV, "size": 10}, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
