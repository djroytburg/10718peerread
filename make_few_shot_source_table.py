"""
Compare few-shot 3R/2A results across example sources (2024 vs 2025).
Run after all few-shot jobs complete:
  python make_few_shot_source_table.py
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

RESULTS   = Path("results")
TABLE_DIR = RESULTS / "tables"
TABLE_DIR.mkdir(exist_ok=True)

ref   = json.loads((RESULTS / "debate_jury_dsv3_46papers_neurips_2025_full.json").read_text())
ids46 = {r["paper_id"] for r in ref["results"]}

MODELS   = [("gemma", "Gemma 3 12B"), ("llama", "Llama 3.3 70B"), ("deepseek", "DeepSeek V3.2")]
SOURCES  = [
    ("neurips_2025_full", "NeurIPS 2025 (in-dist)"),
    ("neurips_2024",      "NeurIPS 2024 (out-of-dist)"),
    ("iclr_reviewhub",    "ICLR 2025 (out-of-dist)"),
]
TRUNC    = 30


def ci_half(arr: np.ndarray, n: int = 2000, seed: int = 10718) -> float:
    rng = np.random.default_rng(seed)
    if len(arr) < 2:
        return float("nan")
    s = [np.mean(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
    return (float(np.percentile(s, 97.5)) - float(np.percentile(s, 2.5))) / 2


def metrics(results: list[dict]) -> dict | None:
    ev = [r for r in results if r.get("prediction") and r["paper_id"] in ids46]
    if not ev:
        return None
    acc_a = np.array([1. if r["prediction"] == r["ground_truth"] else 0. for r in ev])
    fpr_a = np.array([1. if r["prediction"] == "ACCEPT" else 0.
                      for r in ev if r["ground_truth"] == "REJECT"])
    tpr_a = np.array([1. if r["prediction"] == "ACCEPT" else 0.
                      for r in ev if r["ground_truth"] == "ACCEPT"])
    prec  = sum(1 for r in ev if r["prediction"] == "ACCEPT" and r["ground_truth"] == "ACCEPT")
    prec_d= sum(1 for r in ev if r["prediction"] == "ACCEPT")
    prec_v= prec / prec_d if prec_d else float("nan")
    tpr_v = float(np.mean(tpr_a))
    f1    = 2 * prec_v * tpr_v / (prec_v + tpr_v) if (prec_v + tpr_v) else float("nan")
    return dict(
        n=len(ev),
        acc=float(np.mean(acc_a)), acc_ci=ci_half(acc_a),
        fpr=float(np.mean(fpr_a)), fpr_ci=ci_half(fpr_a),
        tpr=tpr_v,                  tpr_ci=ci_half(tpr_a),
        f1=f1,
    )


def tokens_pp(path: Path) -> tuple[float, float]:
    d = json.loads(path.read_text())
    u = d.get("usage", {})
    n = len([r for r in d.get("results", []) if r.get("prediction")])
    if not n:
        return 0.0, 0.0
    return u.get("input_tokens", 0) / n, u.get("output_tokens", 0) / n


def fmt(val: float, ci: float) -> str:
    if val is None or np.isnan(val):
        return "—"
    return f"{val:.3f} ±{ci:.3f}"


lines = [
    "# Few-Shot 3R/2A — Example Source Comparison (No Reviews)\n",
    "> Evaluated on canonical 46-paper balanced set (NeurIPS 2025).\n"
    "> Acc / FPR / TPR = value ± half-width of 95% bootstrap CI (2000 resamples).\n"
    "> 2025 = in-distribution examples (eval papers excluded); 2024 = out-of-distribution.\n",
    "| Model | Example source | n | Acc | FPR | TPR | F1 | In tok/paper | Out tok/paper |",
    "|---|---|---|---|---|---|---|---|---|",
]

for model_key, model_label in MODELS:
    first = True
    for src_key, src_label in SOURCES:
        p = RESULTS / f"few_shot_3r2a_{model_key}_{src_key}_trunc{TRUNC}_46papers_neurips_2025_full.json"
        lbl = model_label if first else ""
        first = False
        if not p.exists():
            lines.append(f"| {lbl} | {src_label} | — | — | — | — | — | — | — |")
            continue
        d   = json.loads(p.read_text())
        m   = metrics(d.get("results", []))
        i_pp, o_pp = tokens_pp(p)
        if m is None:
            lines.append(f"| {lbl} | {src_label} | — | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {lbl} | {src_label} | {m['n']} "
                f"| {fmt(m['acc'], m['acc_ci'])} "
                f"| {fmt(m['fpr'], m['fpr_ci'])} "
                f"| {fmt(m['tpr'], m['tpr_ci'])} "
                f"| {m['f1']:.3f} "
                f"| {i_pp:,.0f} | {o_pp:,.0f} |"
            )

out = TABLE_DIR / "few_shot_source_comparison.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved to {out}")
