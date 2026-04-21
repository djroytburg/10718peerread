"""
Confidence intervals for binary classification metrics (FPR, TPR, Accuracy).

At n=10 per class, Clopper-Pearson (exact binomial) is the right default:
- Wald/Normal intervals break down near boundaries and at small n
- CP gives guaranteed coverage and handles TPR=1.0/FPR=0.0 naturally
- Stratified bootstrap provided as an assumption-free complement

Method justification from agent review:
  CP chosen over Wilson because boundary cases (TPR=1.0, FPR=0.0) are common
  in this dataset. CP's beta-quantile formulation handles k=n and k=0 without
  ad-hoc corrections. Stratified bootstrap preserves class-size denominators
  so FPR/TPR variance is not inflated by unstratified resampling.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Exact (Clopper-Pearson) two-sided CI for a binomial proportion k/n.
    Handles k=0 and k=n boundary cases without ad-hoc correction.
    """
    if n == 0:
        return (0.0, 1.0)
    lower = beta_dist.ppf(alpha / 2,     a=k,     b=n - k + 1) if k > 0 else 0.0
    upper = beta_dist.ppf(1 - alpha / 2, a=k + 1, b=n - k)     if k < n else 1.0
    return (float(lower), float(upper))


def compute_cis(results: list[dict], alpha: float = 0.05) -> dict:
    """
    Clopper-Pearson CIs for TPR, FPR, and Accuracy.

    Parameters
    ----------
    results : list of dicts with keys:
                  "prediction"   : "ACCEPT" | "REJECT" | None
                  "ground_truth" : "ACCEPT" | "REJECT"
    alpha   : significance level (default 0.05 → 95% CI)

    Returns
    -------
    {
        "tpr": {"point": float, "lower": float, "upper": float},
        "fpr": {"point": float, "lower": float, "upper": float},
        "acc": {"point": float, "lower": float, "upper": float},
    }
    """
    evaluated = [r for r in results if r.get("prediction") is not None]
    if not evaluated:
        raise ValueError("No evaluated results (all predictions are None).")

    positives = [r for r in evaluated if r["ground_truth"] == "ACCEPT"]
    negatives = [r for r in evaluated if r["ground_truth"] == "REJECT"]

    tp = sum(1 for r in positives if r["prediction"] == "ACCEPT")
    fp = sum(1 for r in negatives if r["prediction"] == "ACCEPT")
    tn = len(negatives) - fp
    fn = len(positives) - tp

    n_pos, n_neg, n_all = tp + fn, fp + tn, len(evaluated)

    tpr_pt = tp / n_pos if n_pos > 0 else float("nan")
    fpr_pt = fp / n_neg if n_neg > 0 else float("nan")
    acc_pt = (tp + tn) / n_all

    tpr_lo, tpr_hi = _clopper_pearson(tp, n_pos, alpha) if n_pos > 0 else (float("nan"), float("nan"))
    fpr_lo, fpr_hi = _clopper_pearson(fp, n_neg, alpha) if n_neg > 0 else (float("nan"), float("nan"))
    acc_lo, acc_hi = _clopper_pearson(tp + tn, n_all, alpha)

    return {
        "tpr": {"point": tpr_pt, "lower": tpr_lo, "upper": tpr_hi},
        "fpr": {"point": fpr_pt, "lower": fpr_lo, "upper": fpr_hi},
        "acc": {"point": acc_pt, "lower": acc_lo, "upper": acc_hi},
    }


def bootstrap_cis(
    results: list[dict],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Stratified bootstrap CIs for TPR, FPR, and Accuracy.

    Stratifies by ground-truth class so denominators of FPR/TPR are fixed
    across resamples (avoids inflating variance via fluctuating class sizes).
    Uses the percentile method; point estimates come from original data.
    """
    evaluated = [r for r in results if r.get("prediction") is not None]
    if not evaluated:
        raise ValueError("No evaluated results.")

    positives = [r for r in evaluated if r["ground_truth"] == "ACCEPT"]
    negatives = [r for r in evaluated if r["ground_truth"] == "REJECT"]
    if not positives or not negatives:
        raise ValueError("Need at least one example from each class.")

    rng = np.random.default_rng(seed)
    pos_arr = np.array([1 if r["prediction"] == "ACCEPT" else 0 for r in positives])
    neg_arr = np.array([1 if r["prediction"] == "ACCEPT" else 0 for r in negatives])
    n_pos, n_neg = len(pos_arr), len(neg_arr)

    tpr_boot = np.empty(n_boot)
    fpr_boot = np.empty(n_boot)
    acc_boot = np.empty(n_boot)

    for i in range(n_boot):
        pos_s = rng.choice(pos_arr, size=n_pos, replace=True)
        neg_s = rng.choice(neg_arr, size=n_neg, replace=True)
        tp_b = pos_s.sum()
        fp_b = neg_s.sum()
        tpr_boot[i] = tp_b / n_pos
        fpr_boot[i] = fp_b / n_neg
        acc_boot[i] = (tp_b + (n_neg - fp_b)) / (n_pos + n_neg)

    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    tp = int(pos_arr.sum())
    fp = int(neg_arr.sum())

    return {
        "tpr": {"point": tp / n_pos,
                "lower": float(np.percentile(tpr_boot, lo)),
                "upper": float(np.percentile(tpr_boot, hi))},
        "fpr": {"point": fp / n_neg,
                "lower": float(np.percentile(fpr_boot, lo)),
                "upper": float(np.percentile(fpr_boot, hi))},
        "acc": {"point": (tp + (n_neg - fp)) / (n_pos + n_neg),
                "lower": float(np.percentile(acc_boot, lo)),
                "upper": float(np.percentile(acc_boot, hi))},
    }
