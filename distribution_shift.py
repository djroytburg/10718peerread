#!/usr/bin/env python3
"""
Distribution shift statistics comparing our NeurIPS corpora (2023, 2024, 2025)
against the original PeerRead corpus (iclr_2017 + acl_2017 + conll_2016).

Metadata statistics: acceptance rate, review scores, reviewer confidence,
  aspect ratings (originality, clarity, impact, soundness), reviews per paper,
  authors per paper, keywords per paper.
Length statistics: abstract length, review comment length (word count).
"""

import json
import glob
import os
import collections
import math
import statistics
from pathlib import Path

REPO = Path(__file__).parent

# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus(review_files, conference_filter=None):
    papers = []
    for f in review_files:
        try:
            d = json.load(open(f))
            if conference_filter and d.get("conference") != conference_filter:
                continue
            papers.append(d)
        except Exception:
            pass
    return papers


def get_peerread_files():
    """Load original PeerRead: iclr_2017, acl_2017, conll_2016."""
    files = []
    for corpus in ["iclr_2017", "acl_2017", "conll_2016"]:
        files += glob.glob(str(REPO / "PeerRead/data" / corpus / "*/reviews/*.json"))
    return files


def get_neurips_files(year):
    return glob.glob(str(REPO / f"output/neurips_{year}/reviews/*.json"))


# ── Feature extraction ────────────────────────────────────────────────────────

def word_count(text):
    if not text:
        return 0
    return len(str(text).split())


def to_float(v):
    """Convert a value to float, returning None if not numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def extract_features(papers):
    """Return dict of lists, one value per paper or per review."""
    paper_feats = collections.defaultdict(list)
    review_feats = collections.defaultdict(list)

    for d in papers:
        reviews = [r for r in d.get("reviews", []) if not r.get("IS_META_REVIEW", False)]

        # --- paper-level ---
        paper_feats["accepted"].append(int(bool(d.get("accepted"))))
        paper_feats["n_authors"].append(len(d.get("authors") or []))
        paper_feats["n_keywords"].append(len(d.get("keywords") or []))
        paper_feats["abstract_words"].append(word_count(d.get("abstract", "")))
        paper_feats["n_reviews"].append(len(reviews))

        # aggregate review scores per paper
        recs = [to_float(r["RECOMMENDATION"]) for r in reviews if "RECOMMENDATION" in r]
        recs = [v for v in recs if v is not None]
        if recs:
            paper_feats["mean_recommendation"].append(statistics.mean(recs))
            paper_feats["std_recommendation"].append(statistics.stdev(recs) if len(recs) > 1 else 0.0)

        confs = [to_float(r["REVIEWER_CONFIDENCE"]) for r in reviews if "REVIEWER_CONFIDENCE" in r]
        confs = [v for v in confs if v is not None]
        if confs:
            paper_feats["mean_confidence"].append(statistics.mean(confs))

        for aspect in ["ORIGINALITY", "CLARITY", "IMPACT", "SOUNDNESS_CORRECTNESS"]:
            vals = [to_float(r[aspect]) for r in reviews if aspect in r]
            vals = [v for v in vals if v is not None]
            if vals:
                paper_feats[f"mean_{aspect.lower()}"].append(statistics.mean(vals))

        # --- review-level ---
        for r in reviews:
            review_feats["comment_words"].append(word_count(r.get("comments", "")))
            rec = to_float(r.get("RECOMMENDATION"))
            if rec is not None:
                review_feats["recommendation"].append(rec)
            conf = to_float(r.get("REVIEWER_CONFIDENCE"))
            if conf is not None:
                review_feats["confidence"].append(conf)
            for aspect in ["ORIGINALITY", "CLARITY", "IMPACT", "SOUNDNESS_CORRECTNESS"]:
                v = to_float(r.get(aspect))
                if v is not None:
                    review_feats[aspect.lower()].append(v)

    return dict(paper_feats), dict(review_feats)


# ── Statistics ────────────────────────────────────────────────────────────────

def summarize(values):
    if not values:
        return {"n": 0}
    n = len(values)
    mu = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    lo = min(values)
    hi = max(values)
    return {"n": n, "mean": round(mu, 3), "median": round(med, 3),
            "std": round(sd, 3), "min": lo, "max": hi}


def ks_statistic(a, b):
    """Two-sample KS statistic (no scipy needed)."""
    if not a or not b:
        return None
    combined = sorted(set(a) | set(b))
    na, nb = len(a), len(b)
    ca = collections.Counter(a)
    cb = collections.Counter(b)
    cum_a = cum_b = 0
    ks = 0.0
    for v in combined:
        cum_a += ca.get(v, 0) / na
        cum_b += cb.get(v, 0) / nb
        ks = max(ks, abs(cum_a - cum_b))
    return round(ks, 4)


def mean_diff(a, b):
    if not a or not b:
        return None
    return round(statistics.mean(a) - statistics.mean(b), 3)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_sd = math.sqrt((statistics.variance(a) + statistics.variance(b)) / 2)
    if pooled_sd == 0:
        return 0.0
    return round((statistics.mean(a) - statistics.mean(b)) / pooled_sd, 3)


# ── Printing ──────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_feature_table(feature, corpora_data, label=""):
    """Print summary stats + shift metrics for one feature across corpora."""
    print(f"\n  [{feature}{' (' + label + ')' if label else ''}]")
    header = f"  {'Corpus':<25} {'n':>6} {'mean':>8} {'median':>8} {'std':>8} {'min':>6} {'max':>6}"
    print(header)
    print("  " + "-" * 68)

    summaries = {}
    for name, vals in corpora_data.items():
        s = summarize(vals)
        summaries[name] = (s, vals)
        if s["n"] == 0:
            print(f"  {name:<25} {'—':>6}")
        else:
            print(f"  {name:<25} {s['n']:>6} {s['mean']:>8} {s['median']:>8} "
                  f"{s['std']:>8} {str(s['min']):>6} {str(s['max']):>6}")

    # Shift vs PeerRead
    if "PeerRead" in summaries and summaries["PeerRead"][0]["n"] > 0:
        pr_vals = summaries["PeerRead"][1]
        print(f"\n  Shift vs PeerRead — {'Corpus':<20} {'Δmean':>8} {'Cohen d':>9} {'KS':>7}")
        print("  " + "-" * 50)
        for name, (s, vals) in summaries.items():
            if name == "PeerRead" or s["n"] == 0:
                continue
            dm = mean_diff(vals, pr_vals)
            cd = cohens_d(vals, pr_vals)
            ks = ks_statistic(vals, pr_vals)
            print(f"  {'vs ' + name:<25} {str(dm):>8} {str(cd):>9} {str(ks):>7}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading corpora...")

    corpora_raw = {
        "PeerRead":      load_corpus(get_peerread_files()),
        "NeurIPS 2023":  load_corpus(get_neurips_files("2023"),    conference_filter="NeurIPS 2023"),
        "NeurIPS 2024":  load_corpus(get_neurips_files("2024"),    conference_filter="NeurIPS 2024"),
        "NeurIPS 2025":  load_corpus(get_neurips_files("2025_full"), conference_filter="NeurIPS 2025"),
    }

    for name, papers in corpora_raw.items():
        print(f"  {name:<20}: {len(papers):>5} papers loaded")

    # Extract features
    paper_data = {}
    review_data = {}
    for name, papers in corpora_raw.items():
        pf, rf = extract_features(papers)
        paper_data[name] = pf
        review_data[name] = rf

    # ── PAPER-LEVEL METADATA ─────────────────────────────────────────────────
    print_section("PAPER-LEVEL METADATA")

    for feat, label in [
        ("accepted",           "binary 0/1"),
        ("n_authors",          "count"),
        ("n_keywords",         "count"),
        ("n_reviews",          "non-meta reviews per paper"),
        ("abstract_words",     "word count"),
        ("mean_recommendation","avg reviewer score"),
        ("std_recommendation", "score disagreement"),
        ("mean_confidence",    "avg reviewer confidence"),
        ("mean_originality",   "avg originality rating"),
        ("mean_clarity",       "avg clarity rating"),
        ("mean_impact",        "avg impact rating"),
        ("mean_soundness_correctness", "avg soundness rating"),
    ]:
        corpora_feat = {name: pf.get(feat, []) for name, pf in paper_data.items()}
        print_feature_table(feat, corpora_feat, label)

    # ── REVIEW-LEVEL ─────────────────────────────────────────────────────────
    print_section("REVIEW-LEVEL STATISTICS")

    for feat, label in [
        ("comment_words",    "word count per review"),
        ("recommendation",   "per-review score"),
        ("confidence",       "per-review confidence"),
        ("originality",      "per-review"),
        ("clarity",          "per-review"),
        ("impact",           "per-review"),
        ("soundness_correctness", "per-review"),
    ]:
        corpora_feat = {name: rf.get(feat, []) for name, rf in review_data.items()}
        print_feature_table(feat, corpora_feat, label)

    # ── COVERAGE SUMMARY ─────────────────────────────────────────────────────
    print_section("FIELD COVERAGE (% of papers with field populated)")
    fields = ["mean_recommendation", "mean_confidence", "mean_originality",
              "mean_clarity", "mean_impact", "mean_soundness_correctness"]
    print(f"\n  {'Field':<35}", end="")
    for name in paper_data:
        print(f" {name[:12]:>12}", end="")
    print()
    print("  " + "-" * (35 + 13 * len(paper_data)))
    for feat in fields:
        print(f"  {feat:<35}", end="")
        for name, pf in paper_data.items():
            n_total = len(pf.get("accepted", []))
            n_field = len(pf.get(feat, []))
            pct = f"{100*n_field/n_total:.0f}%" if n_total else "—"
            print(f" {pct:>12}", end="")
        print()

    print("\n")


if __name__ == "__main__":
    main()
