#!/usr/bin/env python3
"""
Print distribution shift statistics as compact markdown tables.
"""

import json, glob, collections, math, statistics
from pathlib import Path

REPO = Path(__file__).parent

def load_corpus(files, conference_filter=None):
    papers = []
    for f in files:
        try:
            d = json.load(open(f))
            if conference_filter and d.get("conference") != conference_filter:
                continue
            papers.append(d)
        except Exception:
            pass
    return papers

def get_peerread_files():
    files = []
    for corpus in ["iclr_2017", "acl_2017", "conll_2016"]:
        files += glob.glob(str(REPO / "PeerRead/data" / corpus / "*/reviews/*.json"))
    return files

def word_count(text):
    return len(str(text).split()) if text else 0

def to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def extract(papers):
    pf = collections.defaultdict(list)
    rf = collections.defaultdict(list)
    for d in papers:
        reviews = [r for r in d.get("reviews", []) if not r.get("IS_META_REVIEW", False)]
        pf["accepted"].append(int(bool(d.get("accepted"))))
        pf["n_authors"].append(len(d.get("authors") or []))
        pf["n_keywords"].append(len(d.get("keywords") or []))
        pf["abstract_words"].append(word_count(d.get("abstract", "")))
        pf["n_reviews"].append(len(reviews))
        for key, dest in [("RECOMMENDATION","mean_rec"), ("REVIEWER_CONFIDENCE","mean_conf")]:
            vals = [to_float(r[key]) for r in reviews if key in r]
            vals = [v for v in vals if v is not None]
            if vals:
                pf[dest].append(statistics.mean(vals))
        for aspect in ["ORIGINALITY","CLARITY","IMPACT","SOUNDNESS_CORRECTNESS"]:
            vals = [to_float(r[aspect]) for r in reviews if aspect in r]
            vals = [v for v in vals if v is not None]
            if vals:
                pf[f"mean_{aspect.lower()}"].append(statistics.mean(vals))
        if len([r for r in reviews if "RECOMMENDATION" in r]) > 1:
            recs = [to_float(r["RECOMMENDATION"]) for r in reviews if "RECOMMENDATION" in r]
            recs = [v for v in recs if v is not None]
            if len(recs) > 1:
                pf["std_rec"].append(statistics.stdev(recs))
        for r in reviews:
            rf["comment_words"].append(word_count(r.get("comments","")))
            rec = to_float(r.get("RECOMMENDATION"))
            if rec is not None: rf["recommendation"].append(rec)
            conf = to_float(r.get("REVIEWER_CONFIDENCE"))
            if conf is not None: rf["confidence"].append(conf)
            for aspect in ["ORIGINALITY","CLARITY","IMPACT","SOUNDNESS_CORRECTNESS"]:
                v = to_float(r.get(aspect))
                if v is not None: rf[aspect.lower()].append(v)
    return dict(pf), dict(rf)

def stats(vals):
    if not vals: return None
    n = len(vals)
    return {
        "n": n,
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if n > 1 else 0.0,
    }

def ks(a, b):
    if not a or not b: return None
    na, nb = len(a), len(b)
    ca, cb = collections.Counter(a), collections.Counter(b)
    cum_a = cum_b = ks_val = 0.0
    for v in sorted(set(a) | set(b)):
        cum_a += ca.get(v, 0) / na
        cum_b += cb.get(v, 0) / nb
        ks_val = max(ks_val, abs(cum_a - cum_b))
    return ks_val

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2: return None
    pooled = math.sqrt((statistics.variance(a) + statistics.variance(b)) / 2)
    return (statistics.mean(a) - statistics.mean(b)) / pooled if pooled else 0.0

def fmt(v, decimals=2):
    if v is None: return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

def fmt_n(n):
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)

# ── Build data ────────────────────────────────────────────────────────────────

print("Loading...", flush=True)
corpora_raw = {
    "PeerRead":     load_corpus(get_peerread_files()),
    "NeurIPS 2023": load_corpus(glob.glob(str(REPO / "output/neurips_2023/reviews/*.json")),
                                conference_filter="NeurIPS 2023"),
    "NeurIPS 2024": load_corpus(glob.glob(str(REPO / "output/neurips_2024/reviews/*.json")),
                                conference_filter="NeurIPS 2024"),
    "NeurIPS 2025": load_corpus(glob.glob(str(REPO / "output/neurips_2025_full/reviews/*.json")),
                                conference_filter="NeurIPS 2025"),
}

paper_feats, review_feats = {}, {}
for name, papers in corpora_raw.items():
    pf, rf = extract(papers)
    paper_feats[name] = pf
    review_feats[name] = rf

CORPORA = ["PeerRead", "NeurIPS 2023", "NeurIPS 2024", "NeurIPS 2025"]
SHORT   = ["PeerRead",  "N'IPS 2023",  "N'IPS 2024",  "N'IPS 2025"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def md_table(title, rows, col_headers, col_widths):
    """Print a markdown table."""
    print(f"\n### {title}\n")
    # header
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(col_headers, col_widths)) + " |"
    sep    = "| " + " | ".join("-" * w for w in col_widths) + " |"
    print(header)
    print(sep)
    for row in rows:
        cells = "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |"
        print(cells)

def build_summary_rows(feat_key, feats_dict, label, decimals=2):
    """One row per corpus: Feature | label | n | mean | median | std"""
    rows = []
    for name, short in zip(CORPORA, SHORT):
        vals = feats_dict.get(name, {}).get(feat_key, [])
        s = stats(vals)
        if s:
            rows.append([short, fmt_n(s["n"]), fmt(s["mean"], decimals),
                         fmt(s["median"], decimals), fmt(s["std"], decimals)])
        else:
            rows.append([short, "—", "—", "—", "—"])
    return rows

def build_shift_rows(feat_key, feats_dict):
    """One row per NeurIPS corpus showing shift vs PeerRead."""
    pr = feats_dict.get("PeerRead", {}).get(feat_key, [])
    rows = []
    for name, short in zip(CORPORA[1:], SHORT[1:]):
        vals = feats_dict.get(name, {}).get(feat_key, [])
        if vals and pr:
            dm  = fmt(statistics.mean(vals) - statistics.mean(pr), 3)
            cd  = fmt(cohens_d(vals, pr), 3)
            ks_ = fmt(ks(vals, pr), 3)
            cov = f"{100*len(vals)//len(feats_dict[name].get('accepted',[1])):.0f}%" \
                  if feats_dict[name].get("accepted") else "—"
        else:
            dm = cd = ks_ = cov = "—"
        rows.append([short, dm, cd, ks_, cov])
    return rows

# ── TABLE 1: Paper-level summary ──────────────────────────────────────────────

print("\n\n## Distribution Shift Statistics\n")
print(f"*Corpora: PeerRead (n={fmt_n(len(corpora_raw['PeerRead']))}), "
      f"NeurIPS 2023 (n={fmt_n(len(corpora_raw['NeurIPS 2023']))}), "
      f"NeurIPS 2024 (n={fmt_n(len(corpora_raw['NeurIPS 2024']))}), "
      f"NeurIPS 2025 (n={fmt_n(len(corpora_raw['NeurIPS 2025']))})*\n")

# ── Paper metadata summary ────────────────────────────────────────────────────
rows = []
paper_fields = [
    ("accepted",           "Acceptance rate",          "% accepted",    0),
    ("n_authors",          "Authors / paper",          "count",         2),
    ("n_keywords",         "Keywords / paper",         "count",         2),
    ("abstract_words",     "Abstract length",          "words",         1),
    ("n_reviews",          "Reviews / paper",          "count",         2),
    ("mean_rec",           "Mean reviewer score",      "1–10 scale",    2),
    ("std_rec",            "Score std dev",            "disagreement",  2),
    ("mean_conf",          "Mean reviewer confidence", "1–5 scale",     2),
    ("mean_originality",   "Originality",              "1–5 scale",     2),
    ("mean_clarity",       "Clarity",                  "1–5 scale",     2),
    ("mean_impact",        "Impact",                   "1–5 scale",     2),
    ("mean_soundness_correctness", "Soundness",        "1–5 scale",     2),
]

# Summary table
summary_rows = []
for feat, label, unit, dec in paper_fields:
    for name, short in zip(CORPORA, SHORT):
        vals = paper_feats.get(name, {}).get(feat, [])
        s = stats(vals)
        n_papers = len(paper_feats.get(name, {}).get("accepted", []))
        cov = f"({100*len(vals)//n_papers:.0f}%)" if n_papers and vals else "(—)"
        if s:
            summary_rows.append([
                label if name == "PeerRead" else "",
                unit  if name == "PeerRead" else "",
                short,
                fmt_n(s["n"]) + " " + cov,
                fmt(s["mean"], dec),
                fmt(s["median"], dec),
                fmt(s["std"], dec),
            ])
        else:
            summary_rows.append([
                label if name == "PeerRead" else "",
                unit  if name == "PeerRead" else "",
                short, "—", "—", "—", "—"
            ])

md_table(
    "Paper-level Summary Statistics",
    summary_rows,
    ["Feature", "Unit", "Corpus", "n (coverage)", "Mean", "Median", "Std"],
    [30, 12, 12, 16, 8, 8, 8],
)

# ── Review-level summary ──────────────────────────────────────────────────────
review_fields = [
    ("comment_words",    "Review length",    "words",    1),
    ("recommendation",   "Score",            "1–10",     2),
    ("confidence",       "Confidence",       "1–5",      2),
    ("originality",      "Originality",      "1–5",      2),
    ("clarity",          "Clarity",          "1–5",      2),
    ("impact",           "Impact",           "1–5",      2),
    ("soundness_correctness", "Soundness",   "1–5",      2),
]

rev_rows = []
for feat, label, unit, dec in review_fields:
    for name, short in zip(CORPORA, SHORT):
        vals = review_feats.get(name, {}).get(feat, [])
        s = stats(vals)
        if s:
            rev_rows.append([
                label if name == "PeerRead" else "",
                unit  if name == "PeerRead" else "",
                short, fmt_n(s["n"]),
                fmt(s["mean"], dec), fmt(s["median"], dec), fmt(s["std"], dec),
            ])
        else:
            rev_rows.append([
                label if name == "PeerRead" else "",
                unit  if name == "PeerRead" else "",
                short, "—", "—", "—", "—"
            ])

md_table(
    "Review-level Summary Statistics",
    rev_rows,
    ["Feature", "Unit", "Corpus", "n", "Mean", "Median", "Std"],
    [20, 8, 12, 8, 8, 8, 8],
)

# ── Distribution shift table ──────────────────────────────────────────────────

all_shift_rows = []
for feat, label, unit, dec in paper_fields:
    pr = paper_feats.get("PeerRead", {}).get(feat, [])
    for name, short in zip(CORPORA[1:], SHORT[1:]):
        vals = paper_feats.get(name, {}).get(feat, [])
        is_first = (name == "NeurIPS 2023")
        if vals and pr:
            dm  = fmt(statistics.mean(vals) - statistics.mean(pr), 3)
            cd  = fmt(cohens_d(vals, pr), 3)
            ks_ = fmt(ks(vals, pr), 3)
        else:
            dm = cd = ks_ = "—"
        all_shift_rows.append([
            label if is_first else "",
            short, dm, cd, ks_
        ])

md_table(
    "Distribution Shift vs PeerRead (paper-level)",
    all_shift_rows,
    ["Feature", "Corpus", "Δ Mean", "Cohen's d", "KS stat"],
    [30, 12, 10, 10, 10],
)

rev_shift_rows = []
for feat, label, unit, dec in review_fields:
    pr = review_feats.get("PeerRead", {}).get(feat, [])
    for name, short in zip(CORPORA[1:], SHORT[1:]):
        vals = review_feats.get(name, {}).get(feat, [])
        is_first = (name == "NeurIPS 2023")
        if vals and pr:
            dm  = fmt(statistics.mean(vals) - statistics.mean(pr), 3)
            cd  = fmt(cohens_d(vals, pr), 3)
            ks_ = fmt(ks(vals, pr), 3)
        else:
            dm = cd = ks_ = "—"
        rev_shift_rows.append([
            label if is_first else "",
            short, dm, cd, ks_
        ])

md_table(
    "Distribution Shift vs PeerRead (review-level)",
    rev_shift_rows,
    ["Feature", "Corpus", "Δ Mean", "Cohen's d", "KS stat"],
    [22, 12, 10, 10, 10],
)
