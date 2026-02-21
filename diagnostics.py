#!/usr/bin/env python3
"""
PeerRead Data Quality Diagnostics
==================================
Implements automated diagnostics for four high-impact data problems identified
in DATA_PROBLEMS.md, with visualizations and JSON statistics output.

Diagnostics implemented:
  R1 - Duplicate Reviews Within the Same Paper
  P1 - Section Ordering is Alphabetical, Not Document Order
  F1 - Author/Email Metadata Mixed into Paper Content (institutional email vs acceptance)
  S2 - Inconsistent Acceptance Labels Across Venues

Usage:
    source .venv/bin/activate
    python diagnostics.py
"""

import json
import glob
import os
import hashlib
import re
from collections import defaultdict, Counter

import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Font configuration
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
VOLKHOV_DIR = os.path.join(HOME, "fonts", "Volkhov")
UBUNTU_MONO_DIR = os.path.join(HOME, "fonts", "Ubuntu_Mono")

_volkhov_regular = os.path.join(VOLKHOV_DIR, "Volkhov-Regular.ttf")
_volkhov_bold = os.path.join(VOLKHOV_DIR, "Volkhov-Bold.ttf")
_ubuntu_mono_regular = os.path.join(UBUNTU_MONO_DIR, "UbuntuMono-Regular.ttf")
_ubuntu_mono_bold = os.path.join(UBUNTU_MONO_DIR, "UbuntuMono-Bold.ttf")

for _fp in [_volkhov_regular, _volkhov_bold, _ubuntu_mono_regular, _ubuntu_mono_bold]:
    if os.path.isfile(_fp):
        fm.fontManager.addfont(_fp)

FONT_TITLE = fm.FontProperties(fname=_volkhov_bold, size=14)
FONT_TITLE_SM = fm.FontProperties(fname=_volkhov_bold, size=12)
FONT_SUPTITLE = fm.FontProperties(fname=_volkhov_bold, size=16)
FONT_AXIS = fm.FontProperties(fname=_ubuntu_mono_regular, size=10)
FONT_AXIS_BOLD = fm.FontProperties(fname=_ubuntu_mono_bold, size=10)
FONT_TICK = fm.FontProperties(fname=_ubuntu_mono_regular, size=9)
FONT_LEGEND = fm.FontProperties(fname=_ubuntu_mono_regular, size=9)
FONT_ANNOT = fm.FontProperties(fname=_ubuntu_mono_regular, size=8)
FONT_ANNOT_LG = fm.FontProperties(fname=_ubuntu_mono_regular, size=10)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "PeerRead", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "diagnostic_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VENUES = [
    "iclr_2017",
    "acl_2017",
    "conll_2016",
    "arxiv.cs.cl_2007-2017",
    "arxiv.cs.ai_2007-2017",
    "arxiv.cs.lg_2007-2017",
]

VENUE_SHORT = {
    "iclr_2017": "ICLR'17",
    "acl_2017": "ACL'17",
    "conll_2016": "CoNLL'16",
    "arxiv.cs.cl_2007-2017": "arXiv CL",
    "arxiv.cs.ai_2007-2017": "arXiv AI",
    "arxiv.cs.lg_2007-2017": "arXiv LG",
}


def _set_tick_fonts(ax):
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_TICK)
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_TICK)


def _nice_venue(v):
    return VENUE_SHORT.get(v, v)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def iter_review_files(venue):
    """Yield (filepath, parsed_json) for every review JSON in all splits."""
    venue_dir = os.path.join(DATA_ROOT, venue)
    for split in ("train", "dev", "test"):
        pattern = os.path.join(venue_dir, split, "reviews", "*.json")
        for fp in sorted(glob.glob(pattern)):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    first_line = fh.readline().strip()
                    if not first_line:
                        continue
                    data = json.loads(first_line)
                yield fp, data
            except Exception:
                continue


def iter_parsed_pdf_files(venue):
    """Yield (filepath, parsed_json) for every parsed PDF JSON in all splits."""
    venue_dir = os.path.join(DATA_ROOT, venue)
    for split in ("train", "dev", "test"):
        pattern = os.path.join(venue_dir, split, "parsed_pdfs", "*.json")
        for fp in sorted(glob.glob(pattern)):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                yield fp, data
            except Exception:
                continue


# ===================================================================
# R1 – Duplicate Reviews Within the Same Paper
# ===================================================================

def _hash_review(rev):
    """Create a deterministic hash from review content fields."""
    parts = []
    for key in sorted(rev.keys()):
        parts.append(f"{key}={rev[key]}")
    blob = "||".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


def diagnose_r1():
    """Detect exact-duplicate reviews within individual papers."""
    print("\n" + "=" * 70)
    print("R1: Duplicate Reviews Within the Same Paper")
    print("=" * 70)

    venue_stats = {}

    for venue in VENUES:
        total_papers = 0
        affected_papers = 0
        total_reviews = 0
        total_duplicates = 0
        per_paper_dup_ratios = []

        for fp, data in iter_review_files(venue):
            reviews = data.get("reviews", [])
            if not reviews:
                total_papers += 1
                continue
            total_papers += 1
            total_reviews += len(reviews)

            hashes = [_hash_review(r) for r in reviews]
            hash_counts = Counter(hashes)
            n_unique = len(hash_counts)
            n_total = len(hashes)
            n_dup = n_total - n_unique

            if n_dup > 0:
                affected_papers += 1
                total_duplicates += n_dup
                per_paper_dup_ratios.append(n_dup / n_total)

        dup_ratio = total_duplicates / total_reviews if total_reviews else 0.0
        venue_stats[venue] = {
            "total_papers": total_papers,
            "affected_papers": affected_papers,
            "total_reviews": total_reviews,
            "total_duplicate_reviews": total_duplicates,
            "duplicate_ratio": round(dup_ratio, 4),
            "affected_paper_pct": round(100 * affected_papers / total_papers, 2) if total_papers else 0,
            "mean_per_paper_dup_ratio": round(float(np.mean(per_paper_dup_ratios)), 4) if per_paper_dup_ratios else 0,
        }
        print(f"  {_nice_venue(venue):12s}  papers={total_papers:5d}  "
              f"affected={affected_papers:4d} ({venue_stats[venue]['affected_paper_pct']:5.1f}%)  "
              f"dup_reviews={total_duplicates:5d}/{total_reviews:5d}  "
              f"ratio={dup_ratio:.3f}")

    result = {"diagnostic": "R1_duplicate_reviews", "per_venue": venue_stats}

    # --- Visualization ---
    venues_with_reviews = [v for v in VENUES
                           if venue_stats[v]["total_reviews"] > 0]
    if not venues_with_reviews:
        print("  (no venues with reviews to plot)")
        return result

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("R1: Duplicate Reviews Within the Same Paper",
                 fontproperties=FONT_SUPTITLE, y=0.98)

    # --- Left panel: bar chart of affected paper % ---
    ax = axes[0]
    labels = [_nice_venue(v) for v in venues_with_reviews]
    aff_pcts = [venue_stats[v]["affected_paper_pct"] for v in venues_with_reviews]
    colors = ["#d62728" if p > 0 else "#2ca02c" for p in aff_pcts]
    bars = ax.bar(labels, aff_pcts, color=colors, edgecolor="black", linewidth=0.6)
    for bar, pct in zip(bars, aff_pcts):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{pct:.1f}%", ha="center", va="bottom",
                    fontproperties=FONT_ANNOT_LG, color="#d62728")
    ax.set_ylabel("Papers with duplicated reviews (%)",
                   fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Affected Papers by Venue", fontproperties=FONT_TITLE)
    ax.set_ylim(0, max(aff_pcts) * 1.25 + 1)
    _set_tick_fonts(ax)

    # --- Right panel: stacked bar of unique vs duplicate reviews ---
    ax2 = axes[1]
    unique_counts = [venue_stats[v]["total_reviews"] - venue_stats[v]["total_duplicate_reviews"]
                     for v in venues_with_reviews]
    dup_counts = [venue_stats[v]["total_duplicate_reviews"] for v in venues_with_reviews]
    x_pos = np.arange(len(venues_with_reviews))
    w = 0.55
    ax2.bar(x_pos, unique_counts, w, label="Unique", color="#1f77b4",
            edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos, dup_counts, w, bottom=unique_counts, label="Duplicate",
            color="#d62728", edgecolor="black", linewidth=0.5)
    for i, (u, d) in enumerate(zip(unique_counts, dup_counts)):
        if d > 0:
            ax2.text(i, u + d + 5, f"+{d}", ha="center", va="bottom",
                     fontproperties=FONT_ANNOT, color="#d62728")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Number of review entries", fontproperties=FONT_AXIS_BOLD)
    ax2.set_title("Review Composition by Venue", fontproperties=FONT_TITLE)
    legend = ax2.legend(prop=FONT_LEGEND, framealpha=0.9)
    _set_tick_fonts(ax2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "r1_duplicate_reviews.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# P1 – Section Ordering: Alphabetical vs Document Order
# ===================================================================

def diagnose_p1():
    """Compare document-order section headings vs sorted (alphabetical) order.

    Computes Kendall tau rank correlation between the two orderings per paper.
    A tau of 1.0 means the alphabetical sort matches document order perfectly.
    Lower tau means more disruption.
    """
    print("\n" + "=" * 70)
    print("P1: Section Ordering – Alphabetical vs Document Order")
    print("=" * 70)

    venue_stats = {}
    all_taus = []
    all_venue_labels = []

    for venue in VENUES:
        taus = []
        n_affected = 0
        n_total = 0
        n_dup_heading = 0  # bonus: P2 duplicate heading detection
        heading_collision_papers = 0

        for fp, data in iter_parsed_pdf_files(venue):
            sections = data.get("metadata", {}).get("sections", None)
            if not sections or len(sections) < 2:
                continue
            n_total += 1

            headings_orig = [s.get("heading") or "" for s in sections]

            # P2 piggyback: detect duplicate headings
            if len(headings_orig) != len(set(headings_orig)):
                heading_collision_papers += 1
                dup_h = [h for h, c in Counter(headings_orig).items() if c > 1]
                n_dup_heading += sum(Counter(headings_orig)[h] - 1 for h in dup_h)

            # Build rank arrays
            # original_ranks: 0, 1, 2, ... (document order)
            # sorted_ranks: the position each heading would appear in sorted()
            sorted_headings = sorted(headings_orig)
            # Map from heading to its sorted position(s)
            # Handle duplicate headings by assigning unique positions via index
            sorted_positions = list(range(len(sorted_headings)))

            # Create a mapping: for each heading, its sorted indices (in order)
            heading_to_sorted_idx = defaultdict(list)
            for idx, h in enumerate(sorted_headings):
                heading_to_sorted_idx[h].append(idx)

            # Build the rank array: for each position in original order,
            # what sorted-position does it get?
            heading_sorted_copy = {h: list(idxs) for h, idxs in heading_to_sorted_idx.items()}
            sorted_rank_of_orig = []
            for h in headings_orig:
                sorted_rank_of_orig.append(heading_sorted_copy[h].pop(0))

            orig_rank = list(range(len(headings_orig)))

            if headings_orig == sorted_headings:
                tau = 1.0
            else:
                n_affected += 1
                result_tau = scipy_stats.kendalltau(orig_rank, sorted_rank_of_orig)
                tau = result_tau.statistic if hasattr(result_tau, 'statistic') else result_tau[0]
                if np.isnan(tau):
                    tau = 1.0

            taus.append(tau)
            all_taus.append(tau)
            all_venue_labels.append(venue)

        mean_tau = float(np.mean(taus)) if taus else 1.0
        venue_stats[venue] = {
            "total_papers": n_total,
            "order_differs": n_affected,
            "order_differs_pct": round(100 * n_affected / n_total, 2) if n_total else 0,
            "mean_kendall_tau": round(mean_tau, 4),
            "median_kendall_tau": round(float(np.median(taus)), 4) if taus else 1.0,
            "min_kendall_tau": round(float(np.min(taus)), 4) if taus else 1.0,
            "heading_collision_papers": heading_collision_papers,
            "total_lost_sections_from_collisions": n_dup_heading,
        }
        print(f"  {_nice_venue(venue):12s}  papers={n_total:5d}  "
              f"order_differs={n_affected:4d} ({venue_stats[venue]['order_differs_pct']:5.1f}%)  "
              f"mean_tau={mean_tau:+.4f}  "
              f"heading_collisions={heading_collision_papers}")

    result = {"diagnostic": "P1_section_ordering", "per_venue": venue_stats}

    if not all_taus:
        print("  (no parsed PDFs found)")
        return result

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("P1: Alphabetical vs Document-Order Section Ordering",
                 fontproperties=FONT_SUPTITLE, y=0.98)

    # Panel 1: Histogram of Kendall tau values (all venues combined)
    ax = axes[0]
    tau_arr = np.array(all_taus)
    bins = np.linspace(-1, 1, 41)
    ax.hist(tau_arr, bins=bins, color="#4c72b0", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(x=1.0, color="#2ca02c", linestyle="--", linewidth=1.5, label="Perfect match (τ=1)")
    ax.axvline(x=np.mean(tau_arr), color="#d62728", linestyle="-", linewidth=1.5,
               label=f"Mean τ={np.mean(tau_arr):.3f}")
    ax.set_xlabel("Kendall τ (doc-order vs alphabetical)", fontproperties=FONT_AXIS_BOLD)
    ax.set_ylabel("Number of papers", fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Rank Correlation Distribution", fontproperties=FONT_TITLE_SM)
    ax.legend(prop=FONT_LEGEND, loc="upper left")
    _set_tick_fonts(ax)

    # Panel 2: Per-venue box plot of tau values
    ax2 = axes[1]
    venue_groups = defaultdict(list)
    for tau, v in zip(all_taus, all_venue_labels):
        venue_groups[v].append(tau)

    plot_venues = [v for v in VENUES if v in venue_groups]
    box_data = [venue_groups[v] for v in plot_venues]
    box_labels = [_nice_venue(v) for v in plot_venues]

    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=1.5))
    colors_box = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.axhline(y=1.0, color="#2ca02c", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Kendall τ", fontproperties=FONT_AXIS_BOLD)
    ax2.set_title("Rank Correlation by Venue", fontproperties=FONT_TITLE_SM)
    ax2.tick_params(axis="x", rotation=30)
    _set_tick_fonts(ax2)

    # Panel 3: Scatter – % of papers affected vs mean tau
    ax3 = axes[2]
    plot_venues_nz = [v for v in plot_venues if venue_stats[v]["total_papers"] > 0]
    xs = [venue_stats[v]["order_differs_pct"] for v in plot_venues_nz]
    ys = [venue_stats[v]["mean_kendall_tau"] for v in plot_venues_nz]
    sizes = [venue_stats[v]["total_papers"] for v in plot_venues_nz]
    max_s = max(sizes) if sizes else 1
    scaled_sizes = [120 * s / max_s + 30 for s in sizes]
    scatter_colors = colors_box[:len(plot_venues_nz)]

    for i, v in enumerate(plot_venues_nz):
        ax3.scatter(xs[i], ys[i], s=scaled_sizes[i], c=scatter_colors[i],
                    edgecolors="black", linewidth=0.7, alpha=0.8, zorder=3)
        ax3.annotate(_nice_venue(v), (xs[i], ys[i]),
                     textcoords="offset points", xytext=(6, 6),
                     fontproperties=FONT_ANNOT_LG)

    ax3.set_xlabel("Papers with ordering mismatch (%)", fontproperties=FONT_AXIS_BOLD)
    ax3.set_ylabel("Mean Kendall τ", fontproperties=FONT_AXIS_BOLD)
    ax3.set_title("Impact: Disruption vs Prevalence", fontproperties=FONT_TITLE_SM)
    ax3.set_xlim(-2, max(xs) * 1.15 + 2 if xs else 100)
    _set_tick_fonts(ax3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "p1_section_ordering.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# F1 – Email Domain / Institutional Affiliation vs Acceptance Confound
# ===================================================================

# ---------------------------------------------------------------------------
# Institution clustering: map email domain suffixes -> (institution name, color)
# Colors are official brand / identity colors sourced from each institution.
# Domain suffixes are expansive — any email ending with the suffix matches.
# ---------------------------------------------------------------------------
INSTITUTION_REGISTRY = [
    # (institution_name, brand_color, [domain_suffixes...])
    # --- US Universities ---
    ("UC Berkeley",   "#003262", ["berkeley.edu"]),
    ("Stanford",      "#8C1515", ["stanford.edu"]),
    ("MIT",           "#A31F34", ["mit.edu"]),
    ("CMU",           "#C41230", ["cmu.edu"]),
    ("Cornell",       "#B31B1B", ["cornell.edu"]),
    ("Princeton",     "#FF8F00", ["princeton.edu"]),
    ("UW",            "#4B2E83", ["washington.edu", "uw.edu"]),
    ("U Michigan",    "#00274C", ["umich.edu"]),
    ("Georgia Tech",  "#B3A369", ["gatech.edu"]),
    ("UIUC",          "#E84A27", ["illinois.edu"]),
    ("NYU",           "#57068C", ["nyu.edu"]),
    ("Columbia",      "#B9D9EB", ["columbia.edu"]),
    ("USC",           "#990000", ["usc.edu"]),
    ("UT Austin",     "#BF5700", ["utexas.edu"]),
    ("UMass",         "#881c1c", ["umass.edu"]),
    ("UCLA",          "#2774AE", ["ucla.edu"]),
    ("UCSD",          "#182B49", ["ucsd.edu"]),
    ("UCSC",          "#003C6C", ["ucsc.edu"]),
    ("Duke",          "#003366", ["duke.edu"]),
    ("Purdue",        "#CEB888", ["purdue.edu"]),
    ("Virginia Tech", "#630031", ["vt.edu"]),
    ("ASU",           "#8C1D40", ["asu.edu"]),
    ("Harvard",       "#A51C30", ["harvard.edu"]),
    ("UMD",           "#E21833", ["umd.edu"]),
    ("Penn",          "#011F5B", ["upenn.edu"]),
    ("TTIC",          "#800000", ["ttic.edu"]),
    ("ISI/USC",       "#990000", ["isi.edu"]),
    # --- Canadian Universities ---
    ("U Toronto",     "#002A5C", ["toronto.edu", "utoronto.ca"]),
    ("U Montreal",    "#0057A7", ["umontreal.ca", "mila.quebec"]),
    ("U Alberta",     "#007C41", ["ualberta.ca"]),
    ("McGill",        "#ED1B2F", ["mcgill.ca"]),
    ("U Waterloo",    "#FFD54F", ["uwaterloo.ca"]),
    ("UBC",           "#002145", ["ubc.ca"]),
    # --- UK Universities ---
    ("Cambridge",     "#A3C1AD", ["cam.ac.uk"]),
    ("Oxford",        "#002147", ["ox.ac.uk"]),
    ("Edinburgh",     "#B30C00", ["ed.ac.uk"]),
    ("UCL",           "#AC145A", ["ucl.ac.uk"]),
    # --- European ---
    ("ETH Zurich",    "#1F407A", ["ethz.ch"]),
    ("EPFL",          "#FF0000", ["epfl.ch"]),
    ("INRIA",         "#E63312", ["inria.fr"]),
    ("MPI",           "#006C66", ["mpg.de"]),
    ("U Amsterdam",   "#BE0028", ["uva.nl"]),
    # --- Asian ---
    ("Tsinghua",      "#660874", ["tsinghua.edu.cn"]),
    ("Peking U",      "#8B0000", ["pku.edu.cn"]),
    ("Fudan",         "#274E90", ["fudan.edu.cn"]),
    ("Seoul Nat'l",   "#0B3D91", ["snu.ac.kr"]),
    ("NTU Singapore", "#C41E3A", ["ntu.edu.sg"]),
    ("HKUST",         "#003DA5", ["ust.hk"]),
    ("Technion",      "#4A7C2F", ["technion.ac.il"]),
    ("Hebrew U",      "#8B4513", ["huji.ac.il"]),
    ("Ben-Gurion",    "#E35205", ["bgu.ac.il"]),
    # --- Industry ---
    ("Google",        "#4285F4", ["google.com", "deepmind.com"]),
    ("Meta / FB",     "#1877F2", ["fb.com", "facebook.com", "meta.com"]),
    ("Microsoft",     "#00A4EF", ["microsoft.com"]),
    ("IBM",           "#0530AD", ["ibm.com"]),
    ("Baidu",         "#2319DC", ["baidu.com"]),
    ("Huawei",        "#CF0A2C", ["huawei.com"]),
    ("Yahoo",         "#6001D2", ["yahoo.com", "yahoo-inc.com"]),
    ("OpenAI",        "#10A37F", ["openai.com"]),
]

# Build lookup: suffix -> (name, color)
_INST_LOOKUP = []  # list of (suffix, name, color), longest-suffix-first
for _name, _color, _suffixes in INSTITUTION_REGISTRY:
    for _suf in _suffixes:
        _INST_LOOKUP.append((_suf.lower(), _name, _color))
_INST_LOOKUP.sort(key=lambda t: -len(t[0]))  # longest suffix first for greedy match

# Fallback color for "Other / unmatched"
_OTHER_COLOR = "#AAAAAA"
_GMAIL_COLOR = "#EA4335"  # Gmail red


def _extract_domain(email):
    """Extract domain from email, returning None on failure."""
    if not email or "@" not in str(email):
        return None
    dom = str(email).split("@", 1)[1].lower().strip()
    # Strip trailing punctuation that Science Parse sometimes leaves
    dom = dom.rstrip(".,;:)>]}")
    return dom if dom else None


def _classify_institution(domain):
    """Map a domain to (institution_name, color).  Returns None if unmatched."""
    if domain is None:
        return None
    for suffix, name, color in _INST_LOOKUP:
        if domain == suffix or domain.endswith("." + suffix):
            return (name, color)
    return None


def _classify_paper_institutions(domains):
    """Return set of (institution_name, color) tuples for a paper's domains."""
    insts = set()
    for d in domains:
        result = _classify_institution(d)
        if result is not None:
            insts.add(result)
    return insts


def diagnose_f1():
    """Measure correlation between institutional email domains and acceptance.

    For venues where both parsed PDFs (with emails) and acceptance labels exist,
    we cluster papers by institution (via email domain) and measure acceptance
    rates per institution, colored by official university / company brand colors.
    """
    print("\n" + "=" * 70)
    print("F1: Email Domain / Institutional Affiliation vs Acceptance")
    print("=" * 70)

    venues_with_acceptance = ["iclr_2017", "arxiv.cs.cl_2007-2017",
                              "arxiv.cs.ai_2007-2017", "arxiv.cs.lg_2007-2017"]

    records = []
    venue_records = defaultdict(list)
    domain_counter = Counter()
    inst_counter = Counter()       # institution name -> paper count
    inst_accepted = defaultdict(list)  # institution name -> [bool, ...]

    for venue in venues_with_acceptance:
        # Load acceptance labels from review files
        acceptance_map = {}
        for fp, data in iter_review_files(venue):
            pid = data.get("id", os.path.basename(fp).replace(".json", ""))
            acc = data.get("accepted", None)
            if acc is not None:
                acceptance_map[pid] = bool(acc)

        # Load email domains from parsed PDFs
        for fp, pdf_data in iter_parsed_pdf_files(venue):
            basename = os.path.basename(fp)
            pid = basename.replace(".pdf.json", "")

            if pid not in acceptance_map:
                continue

            emails = pdf_data.get("metadata", {}).get("emails", []) or []
            domains = [_extract_domain(e) for e in emails if e]
            domains = [d for d in domains if d is not None]

            paper_insts = _classify_paper_institutions(domains)
            has_known_inst = len(paper_insts) > 0
            n_emails = len(emails)

            for d in domains:
                domain_counter[d] += 1

            for inst_name, inst_color in paper_insts:
                inst_counter[inst_name] += 1
                inst_accepted[inst_name].append(acceptance_map[pid])

            rec = {
                "paper_id": pid,
                "venue": venue,
                "accepted": acceptance_map[pid],
                "emails": emails,
                "domains": domains,
                "institutions": paper_insts,
                "has_known_inst": has_known_inst,
                "n_emails": n_emails,
            }
            records.append(rec)
            venue_records[venue].append(rec)

    if not records:
        print("  No matching records found (no papers with both emails and labels).")
        return {"diagnostic": "F1_email_confound", "per_venue": {}}

    # --- Build institution-level stats ---
    inst_color_map = {}
    for name, color, _ in INSTITUTION_REGISTRY:
        inst_color_map[name] = color

    inst_stats = {}
    for inst_name, count in inst_counter.most_common():
        acc_list = inst_accepted[inst_name]
        inst_stats[inst_name] = {
            "papers": count,
            "acceptance_rate": round(float(np.mean(acc_list)), 4),
            "color": inst_color_map.get(inst_name, _OTHER_COLOR),
        }

    # --- Per-venue: known-institution vs other ---
    venue_stats = {}
    overall_known_acc = []
    overall_other_acc = []

    for venue in venues_with_acceptance:
        recs = venue_records.get(venue, [])
        if not recs:
            continue

        known = [r for r in recs if r["has_known_inst"]]
        other = [r for r in recs if not r["has_known_inst"]]
        no_email = [r for r in recs if r["n_emails"] == 0]

        known_rate = np.mean([r["accepted"] for r in known]) if known else float("nan")
        other_rate = np.mean([r["accepted"] for r in other]) if other else float("nan")
        no_email_rate = np.mean([r["accepted"] for r in no_email]) if no_email else float("nan")
        overall_rate = np.mean([r["accepted"] for r in recs])

        if known and other:
            x_binary = np.array([1 if r["has_known_inst"] else 0 for r in recs])
            y_binary = np.array([1 if r["accepted"] else 0 for r in recs])
            corr_result = scipy_stats.pointbiserialr(x_binary, y_binary)
            pb_corr = corr_result.statistic if hasattr(corr_result, 'statistic') else corr_result[0]
            pb_pval = corr_result.pvalue if hasattr(corr_result, 'pvalue') else corr_result[1]
        else:
            pb_corr = float("nan")
            pb_pval = float("nan")

        for r in known:
            overall_known_acc.append(r["accepted"])
        for r in other:
            overall_other_acc.append(r["accepted"])

        venue_stats[venue] = {
            "total_papers_matched": len(recs),
            "known_inst_papers": len(known),
            "other_papers": len(other),
            "no_email_papers": len(no_email),
            "known_inst_acceptance_rate": round(float(known_rate), 4) if not np.isnan(known_rate) else None,
            "other_acceptance_rate": round(float(other_rate), 4) if not np.isnan(other_rate) else None,
            "no_email_acceptance_rate": round(float(no_email_rate), 4) if not np.isnan(no_email_rate) else None,
            "overall_acceptance_rate": round(float(overall_rate), 4),
            "point_biserial_corr": round(float(pb_corr), 4) if not np.isnan(pb_corr) else None,
            "point_biserial_pval": float(pb_pval) if not np.isnan(pb_pval) else None,
        }
        sig_marker = "***" if (not np.isnan(pb_pval) and pb_pval < 0.001) else \
                     "**" if (not np.isnan(pb_pval) and pb_pval < 0.01) else \
                     "*" if (not np.isnan(pb_pval) and pb_pval < 0.05) else ""
        print(f"  {_nice_venue(venue):12s}  n={len(recs):4d}  "
              f"known={len(known):4d} ({100*known_rate:.1f}% acc)  "
              f"other={len(other):4d} ({100*other_rate:.1f}% acc)  "
              f"r_pb={pb_corr:+.3f}{sig_marker}")

    result = {
        "diagnostic": "F1_email_confound",
        "per_venue": venue_stats,
        "institution_stats": inst_stats,
        "overall_known_inst_acceptance_rate": round(float(np.mean(overall_known_acc)), 4) if overall_known_acc else None,
        "overall_other_acceptance_rate": round(float(np.mean(overall_other_acc)), 4) if overall_other_acc else None,
    }

    # ---------------------------------------------------------------
    # Visualization — 3-panel figure
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5))
    fig.suptitle("F1: Institutional Email Affiliation vs Acceptance Rate",
                 fontproperties=FONT_SUPTITLE, y=0.99)

    # --- Panel 1: Per-venue grouped bars (known institution vs other) ---
    ax = axes[0]
    plot_venues = [v for v in venues_with_acceptance if v in venue_stats]
    labels = [_nice_venue(v) for v in plot_venues]
    x_pos = np.arange(len(plot_venues))
    w = 0.3
    known_rates_pct = []
    other_rates_pct = []
    for v in plot_venues:
        kr = venue_stats[v].get("known_inst_acceptance_rate")
        otr = venue_stats[v].get("other_acceptance_rate")
        known_rates_pct.append(100 * kr if kr is not None else 0)
        other_rates_pct.append(100 * otr if otr is not None else 0)

    ax.bar(x_pos - w / 2, known_rates_pct, w, label="Known institution",
           color="#d62728", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.bar(x_pos + w / 2, other_rates_pct, w, label="Other / no email",
           color="#1f77b4", edgecolor="black", linewidth=0.5, alpha=0.85)
    for i, (kr, otr) in enumerate(zip(known_rates_pct, other_rates_pct)):
        ax.text(x_pos[i] - w / 2, kr + 0.8, f"{kr:.0f}%", ha="center",
                va="bottom", fontproperties=FONT_ANNOT, color="#d62728")
        ax.text(x_pos[i] + w / 2, otr + 0.8, f"{otr:.0f}%", ha="center",
                va="bottom", fontproperties=FONT_ANNOT, color="#1f77b4")
    # Add point-biserial r below venue label
    for i, v in enumerate(plot_venues):
        c = venue_stats[v].get("point_biserial_corr")
        p = venue_stats[v].get("point_biserial_pval")
        if c is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(x_pos[i], -6, f"r={c:+.2f}{sig}", ha="center",
                    va="top", fontproperties=FONT_ANNOT, color="#555555")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Acceptance rate (%)", fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Acceptance by Affiliation Status", fontproperties=FONT_TITLE_SM)
    ax.legend(prop=FONT_LEGEND)
    ax.set_ylim(-8, max(max(known_rates_pct), max(other_rates_pct)) * 1.25 + 5)
    _set_tick_fonts(ax)

    # --- Panel 2: Top 25 institutions, horizontal bars colored by brand ---
    ax2 = axes[1]
    top_n = 25
    top_insts = inst_counter.most_common(top_n)
    if top_insts:
        inst_names = [t[0] for t in top_insts]
        inst_counts = [t[1] for t in top_insts]
        inst_acc_rates = [float(np.mean(inst_accepted[n])) for n in inst_names]
        inst_colors = [inst_color_map.get(n, _OTHER_COLOR) for n in inst_names]

        y_pos = np.arange(len(inst_names))
        hbars = ax2.barh(y_pos, inst_counts, color=inst_colors,
                         edgecolor="black", linewidth=0.5, alpha=0.88)
        for i, (name, cnt, rate) in enumerate(zip(inst_names, inst_counts, inst_acc_rates)):
            ax2.text(cnt + max(inst_counts) * 0.01, i,
                     f" {rate*100:.0f}% acc  (n={cnt})",
                     va="center", fontproperties=FONT_ANNOT)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(inst_names)
        ax2.invert_yaxis()
        ax2.set_xlabel("# papers affiliated", fontproperties=FONT_AXIS_BOLD)
        ax2.set_title("Top 25 Institutions by Volume", fontproperties=FONT_TITLE_SM)
        ax2.set_xlim(0, max(inst_counts) * 1.45)
        _set_tick_fonts(ax2)

    # --- Panel 3: Scatter – acceptance rate vs volume, per institution ---
    ax3 = axes[2]
    if top_insts:
        # Plot ALL institutions with >= 20 papers
        scatter_insts = [(n, c) for n, c in inst_counter.items() if c >= 20]
        scatter_insts.sort(key=lambda t: -t[1])
        s_names = [t[0] for t in scatter_insts]
        s_counts = [t[1] for t in scatter_insts]
        s_rates = [100 * float(np.mean(inst_accepted[n])) for n in s_names]
        s_colors = [inst_color_map.get(n, _OTHER_COLOR) for n in s_names]

        # Also compute baseline: overall acceptance rate for unmatched papers
        other_acc_pct = 100 * float(np.mean(overall_other_acc)) if overall_other_acc else 0

        ax3.scatter(s_counts, s_rates, c=s_colors, s=70,
                    edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3)
        ax3.axhline(y=other_acc_pct, color="#AAAAAA", linestyle="--", linewidth=1.2,
                    label=f"Unaffiliated baseline ({other_acc_pct:.0f}%)", zorder=1)

        # Label the top-15 by count, and any with extreme acceptance rates
        labeled = set()
        for name, cnt, rate in sorted(zip(s_names, s_counts, s_rates),
                                       key=lambda t: -t[1])[:15]:
            ax3.annotate(name, (cnt, rate), textcoords="offset points",
                         xytext=(5, 4), fontproperties=FONT_ANNOT,
                         color=inst_color_map.get(name, "#333333"))
            labeled.add(name)
        # Also label outliers not yet labeled
        for name, cnt, rate in zip(s_names, s_counts, s_rates):
            if name not in labeled and (rate > 65 or rate < 15) and cnt >= 30:
                ax3.annotate(name, (cnt, rate), textcoords="offset points",
                             xytext=(5, -6), fontproperties=FONT_ANNOT,
                             color=inst_color_map.get(name, "#333333"))

        ax3.set_xlabel("# papers affiliated", fontproperties=FONT_AXIS_BOLD)
        ax3.set_ylabel("Acceptance rate (%)", fontproperties=FONT_AXIS_BOLD)
        ax3.set_title("Institution Volume vs Acceptance", fontproperties=FONT_TITLE_SM)
        ax3.legend(prop=FONT_LEGEND, loc="upper right")
        _set_tick_fonts(ax3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "f1_email_confound.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# S2 – Inconsistent Acceptance Labels Across Venues
# ===================================================================

def diagnose_s2():
    """Compute acceptance rates per venue, flag venues where the label source
    is indirect (e.g. arXiv) or where 100% acceptance indicates a filtered corpus."""
    print("\n" + "=" * 70)
    print("S2: Inconsistent Acceptance Labels Across Venues")
    print("=" * 70)

    venue_stats = {}

    for venue in VENUES:
        total = 0
        accepted = 0
        rejected = 0
        missing_label = 0
        has_reviews = 0
        empty_reviews = 0
        label_source = "unknown"

        if venue.startswith("arxiv"):
            label_source = "inferred_from_publication"
        elif venue == "iclr_2017":
            label_source = "explicit_reviewer_decision"
        elif venue in ("acl_2017", "conll_2016"):
            label_source = "softconf_review_dump"
        elif venue.startswith("nips"):
            label_source = "published_proceedings_only"

        conf_counter = Counter()

        for fp, data in iter_review_files(venue):
            total += 1
            acc = data.get("accepted", None)
            reviews = data.get("reviews", [])

            if reviews:
                has_reviews += 1
            else:
                empty_reviews += 1

            if acc is True:
                accepted += 1
            elif acc is False:
                rejected += 1
            else:
                missing_label += 1

            if venue.startswith("arxiv"):
                conf = data.get("conference", "unknown")
                conf_counter[conf] += 1

        rate = accepted / total if total else 0
        venue_stats[venue] = {
            "total_papers": total,
            "accepted": accepted,
            "rejected": rejected,
            "missing_label": missing_label,
            "acceptance_rate": round(rate, 4),
            "acceptance_rate_pct": round(100 * rate, 2),
            "papers_with_reviews": has_reviews,
            "papers_without_reviews": empty_reviews,
            "label_source": label_source,
        }

        if conf_counter:
            top_confs = conf_counter.most_common(10)
            venue_stats[venue]["top_conferences"] = {
                c: n for c, n in top_confs
            }

        flag = ""
        if rate == 1.0:
            flag = " ⚠ 100% ACCEPTANCE (filtered corpus?)"
        elif missing_label > 0:
            flag = f" ⚠ {missing_label} papers MISSING acceptance label"
        if empty_reviews == total and total > 0:
            flag += " ⚠ NO REVIEWS"

        print(f"  {_nice_venue(venue):12s}  n={total:5d}  "
              f"accepted={accepted:5d}  rejected={rejected:5d}  "
              f"rate={100*rate:5.1f}%  "
              f"source={label_source}{flag}")

    result = {"diagnostic": "S2_acceptance_labels", "per_venue": venue_stats}

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    fig.suptitle("S2: Inconsistent Acceptance Labels Across Venues",
                 fontproperties=FONT_SUPTITLE, y=0.99)

    plot_venues = [v for v in VENUES if venue_stats[v]["total_papers"] > 0]
    labels = [_nice_venue(v) for v in plot_venues]

    # Panel 1: Acceptance rate bar chart with label source color coding
    ax = axes[0]
    rates = [venue_stats[v]["acceptance_rate_pct"] for v in plot_venues]
    source_colors = {
        "explicit_reviewer_decision": "#2ca02c",
        "softconf_review_dump": "#1f77b4",
        "inferred_from_publication": "#d62728",
        "published_proceedings_only": "#ff7f0e",
        "unknown": "#7f7f7f",
    }
    bar_colors = [source_colors.get(venue_stats[v]["label_source"], "#7f7f7f")
                  for v in plot_venues]
    bars = ax.bar(labels, rates, color=bar_colors, edgecolor="black", linewidth=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom",
                fontproperties=FONT_ANNOT_LG)
    ax.set_ylabel("Acceptance rate (%)", fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Acceptance Rates by Venue", fontproperties=FONT_TITLE_SM)
    ax.set_ylim(0, max(rates) * 1.2 + 5)
    ax.tick_params(axis="x", rotation=25)
    _set_tick_fonts(ax)
    # Manual legend for label sources
    from matplotlib.patches import Patch
    legend_items = []
    seen = set()
    for v in plot_venues:
        src = venue_stats[v]["label_source"]
        if src not in seen:
            seen.add(src)
            nice_src = src.replace("_", " ").title()
            legend_items.append(Patch(facecolor=source_colors.get(src, "#7f7f7f"),
                                      edgecolor="black", linewidth=0.5,
                                      label=nice_src))
    ax.legend(handles=legend_items, prop=FONT_LEGEND, loc="upper right",
              title="Label Source", title_fontproperties=FONT_LEGEND)

    # Panel 2: Stacked bar – accepted vs rejected vs missing
    ax2 = axes[1]
    x_pos = np.arange(len(plot_venues))
    w = 0.5
    acc_counts = [venue_stats[v]["accepted"] for v in plot_venues]
    rej_counts = [venue_stats[v]["rejected"] for v in plot_venues]
    mis_counts = [venue_stats[v]["missing_label"] for v in plot_venues]

    ax2.bar(x_pos, acc_counts, w, label="Accepted", color="#2ca02c",
            edgecolor="black", linewidth=0.4)
    ax2.bar(x_pos, rej_counts, w, bottom=acc_counts, label="Rejected",
            color="#d62728", edgecolor="black", linewidth=0.4)
    bottom2 = [a + r for a, r in zip(acc_counts, rej_counts)]
    ax2.bar(x_pos, mis_counts, w, bottom=bottom2, label="No label",
            color="#7f7f7f", edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Number of papers", fontproperties=FONT_AXIS_BOLD)
    ax2.set_title("Label Distribution by Venue", fontproperties=FONT_TITLE_SM)
    ax2.legend(prop=FONT_LEGEND)
    ax2.tick_params(axis="x", rotation=25)
    _set_tick_fonts(ax2)

    # Panel 3: Diagnostic summary - reviews availability vs label source
    ax3 = axes[2]
    with_rev = [venue_stats[v]["papers_with_reviews"] for v in plot_venues]
    without_rev = [venue_stats[v]["papers_without_reviews"] for v in plot_venues]
    with_rev_pct = [100 * wr / (wr + wor) if (wr + wor) > 0 else 0
                    for wr, wor in zip(with_rev, without_rev)]
    without_rev_pct = [100 - p for p in with_rev_pct]

    bars_wr = ax3.barh(x_pos, with_rev_pct, 0.4, label="Has reviews",
                       color="#1f77b4", edgecolor="black", linewidth=0.4)
    bars_nor = ax3.barh(x_pos, without_rev_pct, 0.4, left=with_rev_pct,
                        label="No reviews", color="#ff7f0e",
                        edgecolor="black", linewidth=0.4)
    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel("% of papers", fontproperties=FONT_AXIS_BOLD)
    ax3.set_title("Review Availability", fontproperties=FONT_TITLE_SM)
    ax3.legend(prop=FONT_LEGEND, loc="lower right")
    ax3.set_xlim(0, 105)
    # Annotate percentages
    for i, (wr_pct, nor_pct) in enumerate(zip(with_rev_pct, without_rev_pct)):
        if wr_pct > 5:
            ax3.text(wr_pct / 2, i, f"{wr_pct:.0f}%", ha="center", va="center",
                     fontproperties=FONT_ANNOT, color="white")
        if nor_pct > 5:
            ax3.text(wr_pct + nor_pct / 2, i, f"{nor_pct:.0f}%",
                     ha="center", va="center",
                     fontproperties=FONT_ANNOT, color="white")
    _set_tick_fonts(ax3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "s2_acceptance_labels.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# S1 – Inconsistent Review Schemas Across Venues
# ===================================================================

def diagnose_s1():
    """Build a field-presence heatmap for review-level and paper-level fields
    across all venues, quantifying schema inconsistency."""
    print("\n" + "=" * 70)
    print("S1: Inconsistent Review Schemas Across Venues")
    print("=" * 70)

    # --- Collect review-level fields ---
    review_field_counts = {}   # venue -> Counter(field -> count)
    review_totals = {}         # venue -> total review entries
    paper_field_counts = {}    # venue -> Counter(field -> count)
    paper_totals = {}          # venue -> total papers

    for venue in VENUES:
        r_fields = Counter()
        p_fields = Counter()
        n_reviews = 0
        n_papers = 0

        for fp, data in iter_review_files(venue):
            n_papers += 1
            for k in data.keys():
                p_fields[k] += 1
            for rev in data.get("reviews", []):
                n_reviews += 1
                for k in rev.keys():
                    r_fields[k] += 1

        review_field_counts[venue] = r_fields
        review_totals[venue] = n_reviews
        paper_field_counts[venue] = p_fields
        paper_totals[venue] = n_papers

        print(f"  {_nice_venue(venue):12s}  papers={n_papers:5d}  "
              f"reviews={n_reviews:5d}  "
              f"review_fields={len(r_fields):2d}  "
              f"paper_fields={len(p_fields):2d}")

    # --- Build union of all field names ---
    all_review_fields = sorted(set().union(*(rc.keys() for rc in review_field_counts.values())))
    all_paper_fields = sorted(set().union(*(pc.keys() for pc in paper_field_counts.values())))

    # --- Build presence-percentage matrices ---
    venues_with_reviews = [v for v in VENUES if review_totals[v] > 0]

    review_matrix = []  # rows = fields, cols = venues_with_reviews
    for field in all_review_fields:
        row = []
        for venue in venues_with_reviews:
            total = review_totals[venue]
            count = review_field_counts[venue].get(field, 0)
            row.append(100.0 * count / total if total > 0 else 0.0)
        review_matrix.append(row)
    review_matrix = np.array(review_matrix) if review_matrix else np.zeros((0, 0))

    paper_matrix = []  # rows = fields, cols = all VENUES
    for field in all_paper_fields:
        row = []
        for venue in VENUES:
            total = paper_totals[venue]
            count = paper_field_counts[venue].get(field, 0)
            row.append(100.0 * count / total if total > 0 else 0.0)
        paper_matrix.append(row)
    paper_matrix = np.array(paper_matrix) if paper_matrix else np.zeros((0, 0))

    # --- Build JSON result ---
    review_field_stats = {}
    for field in all_review_fields:
        per_venue = {}
        for venue in VENUES:
            total = review_totals[venue]
            count = review_field_counts[venue].get(field, 0)
            per_venue[venue] = {
                "count": count,
                "total": total,
                "pct": round(100.0 * count / total, 2) if total > 0 else 0.0,
            }
        review_field_stats[field] = per_venue

    paper_field_stats = {}
    for field in all_paper_fields:
        per_venue = {}
        for venue in VENUES:
            total = paper_totals[venue]
            count = paper_field_counts[venue].get(field, 0)
            per_venue[venue] = {
                "count": count,
                "total": total,
                "pct": round(100.0 * count / total, 2) if total > 0 else 0.0,
            }
        paper_field_stats[field] = per_venue

    result = {
        "diagnostic": "S1_schema_inconsistency",
        "venues_with_reviews": [_nice_venue(v) for v in venues_with_reviews],
        "review_level_fields": review_field_stats,
        "paper_level_fields": paper_field_stats,
        "num_review_fields_total": len(all_review_fields),
        "num_paper_fields_total": len(all_paper_fields),
    }

    # --- Visualization: 2-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("S1: Review & Paper Schema Inconsistency Across Venues",
                 fontproperties=FONT_SUPTITLE, y=0.99)

    # --- Panel 1: Review-level field × venue heatmap ---
    ax = axes[0]
    if review_matrix.size > 0:
        im = ax.imshow(review_matrix, cmap="YlOrRd", aspect="auto",
                        vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(venues_with_reviews)))
        ax.set_xticklabels([_nice_venue(v) for v in venues_with_reviews],
                           rotation=35, ha="right")
        ax.set_yticks(np.arange(len(all_review_fields)))
        ax.set_yticklabels(all_review_fields)
        # Annotate cells with percentages
        for i in range(len(all_review_fields)):
            for j in range(len(venues_with_reviews)):
                val = review_matrix[i, j]
                text_color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.0f}%" if val > 0 else "—",
                        ha="center", va="center",
                        fontproperties=FONT_ANNOT, color=text_color)
        ax.set_title("Review-Level Fields", fontproperties=FONT_TITLE_SM)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Field presence (%)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No review-level data", ha="center", va="center",
                transform=ax.transAxes, fontproperties=FONT_TITLE)
    _set_tick_fonts(ax)

    # --- Panel 2: Paper-level field × venue heatmap ---
    ax2 = axes[1]
    if paper_matrix.size > 0:
        im2 = ax2.imshow(paper_matrix, cmap="YlGnBu", aspect="auto",
                          vmin=0, vmax=100)
        ax2.set_xticks(np.arange(len(VENUES)))
        ax2.set_xticklabels([_nice_venue(v) for v in VENUES],
                            rotation=35, ha="right")
        ax2.set_yticks(np.arange(len(all_paper_fields)))
        ax2.set_yticklabels(all_paper_fields)
        for i in range(len(all_paper_fields)):
            for j in range(len(VENUES)):
                val = paper_matrix[i, j]
                text_color = "white" if val > 60 else "black"
                ax2.text(j, i, f"{val:.0f}%" if val > 0 else "—",
                         ha="center", va="center",
                         fontproperties=FONT_ANNOT, color=text_color)
        ax2.set_title("Paper-Level Fields", fontproperties=FONT_TITLE_SM)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Field presence (%)", fontsize=9)
    _set_tick_fonts(ax2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "s1_schema_inconsistency.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# R4 – Review Type Classification (Non-Review Entries in reviews[])
# ===================================================================

def _classify_review_type(rev):
    """Classify a single review entry into a semantic type.

    Returns one of:
        'meta_review'         — IS_META_REVIEW or is_meta_review is True
        'committee_decision'  — OTHER_KEYS contains 'pcs'
        'peer_review'         — AnonReviewer with RECOMMENDATION, or structured
                                review from ACL/CoNLL (has RECOMMENDATION + comments)
        'reviewer_no_rec'     — AnonReviewer without RECOMMENDATION, has comments
        'empty_reviewer'      — AnonReviewer, empty comments, no rec
        'anonymous_question'  — OTHER_KEYS is '(anonymous)'
        'author_response'     — OTHER_KEYS is a named person (not AnonReviewer/pcs/anonymous)
        'other'               — anything else
    """
    # Handle both ICLR uppercase and ACL/CoNLL lowercase meta-review flags
    is_meta = rev.get("IS_META_REVIEW", False) or rev.get("is_meta_review", False)
    # is_meta can be None in ACL/CoNLL — treat None as False
    if is_meta is None:
        is_meta = False
    other_keys = str(rev.get("OTHER_KEYS", ""))
    has_rec = "RECOMMENDATION" in rev
    comments = (rev.get("comments", "") or "").strip()
    ok_lower = other_keys.lower()

    if is_meta is True:
        return "meta_review"
    if "pcs" in ok_lower:
        return "committee_decision"
    if "anonreviewer" in ok_lower:
        if has_rec:
            return "peer_review"
        if not comments:
            return "empty_reviewer"
        return "reviewer_no_rec"
    if "(anonymous)" in other_keys:
        return "anonymous_question"
    # ACL/CoNLL structured reviews: no OTHER_KEYS but have RECOMMENDATION,
    # REVIEWER_CONFIDENCE, and comments — these are real peer reviews
    if not other_keys and has_rec and comments:
        return "peer_review"
    # Named person or bare conference string — treat as author/public comment
    if other_keys and other_keys not in ("", " "):
        return "author_response"
    return "other"


# Canonical display order and colors for review types
_REVIEW_TYPE_META = [
    ("peer_review",        "Peer Review",        "#2ca02c"),
    ("meta_review",        "Meta-Review",        "#1f77b4"),
    ("committee_decision", "Committee Decision",  "#ff7f0e"),
    ("reviewer_no_rec",    "Reviewer (no rec.)",  "#9467bd"),
    ("empty_reviewer",     "Empty Reviewer",      "#d62728"),
    ("anonymous_question", "Anonymous Question",   "#8c564b"),
    ("author_response",    "Author / Public",     "#e377c2"),
    ("other",              "Other",               "#7f7f7f"),
]
_RT_ORDER = [t[0] for t in _REVIEW_TYPE_META]
_RT_LABELS = {t[0]: t[1] for t in _REVIEW_TYPE_META}
_RT_COLORS = {t[0]: t[2] for t in _REVIEW_TYPE_META}


def diagnose_r4():
    """Classify each entry in venues' reviews arrays by semantic type and
    report per-venue breakdowns, highlighting non-review noise."""
    print("\n" + "=" * 70)
    print("R4: Review Type Classification (Non-Review Entries)")
    print("=" * 70)

    venue_type_counts = {}  # venue -> Counter(type -> count)
    venue_paper_counts = {}
    per_paper_details = defaultdict(list)  # venue -> list of per-paper dicts

    for venue in VENUES:
        type_counts = Counter()
        n_papers = 0
        for fp, data in iter_review_files(venue):
            n_papers += 1
            paper_types = Counter()
            for rev in data.get("reviews", []):
                rtype = _classify_review_type(rev)
                type_counts[rtype] += 1
                paper_types[rtype] += 1
            per_paper_details[venue].append(dict(paper_types))

        venue_type_counts[venue] = type_counts
        venue_paper_counts[venue] = n_papers
        total_entries = sum(type_counts.values())
        print(f"  {_nice_venue(venue):12s}  papers={n_papers:5d}  entries={total_entries:5d}")
        for rt in _RT_ORDER:
            cnt = type_counts.get(rt, 0)
            if cnt > 0:
                print(f"    {_RT_LABELS[rt]:22s}  {cnt:5d}  "
                      f"({100*cnt/total_entries:.1f}%)" if total_entries else "")

    # --- Build JSON result ---
    result_per_venue = {}
    for venue in VENUES:
        tc = venue_type_counts[venue]
        total = sum(tc.values())
        breakdown = {}
        for rt in _RT_ORDER:
            cnt = tc.get(rt, 0)
            breakdown[rt] = {
                "count": cnt,
                "pct": round(100.0 * cnt / total, 2) if total > 0 else 0.0,
            }
        result_per_venue[venue] = {
            "total_papers": venue_paper_counts[venue],
            "total_review_entries": total,
            "type_breakdown": breakdown,
        }

    result = {"diagnostic": "R4_review_type_classification", "per_venue": result_per_venue}

    # --- Visualization: 2-panel figure ---
    venues_with_entries = [v for v in VENUES
                           if sum(venue_type_counts[v].values()) > 0]
    if not venues_with_entries:
        print("  (no venues with review entries to plot)")
        return result

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("R4: Composition of reviews[] Arrays by Entry Type",
                 fontproperties=FONT_SUPTITLE, y=0.99)

    # --- Panel 1: Stacked bar chart — absolute counts ---
    ax = axes[0]
    x_pos = np.arange(len(venues_with_entries))
    labels = [_nice_venue(v) for v in venues_with_entries]
    bottoms = np.zeros(len(venues_with_entries))
    bar_handles = []

    for rt in _RT_ORDER:
        counts = np.array([venue_type_counts[v].get(rt, 0) for v in venues_with_entries],
                          dtype=float)
        if np.sum(counts) == 0:
            continue
        bars = ax.bar(x_pos, counts, 0.55, bottom=bottoms,
                      color=_RT_COLORS[rt], edgecolor="black", linewidth=0.4,
                      label=_RT_LABELS[rt])
        bar_handles.append(bars)
        # Annotate non-trivial segments
        for i, (cnt, bot) in enumerate(zip(counts, bottoms)):
            total = sum(venue_type_counts[venues_with_entries[i]].values())
            if cnt > 0 and total > 0 and cnt / total > 0.05:
                ax.text(x_pos[i], bot + cnt / 2, f"{cnt:.0f}",
                        ha="center", va="center",
                        fontproperties=FONT_ANNOT, color="white")
        bottoms += counts

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of entries in reviews[]", fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Entry Type Breakdown by Venue", fontproperties=FONT_TITLE_SM)
    ax.legend(prop=FONT_LEGEND, loc="upper right", fontsize=7)
    _set_tick_fonts(ax)

    # --- Panel 2: Percentage stacked bar (normalized to 100%) ---
    ax2 = axes[1]
    bottoms2 = np.zeros(len(venues_with_entries))
    for rt in _RT_ORDER:
        pcts = []
        for v in venues_with_entries:
            total = sum(venue_type_counts[v].values())
            cnt = venue_type_counts[v].get(rt, 0)
            pcts.append(100.0 * cnt / total if total > 0 else 0.0)
        pcts = np.array(pcts)
        if np.sum(pcts) == 0:
            continue
        ax2.bar(x_pos, pcts, 0.55, bottom=bottoms2,
                color=_RT_COLORS[rt], edgecolor="black", linewidth=0.4,
                label=_RT_LABELS[rt])
        for i, (pct, bot) in enumerate(zip(pcts, bottoms2)):
            if pct > 4:
                ax2.text(x_pos[i], bot + pct / 2, f"{pct:.0f}%",
                         ha="center", va="center",
                         fontproperties=FONT_ANNOT, color="white")
        bottoms2 += pcts

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Percentage of entries", fontproperties=FONT_AXIS_BOLD)
    ax2.set_ylim(0, 105)
    ax2.set_title("Entry Type Proportions (Normalized)", fontproperties=FONT_TITLE_SM)
    ax2.legend(prop=FONT_LEGEND, loc="upper right", fontsize=7)
    _set_tick_fonts(ax2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "r4_review_types.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# F2 – Hardcoded Submission Year in Feature Extraction
# ===================================================================

def diagnose_f2():
    """Compare actual paper submission years against the hardcoded year 2017
    used in featurize.py's get_num_recent_references().

    For arXiv papers (which have DATE_OF_SUBMISSION), compute the year offset
    and show how many papers are mischaracterized by the hardcoded year.
    """
    print("\n" + "=" * 70)
    print("F2: Hardcoded Year (2017) vs Actual Submission Year")
    print("=" * 70)

    HARDCODED_YEAR = 2017
    arxiv_venues = [v for v in VENUES if v.startswith("arxiv")]

    all_years = []
    all_offsets = []
    all_venue_labels = []
    venue_stats = {}

    for venue in arxiv_venues:
        years = []
        offsets = []
        n_total = 0
        n_with_date = 0
        n_missing_date = 0

        for fp, data in iter_review_files(venue):
            n_total += 1
            dos = data.get("DATE_OF_SUBMISSION", "")
            if not dos:
                n_missing_date += 1
                continue
            m = re.search(r"(\d{4})", str(dos))
            if not m:
                n_missing_date += 1
                continue
            year = int(m.group(1))
            n_with_date += 1
            years.append(year)
            offset = HARDCODED_YEAR - year
            offsets.append(offset)
            all_years.append(year)
            all_offsets.append(offset)
            all_venue_labels.append(venue)

        year_counter = Counter(years)
        offset_counter = Counter(offsets)
        n_exact = sum(1 for o in offsets if o == 0)
        n_off_by_1 = sum(1 for o in offsets if abs(o) == 1)
        n_off_by_gt3 = sum(1 for o in offsets if abs(o) > 3)

        venue_stats[venue] = {
            "total_papers": n_total,
            "papers_with_date": n_with_date,
            "papers_missing_date": n_missing_date,
            "year_distribution": {str(y): c for y, c in sorted(year_counter.items())},
            "offset_distribution": {str(o): c for o, c in sorted(offset_counter.items())},
            "mean_offset": round(float(np.mean(offsets)), 2) if offsets else None,
            "median_offset": int(np.median(offsets)) if offsets else None,
            "papers_exact_match": n_exact,
            "papers_off_by_1": n_off_by_1,
            "papers_off_by_gt3": n_off_by_gt3,
            "pct_off_by_gt3": round(100.0 * n_off_by_gt3 / n_with_date, 2) if n_with_date else 0.0,
        }
        print(f"  {_nice_venue(venue):12s}  n={n_total:5d}  "
              f"with_date={n_with_date:5d}  "
              f"mean_offset={np.mean(offsets):+.1f}  "
              f"off_by_>3yr={n_off_by_gt3:4d} ({venue_stats[venue]['pct_off_by_gt3']:.1f}%)")

    result = {
        "diagnostic": "F2_hardcoded_year",
        "hardcoded_year": HARDCODED_YEAR,
        "per_venue": venue_stats,
    }

    if not all_years:
        print("  (no papers with DATE_OF_SUBMISSION found)")
        return result

    # --- Visualization: 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("F2: Hardcoded Year (2017) vs Actual Submission Year",
                 fontproperties=FONT_SUPTITLE, y=0.98)

    # --- Panel 1: Histogram of actual submission years (all arXiv combined) ---
    ax = axes[0]
    year_arr = np.array(all_years)
    year_min, year_max = int(year_arr.min()), int(year_arr.max())
    bins_y = np.arange(year_min - 0.5, year_max + 1.5, 1)
    ax.hist(year_arr, bins=bins_y, color="#4c72b0", edgecolor="black",
            linewidth=0.5, alpha=0.85)
    ax.axvline(x=HARDCODED_YEAR, color="#d62728", linestyle="--", linewidth=2,
               label=f"Hardcoded year ({HARDCODED_YEAR})")
    ax.set_xlabel("Submission year", fontproperties=FONT_AXIS_BOLD)
    ax.set_ylabel("Number of papers", fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Actual Submission Years (arXiv)", fontproperties=FONT_TITLE_SM)
    ax.legend(prop=FONT_LEGEND)
    _set_tick_fonts(ax)

    # --- Panel 2: Histogram of year offsets (2017 - actual) ---
    ax2 = axes[1]
    offset_arr = np.array(all_offsets)
    off_min, off_max = int(offset_arr.min()), int(offset_arr.max())
    bins_o = np.arange(off_min - 0.5, off_max + 1.5, 1)
    n_vals, _, patches = ax2.hist(offset_arr, bins=bins_o, color="#4c72b0",
                                   edgecolor="black", linewidth=0.5, alpha=0.85)
    # Color the bars by severity
    for patch, left_edge in zip(patches, np.arange(off_min, off_max + 1)):
        if left_edge > 3:
            patch.set_facecolor("#d62728")
            patch.set_alpha(0.85)
        elif left_edge > 1:
            patch.set_facecolor("#ff7f0e")
            patch.set_alpha(0.85)
        elif left_edge >= 0:
            patch.set_facecolor("#2ca02c")
            patch.set_alpha(0.85)
        else:
            patch.set_facecolor("#7f7f7f")
            patch.set_alpha(0.7)
    ax2.axvline(x=0, color="#2ca02c", linestyle="-", linewidth=1.5,
                label="Exact match (offset=0)")
    ax2.axvline(x=3, color="#d62728", linestyle=":", linewidth=1.5,
                label="Severe (offset>3)")
    n_severe = sum(1 for o in all_offsets if o > 3)
    pct_severe = 100.0 * n_severe / len(all_offsets)
    ax2.text(0.97, 0.95,
             f"{n_severe:,} papers ({pct_severe:.1f}%)\noff by >3 years",
             transform=ax2.transAxes, ha="right", va="top",
             fontproperties=FONT_ANNOT_LG, color="#d62728",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#d62728", alpha=0.9))
    ax2.set_xlabel(f"Year offset ({HARDCODED_YEAR} − actual year)",
                   fontproperties=FONT_AXIS_BOLD)
    ax2.set_ylabel("Number of papers", fontproperties=FONT_AXIS_BOLD)
    ax2.set_title("Year Offset Distribution", fontproperties=FONT_TITLE_SM)
    ax2.legend(prop=FONT_LEGEND, loc="upper left")
    _set_tick_fonts(ax2)

    # --- Panel 3: Per-venue box plot of offsets ---
    ax3 = axes[2]
    venue_groups = defaultdict(list)
    for off, v in zip(all_offsets, all_venue_labels):
        venue_groups[v].append(off)
    plot_venues = [v for v in arxiv_venues if v in venue_groups]
    box_data = [venue_groups[v] for v in plot_venues]
    box_labels = [_nice_venue(v) for v in plot_venues]
    colors_box = ["#d62728", "#9467bd", "#8c564b"]
    bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     widths=0.5, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.axhline(y=0, color="#2ca02c", linestyle=":", alpha=0.6,
                label="Exact match")
    ax3.axhline(y=3, color="#d62728", linestyle=":", alpha=0.6,
                label="Severe threshold")
    ax3.set_ylabel(f"Year offset ({HARDCODED_YEAR} − actual)",
                   fontproperties=FONT_AXIS_BOLD)
    ax3.set_title("Offset by arXiv Subset", fontproperties=FONT_TITLE_SM)
    ax3.legend(prop=FONT_LEGEND, loc="upper right")
    _set_tick_fonts(ax3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "f2_hardcoded_year.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# R2 – Empty Review Comments
# ===================================================================

def diagnose_r2():
    """Count reviews with empty or whitespace-only comments per venue."""
    print("\n" + "=" * 70)
    print("R2: Empty Review Comments")
    print("=" * 70)

    venue_stats = {}

    for venue in VENUES:
        n_papers = 0
        total_reviews = 0
        empty_comments = 0
        whitespace_only = 0  # non-empty but whitespace-only
        papers_with_empty = 0
        comment_lengths = []

        for fp, data in iter_review_files(venue):
            n_papers += 1
            paper_has_empty = False
            for rev in data.get("reviews", []):
                total_reviews += 1
                raw = rev.get("comments", "")
                if raw is None:
                    raw = ""
                stripped = raw.strip()
                comment_lengths.append(len(stripped))
                if not raw:
                    empty_comments += 1
                    paper_has_empty = True
                elif not stripped:
                    whitespace_only += 1
                    paper_has_empty = True
            if paper_has_empty:
                papers_with_empty += 1

        total_empty = empty_comments + whitespace_only
        venue_stats[venue] = {
            "total_papers": n_papers,
            "total_reviews": total_reviews,
            "empty_comments": empty_comments,
            "whitespace_only": whitespace_only,
            "total_empty_or_ws": total_empty,
            "empty_pct": round(100.0 * total_empty / total_reviews, 2) if total_reviews else 0.0,
            "papers_with_any_empty": papers_with_empty,
            "papers_with_empty_pct": round(100.0 * papers_with_empty / n_papers, 2) if n_papers else 0.0,
            "mean_comment_length": round(float(np.mean(comment_lengths)), 1) if comment_lengths else 0.0,
            "median_comment_length": round(float(np.median(comment_lengths)), 1) if comment_lengths else 0.0,
        }
        if total_reviews > 0:
            print(f"  {_nice_venue(venue):12s}  reviews={total_reviews:5d}  "
                  f"empty={total_empty:5d} ({venue_stats[venue]['empty_pct']:5.1f}%)  "
                  f"papers_affected={papers_with_empty:4d}  "
                  f"mean_len={venue_stats[venue]['mean_comment_length']:.0f}")
        else:
            print(f"  {_nice_venue(venue):12s}  reviews=    0  (no reviews)")

    result = {"diagnostic": "R2_empty_review_comments", "per_venue": venue_stats}

    # --- Visualization: 2-panel figure ---
    venues_with_reviews = [v for v in VENUES if venue_stats[v]["total_reviews"] > 0]
    if not venues_with_reviews:
        print("  (no venues with reviews to plot)")
        return result

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("R2: Empty Review Comments",
                 fontproperties=FONT_SUPTITLE, y=0.98)

    labels = [_nice_venue(v) for v in venues_with_reviews]
    x_pos = np.arange(len(venues_with_reviews))

    # --- Panel 1: Bar chart of empty-comment percentage ---
    ax = axes[0]
    empty_pcts = [venue_stats[v]["empty_pct"] for v in venues_with_reviews]
    colors = ["#d62728" if p > 10 else "#ff7f0e" if p > 0 else "#2ca02c"
              for p in empty_pcts]
    bars = ax.bar(x_pos, empty_pcts, 0.55, color=colors, edgecolor="black",
                  linewidth=0.6)
    for bar, pct, v in zip(bars, empty_pcts, venues_with_reviews):
        cnt = venue_stats[v]["total_empty_or_ws"]
        total = venue_stats[v]["total_reviews"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n({cnt}/{total})",
                ha="center", va="bottom", fontproperties=FONT_ANNOT)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Empty comments (%)", fontproperties=FONT_AXIS_BOLD)
    ax.set_title("Empty/Whitespace Comments by Venue", fontproperties=FONT_TITLE_SM)
    ax.set_ylim(0, max(empty_pcts) * 1.35 + 2 if empty_pcts else 10)
    _set_tick_fonts(ax)

    # --- Panel 2: Stacked bar — non-empty vs empty comment counts ---
    ax2 = axes[1]
    non_empty = [venue_stats[v]["total_reviews"] - venue_stats[v]["total_empty_or_ws"]
                 for v in venues_with_reviews]
    empty_counts = [venue_stats[v]["total_empty_or_ws"] for v in venues_with_reviews]
    w = 0.55
    ax2.bar(x_pos, non_empty, w, label="Has content", color="#2ca02c",
            edgecolor="black", linewidth=0.4)
    ax2.bar(x_pos, empty_counts, w, bottom=non_empty, label="Empty / WS-only",
            color="#d62728", edgecolor="black", linewidth=0.4)
    for i, (ne, emp) in enumerate(zip(non_empty, empty_counts)):
        if emp > 0:
            ax2.text(x_pos[i], ne + emp + 5, f"+{emp}",
                     ha="center", va="bottom",
                     fontproperties=FONT_ANNOT, color="#d62728")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Number of review entries", fontproperties=FONT_AXIS_BOLD)
    ax2.set_title("Comment Content vs Empty", fontproperties=FONT_TITLE_SM)
    ax2.legend(prop=FONT_LEGEND)
    _set_tick_fonts(ax2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outpath = os.path.join(OUTPUT_DIR, "r2_empty_comments.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {outpath}")

    return result


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("PeerRead Data Quality Diagnostics")
    print("=" * 70)
    print(f"Data root: {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")

    all_results = {}

    # Run the original four diagnostics
    all_results["R1"] = diagnose_r1()
    all_results["P1"] = diagnose_p1()
    all_results["F1"] = diagnose_f1()
    all_results["S2"] = diagnose_s2()

    # Run the new four diagnostics
    all_results["S1"] = diagnose_s1()
    all_results["R4"] = diagnose_r4()
    all_results["F2"] = diagnose_f2()
    all_results["R2"] = diagnose_r2()

    # Write combined JSON output
    json_path = os.path.join(OUTPUT_DIR, "diagnostic_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"All results written to {json_path}")
    print(f"Visualizations saved to {OUTPUT_DIR}/")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    results = main()