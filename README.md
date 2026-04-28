## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

AWS Bedrock credentials must be configured (`aws configure`) for all LLM inference scripts.

---

## Reproducing the paper

### 0. Data

The processed dataset (reviews + anonymized paper text) is on HuggingFace and can be skipped to directly:

```bash
# Download the unified dataset
from huggingface_hub import snapshot_download
snapshot_download("djroytburg/NeurIPS-2023-2025", repo_type="dataset", local_dir="hf_data")
```

To rebuild from scratch from OpenReview:

```bash
# 1. Scrape NeurIPS reviews and metadata
python scrape_neurips.py --year 2023 --output output/neurips_2023
python scrape_neurips.py --year 2024 --output output/neurips_2024
python scrape_neurips.py --year 2025 --output output/neurips_2025_full

# 2. Download and parse PDFs
python parse_pdfs_parallel.py --reviews output/neurips_2025_full/reviews \
                               --output  output/neurips_2025_full/parsed_pdfs

# 3. Anonymize parsed PDFs
python anonymize_pdfs.py --input-dir  output/neurips_2025_full/parsed_pdfs \
                          --output-dir output/neurips_2025_full/anonymized_pdfs
# Post-process any residual footnote leaks
python anonymize_current_footnotes.py --input-dir output/neurips_2025_full/anonymized_pdfs

# 4. Validate anonymization
python validate_anonymization.py --input-dir output/neurips_2025_full/anonymized_pdfs

# 5. Build canonical balanced eval set (n=46, seed=10718)
python create_balanced_split.py
```

For ICLR data:

```bash
python scrape_iclr_hf.py          # pulls from ReviewHub/ICLR on HuggingFace
python integrate_reviewhub_iclr.py # normalizes to this repo's review JSON layout
```

---

### 1. Zero-shot and conservative prompting (Table 2, 3)

```bash
# Zero-shot neutral baseline — all three models
for model in llama deepseek gemma; do
  python run_zero_shot_eval.py --model $model
done

# Conservative prompting strategies
for model in llama deepseek gemma; do
  python run_zero_shot_eval.py --model $model --strategy conservative
  python run_zero_shot_eval.py --model $model --strategy severe_conservative
done

# Regenerate tables and figures from saved results
python run_zero_shot_eval.py --plot
```

---

### 2. Few-shot prompting (Table 4)

```bash
# Default 3R/2A configuration, all models
python run_few_shot_3r2a.py

# Source ablation (ICLR vs NeurIPS examples)
python run_few_shot_eval.py --mode balanced --full

# Class-balance ablation (sweep 0R–5R out of 5 examples)
python run_few_shot_ablation.py
python plot_ablation.py
```

---

### 3. Self-debate (Table 4)

```bash
# Self-debate — single model, all roles
for model in llama deepseek gemma; do
  python run_self_debate.py --model $model
done

# Multi-model jury (Magistral/Gemma/Llama reviewers, DeepSeek AC)
python run_debate_jury.py
python run_debate_jury_with_reviews.py
```

---

### 4. Baselines

```bash
# Reviewer-score threshold baseline (calibrated at T=4.25)
python run_score_baseline.py

# BERT / DistilBERT distribution-shift classifier
# Train on PeerRead, eval on PeerRead + NeurIPS
python bert_model.py train --peerread-data PeerRead/data \
                            --output-dir models/bert_peerread

python bert_model.py eval --model-dir models/bert_peerread \
                           --eval-reviews-dir output/neurips_2025_full/reviews \
                           --predictions-out results/bert_predictions_old_on_new.jsonl

# Train on NeurIPS, eval on PeerRead + NeurIPS
python convert_to_peerread_format.py   # converts NeurIPS data to PeerRead format
python bert_model.py train --peerread-data output/neurips_peerread_fmt \
                            --output-dir models/bert_neurips

# Generate BERT result table and figure
python plot_bert_results.py
```

---

### 5. Distribution shift analysis

```bash
python distribution_shift.py        # figures: results/figures/distribution/
python distribution_shift_table.py  # prints markdown tables
python analyze_iclr_neurips_shift.py
```

---

### 6. ICLR temporal shift

```bash
python run_zero_shot_eval_iclr.py --model llama
```

---

## Repository structure

```
.
├── scrape_neurips.py              # OpenReview scraper
├── scrape_iclr_hf.py              # ICLR ReviewHub downloader
├── integrate_reviewhub_iclr.py    # normalize ICLR to review JSON format
├── parse_pdfs_docling.py          # single-process PDF parser
├── parse_pdfs_parallel.py         # parallel PDF parser
├── anonymize_pdfs.py              # 4-layer anonymization pipeline
├── anonymize_batch.py             # batch wrapper for anonymize_pdfs.py
├── anonymize_current_footnotes.py # post-processing pass for footnote leaks
├── validate_anonymization.py      # leak checker
├── convert_json_to_markdown.py    # DocLing JSON → plain Markdown
├── convert_to_peerread_format.py  # NeurIPS → PeerRead format (for BERT)
├── create_balanced_split.py       # build canonical 46/226-paper eval sets
├── bert_model.py                  # DistilBERT acceptance classifier
├── run_zero_shot_eval.py          # zero-shot + conservative prompting
├── run_zero_shot_eval_iclr.py     # zero-shot on ICLR data
├── run_few_shot_eval.py           # few-shot prompting
├── run_few_shot_3r2a.py           # 3R/2A few-shot configuration
├── run_few_shot_ablation.py       # class-balance ablation
├── run_self_debate.py             # self-debate protocol
├── run_debate_jury.py             # multi-model jury
├── run_debate_jury_with_reviews.py
├── run_score_baseline.py          # reviewer-score threshold baseline
├── plot_ablation.py               # few-shot ablation figure
├── plot_bert_results.py           # BERT classifier figure + table
├── make_few_shot_source_table.py  # few-shot source comparison table
├── distribution_shift.py          # distribution shift figures
├── distribution_shift_table.py    # distribution shift tables
├── analyze_iclr_neurips_shift.py  # ICLR vs NeurIPS shift analysis
├── upload_to_hf.py                # build + upload HuggingFace dataset
├── utils/                         # shared utilities (metrics, judge rigs, cache)
├── paper/                         # LaTeX source
├── repetition_results/            # BERT multi-run outputs (for CIs)
├── results/                       # experiment outputs (gitignored except key files)
├── PeerRead/                      # PeerRead submodule / code
├── archive/                       # deprecated scripts kept for reference
├── Dockerfile                     # containerized environment
├── peerread.def                   # Singularity definition
└── requirements.txt
```

