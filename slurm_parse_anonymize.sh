#!/bin/bash
#SBATCH --job-name=peerread_parse_anon
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/droytbur/10718peerread/logs/slurm_%j.log
#SBATCH --error=/data/user_data/droytbur/10718peerread/logs/slurm_%j.err

set -euo pipefail

REPO=/data/user_data/droytbur/10718peerread
source "$REPO/.venv/bin/activate"
TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
    echo "Usage: sbatch slurm_parse_anonymize.sh <year-or-dir>"
    echo "Examples: 2024 | neurips_2024 | neurips_2025_full"
    exit 1
fi

if [[ "$TARGET" != neurips_* ]]; then
    TARGET="neurips_${TARGET}"
fi

INPUT_DIR="$REPO/output/$TARGET"
PARSED_DIR="$INPUT_DIR/parsed_pdfs"
ANON_DIR="$INPUT_DIR/anonymized_pdfs"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -d "$INPUT_DIR/pdfs" ]]; then
    echo "ERROR: Missing PDF directory: $INPUT_DIR/pdfs"
    exit 1
fi

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Target: $TARGET"
echo "Mode:   CPU-only"
echo "Workers:$WORKERS"
echo "Start:  $(date)"
echo "============================================================"

echo ""
echo "=== PHASE 1: Parsing PDFs ($TARGET) ==="
CUDA_VISIBLE_DEVICES="" python3 "$REPO/parse_pdfs_docling.py" \
    --input-dir "$INPUT_DIR" \
    --skip-existing

echo ""
echo "=== PHASE 1 complete at $(date) ==="

echo ""
echo "=== PHASE 2: Anonymizing ($TARGET) ==="
python3 "$REPO/anonymize_batch.py" \
    --input-dir  "$PARSED_DIR" \
    --output-dir "$ANON_DIR" \
    --workers "$WORKERS" \
    --skip-existing

echo ""
echo "=== PHASE 2 complete at $(date) ==="

echo ""
echo "============================================================"
echo "All done at $(date)"
PARSED=$(find "$PARSED_DIR" -maxdepth 1 -name '*.json' | wc -l)
ANON=$(find "$ANON_DIR" -maxdepth 1 -name '*.json' | wc -l)
echo "  $TARGET: $PARSED parsed, $ANON anonymized"
echo "============================================================"
