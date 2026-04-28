#!/bin/bash
set -euo pipefail

REPO=/data/user_data/droytbur/10718peerread
SCRIPT="$REPO/slurm_parse_anonymize.sh"
YEARS=("neurips_2023" "neurips_2024" "neurips_2025_full")

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Missing SLURM script at $SCRIPT"
    exit 1
fi

echo "Submitting parse+anonymize jobs..."

for YEAR in "${YEARS[@]}"; do
    JOB_ID=""

    if JOB_ID=$(sbatch --parsable --partition=cpu "$SCRIPT" "$YEAR" 2>/dev/null); then
        echo "Submitted $YEAR to cpu partition: job $JOB_ID"
    elif JOB_ID=$(sbatch --parsable --partition=debug "$SCRIPT" "$YEAR" 2>/dev/null); then
        echo "Submitted $YEAR to debug partition (cpu unavailable): job $JOB_ID"
    else
        echo "ERROR: Failed to submit $YEAR (cpu and debug both failed)"
        exit 1
    fi
done

echo "Done."
