#!/bin/bash
#
# run_clean_featurize_classify.sh
# ================================
# Runs the data-cleaned featurize→classify pipeline (clean_featurize.py)
# and then re-runs the original pipeline (featurize.py) for comparison.
#
# Data cleaning applied (from DATA_PROBLEMS.md):
#   F1: Remove author/email metadata from paper content features
#   R1: Deduplicate reviews (100% of ICLR papers have duplicated reviews)
#   R2: Remove empty review comments (24.6% of ICLR entries)
#   R4: Filter non-review entries (only 35.8% are actual peer reviews)
#   F2: Use inferred submission year instead of hardcoded 2017
#
# Usage:
#   cd PeerRead/code/accept_classify
#   bash run_clean_featurize_classify.sh

DATADIR=../../data/iclr_2017
DATASETS=("train" "dev" "test")
MAX_VOCAB=30000
ENCODER=w2v
HAND=True

# Use a separate feature directory so clean and original don't clobber each other
CLEAN_FEATDIR=dataset_clean
ORIG_FEATDIR=dataset

echo "============================================================"
echo "  CLEANED PIPELINE (F1 + R1 + R2 + R4 + F2)"
echo "============================================================"
echo

start_time=$(date +%s)
for DATASET in "${DATASETS[@]}"
do
	echo "Extracting CLEAN features... DATASET=$DATASET ENCODER=$ENCODER VOCAB=$MAX_VOCAB HAND=$HAND"
	rm -rf "$DATADIR/$DATASET/$CLEAN_FEATDIR"
	python clean_featurize.py \
		"$DATADIR/$DATASET/reviews/" \
		"$DATADIR/$DATASET/parsed_pdfs/" \
		"$DATADIR/$DATASET/$CLEAN_FEATDIR" \
		"$DATADIR/train/$CLEAN_FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat" \
		"$DATADIR/train/$CLEAN_FEATDIR/vect_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl" \
		"$MAX_VOCAB" "$ENCODER" "$HAND"
	echo
done
clean_feat_time=$(( $(date +%s) - start_time ))

start_time=$(date +%s)
echo "Classifying (CLEAN)..."
python classify.py \
	"$DATADIR/train/$CLEAN_FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt" \
	"$DATADIR/dev/$CLEAN_FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt" \
	"$DATADIR/test/$CLEAN_FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt" \
	"$DATADIR/train/$CLEAN_FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl" \
	"$DATADIR/train/$CLEAN_FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat"
clean_cls_time=$(( $(date +%s) - start_time ))
echo "CLEAN featurize time: ${clean_feat_time}s, classify time: ${clean_cls_time}s"
echo

echo "============================================================"
echo "  ORIGINAL PIPELINE (baseline, no cleaning)"
echo "============================================================"
echo

start_time=$(date +%s)
for DATASET in "${DATASETS[@]}"
do
	echo "Extracting ORIGINAL features... DATASET=$DATASET ENCODER=$ENCODER VOCAB=$MAX_VOCAB HAND=$HAND"
	rm -rf "$DATADIR/$DATASET/$ORIG_FEATDIR"
	python featurize.py \
		"$DATADIR/$DATASET/reviews/" \
		"$DATADIR/$DATASET/parsed_pdfs/" \
		"$DATADIR/$DATASET/$ORIG_FEATDIR" \
		"$DATADIR/train/$ORIG_FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat" \
		"$DATADIR/train/$ORIG_FEATDIR/vect_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl" \
		"$MAX_VOCAB" "$ENCODER" "$HAND"
	echo
done
orig_feat_time=$(( $(date +%s) - start_time ))

start_time=$(date +%s)
echo "Classifying (ORIGINAL)..."
python classify.py \
	"$DATADIR/train/$ORIG_FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt" \
	"$DATADIR/dev/$ORIG_FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt" \
	"$DATADIR/test/$ORIG_FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt" \
	"$DATADIR/train/$ORIG_FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl" \
	"$DATADIR/train/$ORIG_FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat"
orig_cls_time=$(( $(date +%s) - start_time ))
echo "ORIGINAL featurize time: ${orig_feat_time}s, classify time: ${orig_cls_time}s"
echo

echo "============================================================"
echo "  COMPARISON SUMMARY"
echo "============================================================"
echo
echo "Encoder: $ENCODER | Vocab: $MAX_VOCAB | Hand features: $HAND"
echo
echo "See classifier output above for accuracy numbers."
echo "The CLEANED pipeline removes the F1 author/email confound"
echo "that gives the original pipeline an unfair advantage via"
echo "institutional affiliation signal (r = +0.18 to +0.29)."
echo
echo "If CLEAN accuracy drops, the original was partially relying"
echo "on author identity rather than paper content — confirming"
echo "the F1 confound documented in DATA_PROBLEMS.md."
echo
echo "If CLEAN accuracy holds or improves, the model was already"
echo "learning genuine content signals, and cleaning removed noise."
echo "============================================================"