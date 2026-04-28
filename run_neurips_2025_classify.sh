#!/bin/bash
# Run accept classification on NeurIPS 2025 papers (adapted from anonymized version)

# Note: This script assumes the neurips_2025_full data has been split into train/dev/test subdirectories
# under output/neurips_2025_full/, each containing reviews/ and parsed_pdfs/ subdirs.
# If not, run create_balanced_split.py or similar to prepare the data.

DATADIR="output/neurips_2025_full"
DATASETS=("train" "dev" "test")
FEATDIR=dataset
MAX_VOCAB=False
ENCODER="w2v"
HAND=True

echo "======================================================"
echo "NeurIPS 2025 Dataset Classification Pipeline"
echo "======================================================"
echo ""
echo "Dataset: NeurIPS 2025 papers"
echo "  Assuming splits exist in $DATADIR/train, $DATADIR/dev, $DATADIR/test"
echo ""

# Extract features
for DATASET in "${DATASETS[@]}"
do
  echo "------------------------------------------------------"
  echo "Extracting features for $DATASET split..."
  echo "------------------------------------------------------"

  rm -rf $DATADIR/$DATASET/$FEATDIR

  python PeerRead/code/accept_classify/clean_featurize.py \
    $DATADIR/$DATASET/reviews/ \
    $DATADIR/$DATASET/parsed_pdfs/ \
    $DATADIR/$DATASET/$FEATDIR \
    $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat \
    $DATADIR/train/$FEATDIR/vect_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
    $MAX_VOCAB $ENCODER $HAND

  echo ""
done

# Run classifier
echo "======================================================"
echo "Classification"
echo "======================================================"
echo ""

python PeerRead/code/accept_classify/classify.py \
  $DATADIR/train/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/dev/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/test/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/train/$FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
  $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat

echo ""
echo "======================================================"
echo "Pipeline Complete!"
echo "======================================================"
