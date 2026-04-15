#!/bin/bash
# Leave-one-out validation sweep using exp03 hyperparams.
# Each of the 8 available subjects is held out as the sole validation subject.
# Hyperparams: lr=1e-4, batch_size=16, dropout=0.1, weight_decay=1e-4, task_weight=1.0, patience=20, no coords
# Usage: bash run_val_sweep.sh

set -e
cd "$(dirname "$0")"

PYTHON=python3

echo "================================================"
echo "  graphTrip Leave-One-Out Val Sweep (8 splits)"
echo "  Hyperparams: lr=1e-4 bs=16 do=0.1 wd=1e-4"
echo "================================================"

# Available subjects: S01 S02 S05 S06 S07 S10 S12 S13

for SUBJ in S01 S02 S05 S06 S07 S10 S12 S13; do
    echo ""
    echo "--- Val subject: $SUBJ ---"
    $PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
        --task_weight 1.0 --patience 20 --val_subjects "$SUBJ" \
        --run_name "val_sweep_${SUBJ}"
done

echo ""
echo "================================================"
echo "  All 8 leave-one-out splits completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "================================================"
