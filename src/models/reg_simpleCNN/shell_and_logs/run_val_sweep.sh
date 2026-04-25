#!/bin/bash
# Leave-one-out validation sweep using exp02 hyperparams.
# Each of the 8 available subjects is held out as the sole validation subject.
# Hyperparams: lr=1e-4, batch_size=64, dropout=0.3, weight_decay=1e-4, patience=20
# Usage: bash src/models/reg_simpleCNN/run_val_sweep.sh
#   (or: bash run_val_sweep.sh from inside this directory)

set -e

# Resolve project root (three levels up from this script) and run from there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.train"

echo "============================================="
echo "  SimpleCNN Leave-One-Out Val Sweep (8 splits)"
echo "  Hyperparams: lr=1e-4 bs=64 do=0.3 wd=1e-4"
echo "============================================="

# Available subjects: S01 S02 S05 S06 S07 S10 S12 S13

for SUBJ in S01 S02 S05 S06 S07 S10 S12 S13; do
    echo ""
    echo "--- Val subject: $SUBJ ---"
    $PYTHON -m $MODULE --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
        --val_subjects "$SUBJ" --run_name "val_sweep_${SUBJ}"
done

echo ""
echo "============================================="
echo "  All 8 leave-one-out splits completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "============================================="
