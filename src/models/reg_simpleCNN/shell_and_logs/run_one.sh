#!/bin/bash
# Run 12 experiments with different hyperparameter configurations for DMT plasma regression.
# Every experiment uses a SINGLE validation subject.
# Usage: bash src/models/reg_simpleCNN/run_experiments.sh
#   (or: bash run_experiments.sh from inside this directory)

set -e

# Resolve project root (three levels up from this script) and run from there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.train"

echo "============================================="
echo "  SimpleCNN DMT Regression Suite (12 runs)"
echo "============================================="



$PYTHON -m $MODULE --lr 1e-2 --batch_size 16 --dropout 0.8 --weight_decay 1e-3 --patience 40 \
    --val_subjects S12 --run_name "exp21"



echo ""
echo "============================================="
echo "  All 12 experiments completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "============================================="
