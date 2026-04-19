#!/bin/bash
# Leave-one-subject-out cross-validation for SimpleCNN DMT plasma regression.
# Every available subject is held out once as the sole validation subject.
# Early stopping uses val R² (higher is better). Metrics: MAE, RMSE, MSE, R².
# Usage: bash src/models/reg_simpleCNN/run_cv.sh
#   (or: bash run_cv.sh from inside this directory)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.train_cv"

echo "============================================="
echo "  SimpleCNN LOSO Cross-Validation"
echo "  Early stop on val R² (maximize)"
echo "============================================="

$PYTHON -m $MODULE \
    --lr 1e-3 \
    --batch_size 64 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --epochs 300 \
    --patience 20 \
    --seed 42 \
    --run_name "cv_baseline"

echo ""
echo "============================================="
echo "  CV run completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "============================================="
