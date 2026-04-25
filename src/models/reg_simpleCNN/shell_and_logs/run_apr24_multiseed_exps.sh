#!/bin/bash
# apr24 depth × SE ablation — multi-seed LOSO CV.
# Best kernel config from apr23 (k1=63/k2=15/k3=15, RF=623ms) held fixed.
# 6 experiments: 3 depth levels (1/2/3 blocks) × 2 SE conditions.
# Run from project root: bash src/models/reg_simpleCNN/run_apr24_multiseed_exps.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.multiseed_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3
        --weight_decay 1e-4 --loss smoothl1 --huber_beta 10.0
        --k1 63 --k2 15 --k3 15
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "========================================================"
echo "  apr24 depth x SE ablation (6 multi-seed experiments)"
echo "========================================================"

$PYTHON -m $MODULE $COMMON \
    --n_blocks 3 \
    --run_name "apr24_exp1" \
    --description "Full model (3 conv blocks) with SE. apr23_exp4 best-kernel config (k1=63/k2=15/k3=15, RF=623ms). Multi-seed baseline to get honest error bars on the kernel ablation winner."

echo ">>> apr24_exp1 done"

$PYTHON -m $MODULE $COMMON \
    --n_blocks 3 --no_se \
    --run_name "apr24_exp2" \
    --description "Full model (3 conv blocks) without SE. Same architecture as apr24_exp1 but SE replaced with identity. Quantifies SE contribution in the best-kernel setting across seeds."

echo ">>> apr24_exp2 done"

$PYTHON -m $MODULE $COMMON \
    --n_blocks 2 \
    --run_name "apr24_exp3" \
    --description "Two conv blocks with SE (k1=63/k2=15, RF=175ms). Ablates depth by removing block3. Tests whether the third compression stage contributes signal or overfits on this small N."

echo ">>> apr24_exp3 done"

$PYTHON -m $MODULE $COMMON \
    --n_blocks 2 --no_se \
    --run_name "apr24_exp4" \
    --description "Two conv blocks without SE (k1=63/k2=15, RF=175ms). Jointly ablates depth and SE. Isolates the combined contribution of block3 + channel attention."

echo ">>> apr24_exp4 done"

$PYTHON -m $MODULE $COMMON \
    --n_blocks 1 \
    --run_name "apr24_exp5" \
    --description "One conv block with SE (k1=63, RF=63ms). Minimal depth. Tests whether a single wide conv with channel re-weighting is sufficient to capture the DMT signal."

echo ">>> apr24_exp5 done"

$PYTHON -m $MODULE $COMMON \
    --n_blocks 1 --no_se \
    --run_name "apr24_exp6" \
    --description "One conv block without SE (k1=63, RF=63ms). Minimal model, no attention. Lower bound for the depth+SE ablation — pure single-layer feature extractor."

echo ">>> apr24_exp6 done"

echo ""
echo "========================================================"
echo "  All 6 apr24 experiments completed."
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "========================================================"
