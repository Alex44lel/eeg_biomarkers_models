#!/bin/bash
# apr23 kernel-size ablation: 5 experiments testing larger receptive fields.
# Baseline (run55, k1=15/k2=7/k3=7, RF=255ms) is already done — not repeated.
# Run from project root: bash src/models/reg_simpleCNN/run_apr23_kernel_exps.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.train_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3 --weight_decay 1e-4 --loss smoothl1 --huber_beta 10.0"

echo "========================================================"
echo "  apr23 kernel-size ablation (5 experiments)"
echo "========================================================"

$PYTHON -m $MODULE $COMMON \
    --k1 31 --k2 7 --k3 7 \
    --run_name "apr23_exp1" \
    --description "Moderate first-kernel bump. k1 15→31 doubles block-1 RF from 15ms to 31ms, targeting upper-alpha band (8-13Hz) directly in layer 1. All other params identical to run55 baseline. RF=271ms."

echo ">>> apr23_exp1 done"

$PYTHON -m $MODULE $COMMON \
    --k1 63 --k2 7 --k3 7 \
    --run_name "apr23_exp2" \
    --description "Large first kernel. k1=63 spans 63ms in block-1, covering a full theta/alpha cycle (~8Hz). Sharpest single-change hypothesis vs baseline. RF=303ms."

echo ">>> apr23_exp2 done"

$PYTHON -m $MODULE $COMMON \
    --k1 63 --k2 15 --k3 7 \
    --run_name "apr23_exp3" \
    --description "Wide entry + wider mid. k1=63 plus k2 7→15 extends mid-level context after first temporal compression. Tests whether compounding larger kernels in blocks 1+2 adds signal. RF=367ms."

echo ">>> apr23_exp3 done"

$PYTHON -m $MODULE $COMMON \
    --k1 63 --k2 15 --k3 15 \
    --run_name "apr23_exp4" \
    --description "Full large-kernel stack (theta/delta boundary). All three blocks widened to 63/15/15. RF=623ms reaches the theta-delta boundary. Tests max coverage without delta. RF=623ms."

echo ">>> apr23_exp4 done"

$PYTHON -m $MODULE $COMMON \
    --k1 127 --k2 15 --k3 25 \
    --run_name "apr23_exp5" \
    --description "Full delta coverage. k1=127 follows the 7→15→31→63→127 doubling pattern. RF=1007ms spans a full 1Hz delta cycle. Largest receptive field in this ablation. RF=1007ms."

echo ">>> apr23_exp5 done"

echo ""
echo "========================================================"
echo "  All 5 apr23 experiments completed."
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "========================================================"
