#!/bin/bash
# Run 12 experiments with different hyperparameter configurations.
# Usage: bash run_experiments.sh

set -e
cd "$(dirname "$0")"

PYTHON=python3

echo "============================================="
echo "  SimpleCNN EEG Experiment Suite (12 runs)"
echo "============================================="

# Experiment 1: Baseline
$PYTHON train.py --lr 1e-3 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 15 \
    --val_subjects S12 S13 --run_name "exp01_baseline"

# Experiment 2: Lower learning rate
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S12 S13 --run_name "exp02_lr1e-4"

# Experiment 3: Higher learning rate
$PYTHON train.py --lr 5e-3 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 15 \
    --val_subjects S12 S13 --run_name "exp03_lr5e-3"

# Experiment 4: Larger batch size
$PYTHON train.py --lr 1e-3 --batch_size 128 --dropout 0.3 --weight_decay 1e-4 --patience 15 \
    --val_subjects S12 S13 --run_name "exp04_bs128"

# Experiment 5: Smaller batch size
$PYTHON train.py --lr 5e-4 --batch_size 32 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S12 S13 --run_name "exp05_bs32_lr5e-4"

# Experiment 6: No dropout
$PYTHON train.py --lr 1e-3 --batch_size 64 --dropout 0.0 --weight_decay 1e-4 --patience 15 \
    --val_subjects S12 S13 --run_name "exp06_no_dropout"

# Experiment 7: High dropout
$PYTHON train.py --lr 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 --patience 20 \
    --val_subjects S12 S13 --run_name "exp07_dropout0.5"

# Experiment 8: Stronger weight decay
$PYTHON train.py --lr 1e-3 --batch_size 64 --dropout 0.3 --weight_decay 1e-3 --patience 15 \
    --val_subjects S12 S13 --run_name "exp08_wd1e-3"

# Experiment 9: No weight decay
$PYTHON train.py --lr 1e-3 --batch_size 64 --dropout 0.3 --weight_decay 0.0 --patience 15 \
    --val_subjects S12 S13 --run_name "exp09_no_wd"

# Experiment 10: Different val subjects
$PYTHON train.py --lr 1e-3 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 15 \
    --val_subjects S01 S06 --run_name "exp10_val_S01_S06"

# Experiment 11: Different val subjects + low LR
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S05 S10 --run_name "exp11_val_S05_S10_lr1e-4"

# Experiment 12: Best guess combo - low LR, high dropout, small batch
$PYTHON train.py --lr 5e-4 --batch_size 32 --dropout 0.4 --weight_decay 5e-4 --patience 25 \
    --val_subjects S12 S13 --run_name "exp12_tuned_combo"

echo ""
echo "============================================="
echo "  All 12 experiments completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "============================================="
