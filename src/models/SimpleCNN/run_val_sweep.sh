#!/bin/bash
# Sweep over 10 different validation subject combinations using exp11 hyperparams.
# Hyperparams: lr=1e-4, batch_size=64, dropout=0.3, weight_decay=1e-4, patience=20
# Usage: bash run_val_sweep.sh

set -e
cd "$(dirname "$0")"

PYTHON=python3

echo "============================================="
echo "  Validation Subject Sweep (10 splits)"
echo "  Hyperparams: lr=1e-4 bs=64 do=0.3 wd=1e-4"
echo "============================================="

# Subjects: S01 S02 S04 S05 S06 S07 S08 S10 S12 S13

# Split 1
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S01 S02 --run_name "val_sweep_S01_S02"

# Split 2
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S04 S05 --run_name "val_sweep_S04_S05"

# Split 3
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S06 S07 --run_name "val_sweep_S06_S07"

# Split 4
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S08 S10 --run_name "val_sweep_S08_S10"

# Split 5
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S12 S13 --run_name "val_sweep_S12_S13"

# Split 6
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S01 S07 --run_name "val_sweep_S01_S07"

# Split 7
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S02 S10 --run_name "val_sweep_S02_S10"

# Split 8
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S04 S13 --run_name "val_sweep_S04_S13"

# Split 9
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S05 S08 --run_name "val_sweep_S05_S08"

# Split 10
$PYTHON train.py --lr 1e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 --patience 20 \
    --val_subjects S06 S12 --run_name "val_sweep_S06_S12"

echo ""
echo "============================================="
echo "  All 10 validation splits completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "============================================="
