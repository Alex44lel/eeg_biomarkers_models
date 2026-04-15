#!/bin/bash
# Sweep over 12 different validation subject combinations using exp03 hyperparams.
# Hyperparams: lr=1e-4, batch_size=16, dropout=0.1, weight_decay=1e-4, cls_weight=1.0, patience=20, no coords
# Usage: bash run_val_sweep.sh

set -e
cd "$(dirname "$0")"

PYTHON=python3

echo "================================================"
echo "  graphTrip Val Subject Sweep (12 splits)"
echo "  Hyperparams: lr=1e-4 bs=16 do=0.1 wd=1e-4"
echo "================================================"

# Subjects: S01 S02 S04 S05 S06 S07 S08 S10 S12 S13

# Non-overlapping pairs (cover all 10 subjects)
$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S01 S02 \
    --run_name "val_sweep_S01_S02"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S04 S05 \
    --run_name "val_sweep_S04_S05"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S06 S07 \
    --run_name "val_sweep_S06_S07"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S08 S10 \
    --run_name "val_sweep_S08_S10"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S12 S13 \
    --run_name "val_sweep_S12_S13"

# Mixed pairings
$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S01 S07 \
    --run_name "val_sweep_S01_S07"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S02 S10 \
    --run_name "val_sweep_S02_S10"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S04 S13 \
    --run_name "val_sweep_S04_S13"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S05 S08 \
    --run_name "val_sweep_S05_S08"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S06 S12 \
    --run_name "val_sweep_S06_S12"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S01 S13 \
    --run_name "val_sweep_S01_S13"

$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S05 S10 \
    --run_name "val_sweep_S05_S10"

echo ""
echo "================================================"
echo "  All 12 validation splits completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "================================================"
