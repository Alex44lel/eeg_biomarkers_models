#!/bin/bash
#SBATCH --job-name=apr28_sp_wide
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr28 SINGLE-PHASE polyphase sweep — WIDE channels (96/192/384).
# 4 multiseed runs at k=2, 3, 4, 5 on the unfiltered pk_kN datasets,
# keeping only k_idx==0 of each parent window (--single_phase).
#
# Wide capacity ≈ 2× compute vs default → ~60–180 min per run, ~10h total.
# 16h leaves comfortable margin.
#
# Companion job: cluster_apr28_singlephase_default.sh runs the same 4 k's
# at default channels (64/128/256). The two jobs together fill the full
# 4×2 grid.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_singlephase_sweep.sh wide
