#!/bin/bash
#SBATCH --job-name=apr28_k34
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr28 capacity sweep for k=3 and k=4 polyphase — 6 multiseed runs.
# Fills the missing k=3 / k=4 columns of the apr26 capacity grid (narrow/default/wide).
#
#   k=3  kernels=24/4/6   (RF=208 sub-samples = 624 raw-ms, Nyquist≈167Hz, L=1000)
#   k=4  kernels=13/3/5   (RF=157 sub-samples = 628 raw-ms, Nyquist=125Hz,  L=750)
#
# Estimated wall time: ~2h/run at narrow/default, ~3h at wide → ~14h total.
# 20h leaves a comfortable margin.
#
# Reference comparisons (already in mlruns):
#   k=2  narrow/default/wide:  apr26_multiseed_k2_{narrow,default,wide}
#   k=5  narrow/default/wide:  apr26_multiseed_k5_{narrow,default,wide}

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

SWEEP="src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh"

bash "$SWEEP" narrow
bash "$SWEEP" default
bash "$SWEEP" wide
