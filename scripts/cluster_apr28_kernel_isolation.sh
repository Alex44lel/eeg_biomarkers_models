#!/bin/bash
#SBATCH --job-name=apr28_kernels
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr28 KERNEL ISOLATION sweep — 2 multiseed runs that disentangle kernel size
# from polyphase sample-rate as causes of the apr25 k=5 collapse.
#
# Exp A: pk_k5 with kernels (31, 8, 8) — apr28_multiseed_k5_bigkern
# Exp B: pk    with kernels (13, 3, 4) — apr28_multiseed_k1_smallkern
#
# See run_apr28_kernel_isolation.sh and misc/study_notes/findings.md for the
# full hypothesis, predictions, and decision matrix.
#
# Estimated wall time: ~3–4h on a30 (2 multiseed runs × ~100 min each at
# default channels — same compute profile as apr25). 8h is generous but
# leaves headroom if the bigger-kernel run is slower at L=3000 or L=600.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_kernel_isolation.sh
