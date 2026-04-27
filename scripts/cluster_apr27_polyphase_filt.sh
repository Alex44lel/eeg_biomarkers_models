#!/bin/bash
#SBATCH --job-name=apr27_polyfilt
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr27 ANTI-ALIASED polyphase sweep — 3 multiseed runs at k=2/5/10 on the
# filtered datasets (pk_kN_filt). Tests whether the apr25 k=5 / k=10 collapse
# was caused by aliasing in the unfiltered build_downsampled_dataset.py.
#
# Estimated wall time: ~5–6h on a30 (3 runs × ~100min, default channels =
# same compute as apr25). 12h gives ample safety margin.
#
# Prerequisites on the cluster:
#   - data/eeg_dmt_regression_k{2,5,10}_filt.npz must exist. Build them with:
#       python -m src.models.reg_simpleCNN.build_downsampled_dataset_filt --k 2 5 10
#   - dataset.py must include the pk_k{2,5,10}_filt entries in DATASET_PATHS.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr27_polyphase_filt.sh
