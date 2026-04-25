#!/bin/bash
#SBATCH --job-name=apr25_poly_rf
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr25_polyphase_rf_matched.sh
