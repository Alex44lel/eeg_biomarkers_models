#!/bin/bash
#SBATCH --job-name=apr24_multiseed
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/bitbucket/ac5725/tfm_code/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

export PATH=/vol/bitbucket/${USER}/tfm_venv/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg

nvidia-smi

cd /vol/bitbucket/ac5725/tfm_code

bash src/models/reg_simpleCNN/shell_and_logs/run_apr24_multiseed_exps.sh
