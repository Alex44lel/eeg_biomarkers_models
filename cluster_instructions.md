# IC DoC GPU Cluster — Running ML Training Jobs

Reference: https://systems.pages.doc.ic.ac.uk/gpucluster/  
Last checked: 2026-04-25

---

## 1. Connect to the Cluster

From a DoC lab PC:
```bash
ssh gpucluster2.doc.ic.ac.uk
```

From a laptop (SSH key required — password auth disabled since Dec 2024):
```bash
ssh gpucluster2.doc.ic.ac.uk
# or via JumpHost if the above is unreachable:
ssh -J shell5.doc.ic.ac.uk gpucluster2.doc.ic.ac.uk
```

> SSH keys must be set up while physically onsite at Imperial.

---

## 2. Set Up Storage

Use a CephFS workspace (recommended over `/vol/bitbucket` for performance):
```bash
# Run once on the head node
ws_allocate tfm 365
# Creates: /vol/gpudata/${USER}-tfm   (50 GB, auto-deleted after 365 days)
```

Or legacy NFS:
```bash
mkdir -p /vol/bitbucket/${USER}/tfm
```

Clone / copy the project to shared storage so every GPU node can see it:
```bash
cd /vol/bitbucket/${USER}   # or /vol/gpudata/${USER}-tfm
git clone <your-repo-url> tfm_code
# or rsync from local:
rsync -av --exclude='mlruns/' --exclude='data/' ~/Desktop/tfm_code/ \
    ${USER}@shell1.doc.ic.ac.uk:/vol/bitbucket/${USER}/tfm_code/
```

Copy data (mat files / preprocessed datasets) to the same shared location:
```bash
rsync -av ~/Desktop/tfm_code/data/ \
    ${USER}@shell1.doc.ic.ac.uk:/vol/bitbucket/${USER}/tfm_code/data/
```

---

## 3. Set Up Python Environment

**Do this from a lab PC, never from the cluster head node** (head node has limited disk):
```bash
ssh shell1.doc.ic.ac.uk
/vol/linux/bin/sshtolab          # hop to a lab PC with a local disk
cd /vol/bitbucket/${USER}
python3 -m virtualenv tfm_venv
source tfm_venv/bin/activate
pip install -r /vol/bitbucket/${USER}/tfm_code/requirements.txt
```

Or use the pre-built starter env (PyTorch + TF already installed) to avoid setup:
```bash
source /vol/bitbucket/starter/bin/activate
```

---

## 4. Write an sbatch Wrapper Script

The existing shell scripts (e.g. `run_apr25_polyphase_rf_matched.sh`) run multiple
sequential Python experiments. Wrap them in an sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=tfm_poly_rf
#SBATCH --gres=gpu:1
#SBATCH --partition=a30          # 24 GB VRAM; swap for t4 (16 GB) or a40 (48 GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00        # max 4 days; jobs are killed after this
#SBATCH --output=/vol/bitbucket/${USER}/tfm_code/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# --- Environment ---
export PATH=/vol/bitbucket/${USER}/tfm_venv/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh

# Matplotlib must use a non-interactive backend on cluster nodes
export MPLBACKEND=Agg

# --- Project root ---
cd /vol/bitbucket/${USER}/tfm_code

nvidia-smi   # sanity check: confirms GPU allocated

# --- Run experiment sweep ---
bash src/models/reg_simpleCNN/shell_and_logs/run_apr25_polyphase_rf_matched.sh
```

Save as e.g. `scripts/cluster_poly_rf.sh`.

---

## 5. Submit and Monitor

```bash
# From the head node
cd /vol/bitbucket/${USER}/tfm_code
mkdir -p logs

sbatch scripts/cluster_poly_rf.sh

# Check status
squeue --me

# Live-tail output (once job starts)
tail -f logs/slurm-<job_id>.out

# Cancel a job
scancel <job_id>
```

---

## 6. Choose a GPU Partition

| Partition | GPU | VRAM | Notes |
|-----------|-----|------|-------|
| `t4` | Tesla T4 | 16 GB | Shortest queue; enough for batch_size=64 |
| `a30` | Tesla A30 | 24 GB | Good balance of memory and wait time |
| `a40` | Tesla A40 | 48 GB | Longer queue |
| `a100` | Tesla A100 | 80 GB | Longest queue; overkill for 1D CNN |
| `training` | Shared/experimental | — | Use `--gres=shard:1` instead of `gpu:1` |

For a 1D CNN with ~800k parameters and batch_size=64, `t4` or `a30` is sufficient.

---

## 7. Fair Use Limits

- Max **2 running jobs** per user at once (extras queue).
- Max **3 GPUs** and **32 CPU cores** per user.
- Max **4-day runtime** — jobs are killed automatically; add checkpointing for anything that might run longer.
- Do not run computation on the head node itself.

---

## 8. MLflow on the Cluster

The experiment sweep logs to `mlruns/` relative to the working directory. After the
job finishes, either:

- **Sync back locally** and view:
  ```bash
  rsync -av ${USER}@shell1.doc.ic.ac.uk:/vol/bitbucket/${USER}/tfm_code/mlruns/ \
      ~/Desktop/tfm_code/mlruns/
  mlflow ui --backend-store-uri mlruns/
  ```

- **View on the cluster** via an SSH tunnel:
  ```bash
  # On the cluster (interactive session or head node)
  mlflow ui --backend-store-uri /vol/bitbucket/${USER}/tfm_code/mlruns/ --port 5001

  # On your laptop
  ssh -L 5001:localhost:5001 gpucluster2.doc.ic.ac.uk
  # then open http://localhost:5001
  ```

---

## 9. Quick Reference Commands

```bash
squeue --me                    # list your jobs
squeue                         # list all jobs (see load)
scancel <job_id>               # kill a job
salloc --gres=gpu:1            # interactive GPU shell (up to 3 days)
nvidia-smi                     # check GPU in interactive session
ws_list                        # list your CephFS workspaces
ws_allocate <name> <days>      # create a workspace (max 365 days)
ws_release <name>              # delete a workspace
```
