# Plan: Adapt graphTrip to EEG DMT classification

## Context

graphTrip is a Variational Graph Autoencoder (VGAE) originally built for fMRI brain graphs
(80 ROIs, REACT node features, FC edge features) → predicts depression scores (regression).

We adapt it for: EEG 3-second trials (32 channels) → classify pre vs post DMT injection (binary).

## What stays the same from model.py

- `NodeEmbeddingGraphormer` (encoder) — just different input dims
- `DenseOneLayerEncoder` (variational layer: embeddings → mu, logvar)
- `GraphTransformerPooling` (readout: node latents → graph-level vector)
- `MLPNodeDecoder` (reconstructs node features from latents)
- `MLPEdgeDecoder` (reconstructs edge features from latent pairs)
- VGAE loss: reconstruction MSE + KL divergence + L2 regularization

## What changes

| Aspect | Original graphTrip | Our EEG version |
|--------|-------------------|-----------------|
| Nodes | 80 brain ROIs | 32 EEG channels |
| Node features | 3 REACT values (serotonin receptor densities) | 5 spectral band powers (delta, theta, alpha, beta, gamma) |
| Conditional features | 3 MNI coordinates | 3 electrode (x,y,z) coords **or** 0 (flag: `--use_coords`) |
| Edge features | FC correlation (1 value) | Amplitude envelope correlation, AEC (1 value) |
| SPD | Real shortest paths in structural graph | All 1s (fully connected, every node 1 hop from every other) |
| Prediction head | RegressionMLP → QIDS score (MSE) | ClassificationMLP → 2 classes (CrossEntropy) |
| Total loss | VGAE loss + MSE regression | VGAE loss + CrossEntropy classification |

## Files to create

### 1. `eeg_features.py` — Feature extraction functions

Pure functions, no model code.

- `compute_band_powers(eeg_trial, fs=1000)` 
  - Input: (32, 3000) single trial
  - Welch PSD → integrate over 5 bands: delta(1-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-45)
  - Output: (32, 5) — node features
  - Log-transform the powers (standard in EEG lit)

- `compute_aec(eeg_trial, fs=1000)`
  - Input: (32, 3000) single trial  
  - Broadband: Hilbert transform → amplitude envelope → Pearson correlation between all pairs
  - Output: (32, 32) symmetric correlation matrix — edge features

- `ELECTRODE_COORDS_10_10` — dict mapping channel names → (x, y, z) on unit sphere
  - Standard 10-10 system coordinates for our 32 channels
  - Used when `--use_coords` flag is set

### 2. `eeg_graph_dataset.py` — PyG Dataset

- `EEGGraphDataset(InMemoryDataset)`
  - Loads `eeg_dmt_dataset.npz`, filters by subject list
  - For each trial, calls `compute_band_powers` and `compute_aec`
  - Builds `torch_geometric.data.Data` objects with:
    - `data.x` = band powers (32, 5) — node features
    - `data.xc` = electrode coords (32, 3) if `use_coords=True`, else zeros (32, 0)
    - `data.edge_index` = fully connected graph (both directions)
    - `data.edge_attr` = AEC values for each edge (E, 1)
    - `data.spd` = all 1s off-diagonal, 0 on diagonal (32, 32)
    - `data.y` = label (0 or 1)
  - Precomputes all features and holds everything in RAM
  - Caches to disk (PyG `processed_dir`) so feature extraction only runs once

### 3. `eeg_classifier.py` — Model wrapper

Bypasses the `NodeLevelVGAE` config system (which has broken imports to `models.utils`)
and directly instantiates the components from `model.py`:

- `EEGGraphClassifier(nn.Module)`:
  - `self.encoder` = `NodeEmbeddingGraphormer(num_node_attr=5, num_cond_attrs=0or3, num_edge_attr=1, ...)`
  - `self.var_encoder` = `DenseOneLayerEncoder(input_dim=node_emb_dim, latent_dim=latent_dim)`
  - `self.pooling` = `GraphTransformerPooling(pooling_dim=latent_dim)`
  - `self.node_decoder` = `MLPNodeDecoder(latent_dim=latent_dim, output_dim=5)`
  - `self.edge_decoder` = `MLPEdgeDecoder(latent_dim=latent_dim, act='tanh')`
  - `self.classifier` = `nn.Sequential(Linear → ReLU → Linear → 2 classes)`
  
  - `forward(batch)` → returns Outputs + logits
  - `loss(outputs, logits, labels)`:
    - `vgae_loss` = reconstruction (nodes + edges) + KL divergence
    - `cls_loss` = CrossEntropyLoss(logits, labels)
    - `total = vgae_loss + cls_weight * cls_loss`

### 4. `train_eeg.py` — Training script (adapted from SimpleCNN/train.py)

Same structure as SimpleCNN training:
- argparse with: `--val_subjects`, `--lr`, `--batch_size`, `--epochs`, `--patience`,
  `--dropout`, `--weight_decay`, `--use_coords` (flag), `--cls_weight` (balance VGAE vs classification loss)
- Subject-level train/val split (no subject leakage)
- PyG `DataLoader` instead of plain DataLoader
- Adam optimizer, early stopping on val loss
- MLflow logging: train/val loss, accuracy, precision, recall, specificity, NPV
- Confusion matrix plot (with val accuracy)
- Latent space t-SNE plot (graph-level vectors)
- Model saved as MLflow artifact

### 5. `run_eeg_experiments.sh` — Experiment runner

12 experiments varying:
- `--use_coords` on/off
- Learning rate
- `--cls_weight` (how much to weight classification vs reconstruction)
- Dropout
- Validation subject splits

## Import fix for model.py

`model.py` line 22 does `from models.utils import get_model_configs, init_model`.
`utils.py` line 12 does `import models`. These assume a package layout from the original
graphTrip repo that doesn't match our directory structure.

Solution: `eeg_classifier.py` will import the **classes directly** from `model.py` using
a relative import or sys.path fix, bypassing `NodeLevelVGAE` and its config system entirely.
We only need the building-block classes (`NodeEmbeddingGraphormer`, `DenseOneLayerEncoder`,
`GraphTransformerPooling`, `MLPNodeDecoder`, `MLPEdgeDecoder`) which are plain `nn.Module`
subclasses with normal `__init__` signatures — no config system needed.

## Execution order

1. `eeg_features.py` (no dependencies)
2. `eeg_graph_dataset.py` (depends on 1)
3. `eeg_classifier.py` (depends on model.py classes)
4. `train_eeg.py` (depends on 2 + 3)
5. `run_eeg_experiments.sh` (calls 4)
