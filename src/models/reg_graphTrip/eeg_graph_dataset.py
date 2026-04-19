"""
PyG dataset for EEG DMT regression using graph representation.

Each 3-second EEG trial becomes a graph:
  - Nodes = EEG channels (32)
  - Node features = spectral band powers (5 bands)
  - Edge features = amplitude envelope correlation (AEC)
  - SPD = all 1s (fully connected graph)
  - Conditional features = electrode 10-10 coordinates (optional)
  - Label = plasma DMT concentration (ng/mL) from PK model

Caches per-subject .pt files. Multi-subject datasets load and concatenate
individual subject caches — no duplication.
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path

from .eeg_features import compute_band_powers, compute_aec, get_electrode_coords

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "eeg_dmt_regression.npz"
CACHE_ROOT = PROJECT_ROOT / "data" / "processed_graphs_regression"


def _build_fully_connected_edge_index(n_nodes):
    """Build edge_index for a fully connected graph (both directions, no self-loops)."""
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _build_spd(n_nodes):
    """SPD matrix for fully connected graph: 0 on diagonal, 1 everywhere else."""
    spd = torch.ones(n_nodes, n_nodes, dtype=torch.long)
    spd.fill_diagonal_(0)
    return spd


def _process_subject(subject, use_coords, data_path=None):
    """
    Process a single subject's trials into a list of PyG Data objects.
    Caches the result to disk so it only runs once per subject/coords combo.
    """
    data_path = data_path or DEFAULT_DATA_PATH
    coords_key = "coords" if use_coords else "nocoords"
    cache_path = CACHE_ROOT / coords_key / f"{subject}.pt"

    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    ds = np.load(data_path, allow_pickle=True)
    all_eeg = ds["eeg_data"]
    all_subjects = ds["subjects"]
    all_labels = ds["labels"]
    channel_labels = list(ds["channel_labels"])

    # Filter to this subject
    mask = all_subjects == subject
    subj_eeg = all_eeg[mask]       # (N_trials, 32, 3000)
    subj_labels = all_labels[mask]  # (N_trials,) ng/mL

    n_nodes = subj_eeg.shape[1]  # 32

    # Shared graph structure
    edge_index = _build_fully_connected_edge_index(n_nodes)
    spd = _build_spd(n_nodes)

    # Electrode coordinates
    if use_coords:
        coords = torch.from_numpy(get_electrode_coords(channel_labels))
    else:
        coords = torch.zeros(n_nodes, 0)

    data_list = []
    n_trials = len(subj_eeg)
    print(f"  Processing {subject}: {n_trials} trials...", end=" ", flush=True)

    for i in range(n_trials):
        eeg = subj_eeg[i]
        x = torch.from_numpy(compute_band_powers(eeg))
        aec_matrix = compute_aec(eeg)
        edge_attr = aec_matrix[edge_index[0].numpy(), edge_index[1].numpy()]
        edge_attr = torch.from_numpy(edge_attr).unsqueeze(1)

        data = Data(
            x=x,
            xc=coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            spd=spd,
            y=torch.tensor(subj_labels[i], dtype=torch.float),
        )
        data_list.append(data)

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, cache_path)
    print(f"done ({n_trials} graphs cached)")

    return data_list


class EEGGraphDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for EEG DMT plasma regression.

    Caches are stored per-subject, so a train set of 6 subjects just loads
    6 individual .pt files — no duplication across different split combinations.

    Parameters
    ----------
    subjects : list of str
        Subject IDs to include (e.g. ['S01', 'S02']).
    use_coords : bool
        If True, include electrode 10-10 coordinates as conditional node features.
    data_path : Path
        Path to the .npz file.
    """

    def __init__(self, subjects, use_coords=False, data_path=None):
        self.subjects = sorted(subjects)
        self.use_coords = use_coords
        self.data_path = data_path or DEFAULT_DATA_PATH

        # We bypass InMemoryDataset's process() entirely and handle caching ourselves
        coords_key = "coords" if use_coords else "nocoords"
        root = str(CACHE_ROOT / coords_key)
        super().__init__(root=root)

        # Load and concatenate per-subject caches
        all_data = []
        for subj in self.subjects:
            all_data.extend(_process_subject(subj, use_coords, self.data_path))

        self._indices = None
        self._data_list = all_data
        self.data, self.slices = self.collate(all_data)

    @property
    def processed_file_names(self):
        # Return existing files so InMemoryDataset doesn't call process()
        return [f"{s}.pt" for s in self.subjects]

    def process(self):
        # Handled by _process_subject — this is never called
        pass


def get_subject_split(val_subjects, all_subjects=None):
    """Return (train_subjects, val_subjects) ensuring no overlap."""
    if all_subjects is None:
        all_subjects = ["S01", "S02", "S05", "S06", "S07", "S10", "S12", "S13"]
    train_subjects = [s for s in all_subjects if s not in val_subjects]
    return train_subjects, val_subjects


def precompute_all(data_path=None):
    """Precompute caches for all subjects, both with and without coords."""
    all_subjects = ["S01", "S02", "S05", "S06", "S07", "S10", "S12", "S13"]
    for use_coords in [False, True]:
        tag = "coords" if use_coords else "nocoords"
        print(f"\n=== {tag} ===")
        for subj in all_subjects:
            _process_subject(subj, use_coords, data_path)
    print("\nAll caches built.")


if __name__ == "__main__":
    precompute_all()
