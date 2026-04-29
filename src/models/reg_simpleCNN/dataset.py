"""
PyTorch Dataset for EEG DMT plasma-concentration regression.

Loads the entire .npz dataset into RAM for fast training.
Supports subject-level train/validation splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "eeg_dmt_regression.npz"

# Label-source registry. Keys are the --dataset CLI values consumed by train_cv.
#   pk       : PK-model posterior-mean y1 curve (default, built by reg_graphTrip/build_dmt_dataset.py)
#   biexp    : per-subject bi-exponential fit (built by reg_simpleCNN/build_biexp_dataset.py)
#   pk_k{K}  : polyphase-downsampled view of `pk` with factor K ∈ {2,3,4}.
#              K * N_orig rows; each row has length 3000//K and a `k_idx`
#              field in [0, K). Subject IDs are preserved so LOSO grouping
#              keeps all K splits of a subject together.
#              Built by reg_simpleCNN/build_downsampled_dataset.py.
DATASET_PATHS = {
    "pk":     DEFAULT_DATA_PATH,
    "biexp":  PROJECT_ROOT / "data" / "eeg_dmt_regression_biexp.npz",
    "pk_k2":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k2.npz",
    "pk_k3":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k3.npz",
    "pk_k4":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k4.npz",
    "pk_k5":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k5.npz",
    "pk_k10": PROJECT_ROOT / "data" / "eeg_dmt_regression_k10.npz",
    # Anti-aliased polyphase variants (filter-then-decimate, see
    # build_downsampled_dataset_filt.py). Schema identical to pk_k{K}; only
    # difference is an FIR low-pass applied before each phase is taken,
    # eliminating aliasing of >Nyquist content into the kept bands. Use
    # these to test whether the k=5/k=10 collapse on the unfiltered builds
    # was caused by aliasing or by genuine info loss.
    "pk_k2_filt":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k2_filt.npz",
    "pk_k5_filt":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k5_filt.npz",
    "pk_k10_filt": PROJECT_ROOT / "data" / "eeg_dmt_regression_k10_filt.npz",
    # Baseline-aware variants (extra `is_baseline` column with pre-injection
    # trials kept aside for the linear subject adaptation in train_cv.py
    # `--baseline_subtraction`). Same schema otherwise as the matching
    # `pk*` build.
    "pk_with_baseline":    PROJECT_ROOT / "data" / "eeg_dmt_regression_with_baseline.npz",
    "pk_k2_with_baseline": PROJECT_ROOT / "data" / "eeg_dmt_regression_with_baseline_k2.npz",
}

# Subjects with PK posterior-mean y1 labels (matches reg_graphTrip / build_dmt_dataset.py)
ALL_SUBJECTS = ["S01", "S02", "S05", "S06", "S07", "S10", "S12", "S13"]

# Canonical global subject -> int index mapping. Stable across train/val splits
# so the baseline-subtraction code can gather per-subject mu_s by integer index.
SUBJECT_TO_IDX = {s: i for i, s in enumerate(ALL_SUBJECTS)}


def resolve_dataset_path(dataset=None, data_path=None):
    """Return a Path to the .npz to load.

    Priority: explicit data_path > dataset key > DEFAULT_DATA_PATH.
    """
    if data_path is not None:
        return Path(data_path)
    if dataset is not None:
        if dataset not in DATASET_PATHS:
            raise ValueError(
                f"Unknown dataset key {dataset!r}. "
                f"Available: {sorted(DATASET_PATHS)}"
            )
        return Path(DATASET_PATHS[dataset])
    return Path(DEFAULT_DATA_PATH)


class EEGDataset(Dataset):
    """
    EEG dataset for plasma DMT regression.

    The source .npz already has the 32 pure EEG channels — no exclusion needed.

    Args:
        data_path:     Path to .npz file. If given, takes precedence over `dataset`.
        dataset:       Registry key in DATASET_PATHS ('pk' or 'biexp'). Ignored
                       if data_path is set. Defaults to DEFAULT_DATA_PATH when
                       both are None.
        subjects:      List of subject IDs to include (e.g. ['S01', 'S02']).
        normalize:     If True, z-score normalise per channel per trial.
        load_baseline: If True, expose pre-injection trials via
                       `self.baseline_eeg` (dict subject_id -> tensor of shape
                       (N_baseline, 32, L)). Pre-injection rows are still
                       removed from the supervised arrays (`self.eeg`,
                       `self.labels`, etc.) — they are NOT iterated by
                       __getitem__. Requires the source .npz to have an
                       `is_baseline` column.
                       If False (default), pre-injection rows (if any) are
                       silently dropped — old training scripts work unchanged
                       with a baseline-aware npz.

    Items returned by __getitem__: (eeg, label, subject_idx) where
    subject_idx is a global int from SUBJECT_TO_IDX.
    """

    def __init__(self, data_path=None, subjects=None, normalize=True,
                 dataset=None, single_phase=False, load_baseline=False):
        data_path = resolve_dataset_path(dataset=dataset, data_path=data_path)
        ds = np.load(data_path, allow_pickle=True)

        all_eeg = ds["eeg_data"]        # (N, 32, L)
        all_subjects = ds["subjects"]   # (N,) subject IDs
        all_labels = ds["labels"]       # (N,) plasma ng/mL (NaN for baseline rows)
        all_times = ds["times"]         # (N,) minutes post-dose

        # Optional polyphase-downsampling metadata (present in
        # eeg_dmt_regression_k{K}.npz produced by build_downsampled_dataset).
        all_k_idx = ds["k_idx"] if "k_idx" in ds.files else None
        all_orig_id = ds["orig_trial_id"] if "orig_trial_id" in ds.files else None
        self.k_factor = int(ds["k_factor"]) if "k_factor" in ds.files else 1

        # Optional baseline mask. Absent column = "no baseline rows present".
        if "is_baseline" in ds.files:
            all_is_baseline = ds["is_baseline"].astype(bool)
        else:
            all_is_baseline = np.zeros(len(all_subjects), dtype=bool)

        if load_baseline and not all_is_baseline.any():
            raise ValueError(
                f"load_baseline=True but {data_path} has no is_baseline=True rows. "
                "Build the dataset with --include-baseline first."
            )

        # Filter by subjects
        if subjects is not None:
            mask = np.isin(all_subjects, subjects)
            all_eeg = all_eeg[mask]
            all_labels = all_labels[mask]
            all_subjects = all_subjects[mask]
            all_times = all_times[mask]
            all_is_baseline = all_is_baseline[mask]
            if all_k_idx is not None:
                all_k_idx = all_k_idx[mask]
            if all_orig_id is not None:
                all_orig_id = all_orig_id[mask]

        # Drop all polyphase phases except phase 0. Each parent window then
        # contributes exactly one decimated row (length 3000//K), making the
        # dataset structurally equivalent to a non-polyphase k=1 build at the
        # lower sampling rate — no phase-repetition "augmentation". Triggering
        # this on a non-polyphase dataset is a no-op.
        if single_phase and all_k_idx is not None:
            mask = all_k_idx == 0
            all_eeg = all_eeg[mask]
            all_labels = all_labels[mask]
            all_subjects = all_subjects[mask]
            all_times = all_times[mask]
            all_is_baseline = all_is_baseline[mask]
            all_k_idx = None
            all_orig_id = None
            self.k_factor = 1

        # Convert to float32
        all_eeg = all_eeg.astype(np.float32)

        # Z-score normalize per channel per trial (signal scaling only; labels stay raw ng/mL).
        # Applied to all rows uniformly (post-inj and baseline) so mu_s computed
        # from baseline trials lives in the same feature space as post-inj inputs.
        if normalize:
            mean = all_eeg.mean(axis=2, keepdims=True)
            std = all_eeg.std(axis=2, keepdims=True) + 1e-8
            all_eeg = (all_eeg - mean) / std

        # Split into baseline (kept aside, not iterated) and post-injection
        # (the supervised set returned by __getitem__).
        baseline_mask = all_is_baseline
        post_mask = ~baseline_mask

        if load_baseline:
            # Per-subject baseline buffer; held as raw tensors for use by the
            # training loop's mu_s refresh step.
            baseline_eeg = {}
            base_eeg_arr = all_eeg[baseline_mask]
            base_subj_arr = all_subjects[baseline_mask]
            for subj in np.unique(base_subj_arr):
                rows = base_subj_arr == subj
                baseline_eeg[str(subj)] = torch.from_numpy(base_eeg_arr[rows])
            self.baseline_eeg = baseline_eeg
        else:
            self.baseline_eeg = None

        # Apply the post-inj filter to the supervised arrays
        all_eeg = all_eeg[post_mask]
        all_labels = all_labels[post_mask]
        all_subjects = all_subjects[post_mask]
        all_times = all_times[post_mask]
        if all_k_idx is not None:
            all_k_idx = all_k_idx[post_mask]
        if all_orig_id is not None:
            all_orig_id = all_orig_id[post_mask]

        # Store in RAM as tensors
        self.eeg = torch.from_numpy(all_eeg)
        self.labels = torch.from_numpy(all_labels.astype(np.float32))
        self.subjects = all_subjects
        # Per-sample global subject index, for gathering mu_s in the train loop.
        self.subject_idx = torch.tensor(
            [SUBJECT_TO_IDX[str(s)] for s in all_subjects], dtype=torch.long,
        )
        self.times = all_times.astype(np.float32)
        self.k_idx = all_k_idx              # None when source is non-polyphase
        self.orig_trial_id = all_orig_id    # None when source is non-polyphase
        self.n_channels = self.eeg.shape[1]
        self.signal_length = self.eeg.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx], self.subject_idx[idx]


def get_subject_split(val_subjects, all_subjects=None):
    """Return (train_subjects, val_subjects) ensuring no overlap."""
    if all_subjects is None:
        all_subjects = ALL_SUBJECTS
    train_subjects = [s for s in all_subjects if s not in val_subjects]
    return train_subjects, val_subjects
