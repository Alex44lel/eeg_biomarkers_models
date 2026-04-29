"""
PyTorch Dataset for EEG DMT plasma-concentration regression.

Loads the entire .npz dataset into RAM for fast training.
Supports subject-level train/validation splits.

Optional `bandpower_features=True` mode (used by --model bandpower_linear in
train_cv.py): the dataset computes Welch PSD per (trial × channel), integrates
over a fixed list of frequency bands, optionally log-compresses and z-scores,
and exposes the resulting (N, F) flat feature tensor in place of the raw
waveforms. The baseline buffer and the rest of the public surface stay the
same — feature shape is just 1-D per sample instead of 2-D (32, L).
"""

import numpy as np
import torch
from scipy.signal import welch
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

# Default frequency bands for `bandpower_features=True`. 6 bands × 32 channels
# = 192 features. Edges in Hz; bands are integrated as [lo, hi] inclusive on
# both sides (the integration mask uses lo <= f <= hi). The high-γ band
# (45–100 Hz) is included by default — the source data is software low-passed
# at ~100 Hz so the band is fully usable, and DMT modulates γ.
DEFAULT_BANDPOWER_BANDS = (
    (1.0, 4.0),     # delta
    (4.0, 8.0),     # theta
    (8.0, 13.0),    # alpha
    (13.0, 30.0),   # beta
    (30.0, 45.0),   # low gamma
    (45.0, 100.0),  # high gamma
)

# Subjects with PK posterior-mean y1 labels (matches reg_graphTrip / build_dmt_dataset.py)
ALL_SUBJECTS = ["S01", "S02", "S05", "S06", "S07", "S10", "S12", "S13"]

# Canonical global subject -> int index mapping. Stable across train/val splits
# so the baseline-subtraction code can gather per-subject mu_s by integer index.
SUBJECT_TO_IDX = {s: i for i, s in enumerate(ALL_SUBJECTS)}


def compute_bandpower_features(eeg, fs, bands, nperseg, log=True):
    """Welch PSD → per-band integration → optional log compression.

    Args:
        eeg:     float array of shape (N, C, L) — raw waveforms.
        fs:      sampling rate in Hz.
        bands:   iterable of (lo, hi) Hz pairs.
        nperseg: Welch segment length (auto-clipped to L).
        log:     if True, return log(eps + power) for dynamic-range compression.

    Returns:
        features: float32 array of shape (N, C * len(bands)). Channel-major
                  flatten: row layout is [c0_b0, c0_b1, ..., c0_bB-1,
                  c1_b0, ...] so `features[:, c * B + b]` is channel c,
                  band b.
    """
    n, c, l = eeg.shape
    nperseg_eff = int(min(nperseg, l))
    # welch on shape (N*C, L) is the most efficient call — scipy vectorises
    # the trailing axis, then we unflatten.
    flat = eeg.reshape(n * c, l)
    f, psd = welch(flat, fs=fs, window="hann",
                   nperseg=nperseg_eff,
                   noverlap=nperseg_eff // 2,
                   axis=-1, scaling="density")
    # psd: (n*c, n_freq); f: (n_freq,)
    # Sanity: bands within [0, fs/2]
    nyq = fs / 2.0
    out_per_band = np.empty((n * c, len(bands)), dtype=np.float64)
    for bi, (lo, hi) in enumerate(bands):
        if hi > nyq:
            raise ValueError(
                f"Band ({lo}, {hi}) Hz exceeds Nyquist {nyq} Hz at fs={fs}. "
                "Drop or shrink the high-γ band, or use a higher-fs source."
            )
        mask = (f >= lo) & (f <= hi)
        if not mask.any():
            raise ValueError(
                f"Band ({lo}, {hi}) Hz contains no Welch bins at "
                f"fs={fs} nperseg={nperseg_eff} (bin width "
                f"{fs / nperseg_eff:.3f} Hz)."
            )
        # Trapezoidal integration over the masked bins. np.trapezoid is the
        # NumPy 2.x name; fall back to np.trapz on older numpy.
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        out_per_band[:, bi] = _trapz(psd[:, mask], f[mask], axis=-1)
    out = out_per_band.reshape(n, c * len(bands))
    if log:
        out = np.log(out + 1e-12)
    return out.astype(np.float32)


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
                       Ignored when bandpower_features=True (Welch PSD is
                       computed on the raw waveforms; per-trial z-score in
                       the time domain destroys total power).
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

        bandpower_features:    If True, replace raw waveforms with Welch-PSD
                       band-power features computed lazily here. `self.eeg`
                       becomes (N, F) instead of (N, 32, L). Used by
                       --model bandpower_linear.
        bandpower_bands:       Iterable of (lo, hi) Hz pairs. Defaults to
                       DEFAULT_BANDPOWER_BANDS (6 bands → 192 features).
        bandpower_log:         Apply log(eps + power) compression. Default True.
        bandpower_welch_nperseg: Welch segment length (auto-clipped to L).
        bandpower_zscore_features: Standardise the feature matrix using
                       per-feature mean/std. Default True.
        bandpower_norm_stats: When given as `(mean, std)` arrays of shape
                       (F,), use these instead of computing stats from this
                       dataset's post-inj rows. The val EEGDataset must
                       receive the train EEGDataset's stats — otherwise the
                       train-fit feature scaler leaks across the LOSO split.

    Items returned by __getitem__: (eeg, label, subject_idx) where
    subject_idx is a global int from SUBJECT_TO_IDX.
    """

    def __init__(self, data_path=None, subjects=None, normalize=True,
                 dataset=None, single_phase=False, load_baseline=False,
                 bandpower_features=False, bandpower_bands=None,
                 bandpower_log=True, bandpower_welch_nperseg=512,
                 bandpower_zscore_features=True,
                 bandpower_norm_stats=None):
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

        self.bandpower_features = bool(bandpower_features)
        self.bandpower_norm_stats = None  # populated below if applicable

        if self.bandpower_features:
            # Lazy bandpower path. fs derived from the trial length: every
            # source npz spans 3 s, so fs = L / 3 (k=1 → 1000 Hz, k=2 → 500).
            _, n_ch, l = all_eeg.shape
            fs = l / 3.0
            bands = (tuple(bandpower_bands)
                     if bandpower_bands is not None
                     else DEFAULT_BANDPOWER_BANDS)
            features = compute_bandpower_features(
                all_eeg, fs=fs, bands=bands,
                nperseg=bandpower_welch_nperseg, log=bandpower_log,
            )
            # Optional feature-space z-score. Stats come from this dataset's
            # post-injection rows unless explicitly provided (train -> val).
            if bandpower_zscore_features:
                if bandpower_norm_stats is not None:
                    fmean, fstd = bandpower_norm_stats
                else:
                    post_rows = features[~all_is_baseline]
                    fmean = post_rows.mean(axis=0)
                    fstd = post_rows.std(axis=0) + 1e-8
                features = (features - fmean) / fstd
                self.bandpower_norm_stats = (fmean.astype(np.float32),
                                              fstd.astype(np.float32))

            all_eeg = features  # (N, F)
            self._bandpower_meta = {
                "fs": fs, "n_channels": int(n_ch), "n_bands": len(bands),
                "bands": tuple((float(lo), float(hi)) for lo, hi in bands),
                "log": bool(bandpower_log),
                "nperseg": int(min(bandpower_welch_nperseg, l)),
            }
        else:
            self._bandpower_meta = None
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
        if self.bandpower_features:
            # eeg.shape == (N, F) here. Expose feature_dim and keep the
            # legacy attributes pointing at sensible values for downstream
            # mlflow logging (n_channels=1 is a logical sentinel).
            self.feature_dim = int(self.eeg.shape[1])
            self.n_channels = 1
            self.signal_length = self.feature_dim
        else:
            self.n_channels = self.eeg.shape[1]
            self.signal_length = self.eeg.shape[2]
            self.feature_dim = None

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
