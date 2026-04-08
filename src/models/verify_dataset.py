"""
Verify the extracted .npz dataset against the original .mat files.

Plots:
1. A random trial from .npz and the same trial from .mat side-by-side
2. Distribution of trials per subject and group
3. Mean EEG signal amplitude per group (label 0 vs 1)
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_verification"

SUBJECT_FOLDERS = {
    "S01": "S01AS",
    "S02": "S02WT",
    "S04": "S04SG",
    "S05": "S05LM",
    "S06": "S06ET",
    "S07": "S07CS",
    "S08": "S08EK",
    "S10": "S10DL",
    "S12": "S12AI",
    "S13": "S13MBJ",
}

FS = 1000


def load_trial_from_mat(subject_id, group_idx, trial_idx):
    """Load a single trial directly from the .mat file for comparison."""
    folder = SUBJECT_FOLDERS[subject_id]
    mat_path = RECORDINGS_DIR / folder / "DMT" / "data_trialsmxm_3s.mat"
    d = sio.loadmat(str(mat_path), squeeze_me=True)
    cells = d["data_trialsmxm_3s"]
    cell = cells[group_idx][()]
    return cell["trial"][trial_idx]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the .npz dataset
    npz_path = DATA_DIR / "eeg_dmt_dataset.npz"
    ds = np.load(npz_path, allow_pickle=True)

    eeg = ds["eeg"]
    subjects = ds["subjects"]
    groups = ds["groups"]
    labels = ds["labels"]
    trial_indices = ds["trial_indices"]
    start_times = ds["start_times"]
    end_times = ds["end_times"]
    channel_labels = ds["channel_labels"]
    fs = int(ds["fs"][0])

    print(f"Dataset loaded: {eeg.shape}")
    print(f"Subjects: {np.unique(subjects)}")
    print(f"Groups: {np.unique(groups)}")
    print(f"Labels: {np.unique(labels)}")
    print(f"Channels: {channel_labels}")
    print(f"Sampling rate: {fs} Hz")

    # --- Plot 1: Side-by-side comparison of random trials from .npz vs .mat ---
    n_comparisons = 3
    rng = np.random.default_rng(42)
    random_indices = rng.choice(len(eeg), size=n_comparisons, replace=False)

    fig, axes = plt.subplots(n_comparisons, 2, figsize=(16, 4 * n_comparisons))
    fig.suptitle("Verification: .npz (left) vs .mat (right)", fontsize=14)

    for row, idx in enumerate(random_indices):
        subj = str(subjects[idx])
        grp = int(groups[idx])
        trial_i = int(trial_indices[idx])

        # From .npz
        eeg_npz = eeg[idx]

        # From .mat
        eeg_mat = load_trial_from_mat(subj, grp, trial_i)

        # Pick channel 0 for visualization
        ch = 0
        time_axis = np.arange(eeg_npz.shape[1]) / fs

        axes[row, 0].plot(time_axis, eeg_npz[ch], linewidth=0.5, color="C0")
        axes[row, 0].set_title(f".npz — {subj}, group {grp}, trial {trial_i}, ch {channel_labels[ch]}")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Amplitude")

        axes[row, 1].plot(time_axis, eeg_mat[ch], linewidth=0.5, color="C1")
        axes[row, 1].set_title(f".mat — {subj}, group {grp}, trial {trial_i}, ch {channel_labels[ch]}")
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Amplitude")

        # Verify they are identical
        match = np.allclose(eeg_npz, eeg_mat)
        axes[row, 0].text(0.02, 0.95, f"Match: {match}", transform=axes[row, 0].transAxes,
                          fontsize=10, verticalalignment='top',
                          color="green" if match else "red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trial_comparison.png", dpi=150)
    plt.close()
    print(f"Saved trial_comparison.png")

    # --- Plot 2: Trial count per subject and group ---
    unique_subjects = sorted(np.unique(subjects))
    unique_groups = sorted(np.unique(groups))

    counts = np.zeros((len(unique_subjects), len(unique_groups)), dtype=int)
    for i, subj in enumerate(unique_subjects):
        for j, grp in enumerate(unique_groups):
            counts[i, j] = np.sum((subjects == subj) & (groups == grp))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(unique_subjects))
    width = 0.12
    for j, grp in enumerate(unique_groups):
        ax.bar(x + j * width, counts[:, j], width, label=f"Group {grp}")

    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of trials")
    ax.set_title("Trials per subject and group")
    ax.set_xticks(x + width * (len(unique_groups) - 1) / 2)
    ax.set_xticklabels(unique_subjects, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trials_per_subject_group.png", dpi=150)
    plt.close()
    print(f"Saved trials_per_subject_group.png")

    # --- Plot 3: Mean signal power per group ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Power by label
    for lbl in [0, 1]:
        mask = labels == lbl
        mean_power = np.mean(eeg[mask] ** 2, axis=(1, 2))
        lbl_name = "Pre-injection (0)" if lbl == 0 else "Post-injection (1)"
        axes[0].hist(mean_power, bins=50, alpha=0.6, label=lbl_name)

    axes[0].set_xlabel("Mean signal power")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Signal power distribution by label")
    axes[0].legend()

    # Mean EEG across all channels for one trial per group
    for grp in unique_groups:
        mask = groups == grp
        idx_grp = np.where(mask)[0][0]
        mean_signal = np.mean(eeg[idx_grp], axis=0)
        time_axis = np.arange(len(mean_signal)) / fs
        axes[1].plot(time_axis, mean_signal, label=f"Group {grp}", alpha=0.7)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mean amplitude across channels")
    axes[1].set_title("Example trial mean signal per group")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signal_analysis.png", dpi=150)
    plt.close()
    print(f"Saved signal_analysis.png")

    # --- Numerical verification ---
    print("\n--- Numerical Verification ---")
    n_checks = min(20, len(eeg))
    check_indices = rng.choice(len(eeg), size=n_checks, replace=False)
    all_match = True
    for idx in check_indices:
        subj = str(subjects[idx])
        grp = int(groups[idx])
        trial_i = int(trial_indices[idx])
        eeg_mat = load_trial_from_mat(subj, grp, trial_i)
        if not np.allclose(eeg[idx], eeg_mat):
            print(f"  MISMATCH: idx={idx}, {subj}, group={grp}, trial={trial_i}")
            all_match = False

    if all_match:
        print(f"  All {n_checks} random checks PASSED — .npz matches .mat exactly")
    else:
        print(f"  Some checks FAILED!")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
