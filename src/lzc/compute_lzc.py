"""
Compute Lempel-Ziv Complexity (LZ76) from EEG data under DMT administration.

Pipeline:
1. Load data_trialsmxm_3s.mat for each subject's DMT session
   - Cell 0: baseline (pre-infusion)
   - Cells 1-20: post-infusion segments
2. Downsample each 3-second trial from 1000Hz to 200Hz
3. Compute LZ76 per channel per trial, average across channels
4. Normalize by baseline (cell 0) mean LZc
5. Save results to CSV
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import decimate
from pathlib import Path


# Use all 36 channels as described in the paper (section 4.2)
ALL_CHANNELS = list(range(36))

SUBJECT_FOLDERS = {
    "S01AS": "S01",
    "S02WT": "S02",
    "S03BS": "S03",
    "S04SG": "S04",
    "S05LM": "S05",
    "S06ET": "S06",
    "S07CS": "S07",
    "S08EK": "S08",
    "S09BB": "S09",
    "S10DL": "S10",
    "S11NW": "S11",
    "S12AI": "S12",
    "S13MBJ": "S13",
}


def _lz76_count(s, n):
    """Count LZ76 distinct substrings from a bytes object of length n."""
    sub_strings = set()
    ind = 0
    inc = 1
    while ind + inc <= n:
        sub_str = s[ind:ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)


def lz76(binary_array):
    """Compute normalized LZ76 complexity using surrogate normalization.

    Divides by the complexity of a shuffled version of the same sequence
    (Casali et al. 2013), guaranteeing values in [0, 1].
    """
    n = len(binary_array)
    if n <= 1:
        return 0.0
    if len(np.unique(binary_array)) <= 1:
        return 0.0
    s = binary_array.tobytes()
    c = _lz76_count(s, n)
    # Surrogate: shuffle to destroy temporal structure (max complexity)
    shuffled = np.random.permutation(binary_array).astype(np.uint8)
    c_surr = _lz76_count(shuffled.tobytes(), n)
    if c_surr == 0:
        return 0.0
    return min(c / c_surr, 1.0)


def compute_lzc_trial(eeg_trial):
    """
    Compute LZc for one 3-second trial of multi-channel EEG.

    Parameters
    ----------
    eeg_trial : ndarray, shape (n_channels, n_samples)

    Returns
    -------
    float : mean LZc across EEG channels
    """
    eeg = eeg_trial[ALL_CHANNELS, :]
    lzc_values = np.empty(len(ALL_CHANNELS))
    for i, ch in enumerate(range(len(ALL_CHANNELS))):
        signal = eeg[ch]
        # binarize the signal with respect to the mean (as require by lz76)
        binary = (signal >= np.mean(signal)).astype(np.uint8)
        lzc_values[i] = lz76(binary)
    return np.mean(lzc_values)


def process_subject(mat_path):
    """
    Full LZc pipeline for one subject using data_trialsmxm_3s.mat.

    Cell 0 = baseline, Cells 1+ = post-injection.
    Time axis uses real elapsed time derived from sampleinfo (sample indices
    at 1000 Hz), preserving gaps between recording segments.

    Returns
    -------
    df : DataFrame with columns ['time_min', 'lzc_raw']
    """
    print(f"  Loading {mat_path}...")
    d = sio.loadmat(mat_path, squeeze_me=True)
    cells = d["data_trialsmxm_3s"]
    ds_factor = 5  # 1000Hz -> 200Hz
    fs = 1000.0  # sampling rate in Hz

    # Injection time = start of first post-injection trial (cell 1)
    injection_sample = cells[1][()]["sampleinfo"][0, 0]

    all_lzc = []
    all_times = []

    for cell_idx in range(len(cells)):
        cell = cells[cell_idx][()]
        trials = cell["trial"]
        sampleinfo = cell["sampleinfo"]

        for j in range(len(trials)):
            trial_ds = decimate(trials[j], ds_factor, axis=1)
            lzc_val = compute_lzc_trial(trial_ds)
            # Real time: midpoint of trial from sampleinfo
            midpoint_sample = (sampleinfo[j, 0] + sampleinfo[j, 1]) / 2.0
            t_min = midpoint_sample / fs / 60.0
            all_lzc.append(lzc_val)
            all_times.append(t_min)

    all_lzc = np.array(all_lzc)
    all_times = np.array(all_times)

    injection_time_min = injection_sample / fs / 60.0

    print(f"  Total: {len(all_lzc)} trials, injection at {injection_time_min:.1f} min, "
          f"LZc range [{all_lzc.min():.4f}, {all_lzc.max():.4f}]")

    df = pd.DataFrame({
        "time_min": all_times,
        "lzc_raw": all_lzc,
    })
    df.attrs["injection_time_min"] = injection_time_min
    return df, injection_time_min


def main():
    project_root = Path(__file__).resolve().parents[2]
    recordings_dir = project_root / "data" / "recordings"
    results_dir = project_root / "results" / "lzc"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    injection_offsets = {}

    for folder_name, subj_id in sorted(SUBJECT_FOLDERS.items()):
        mat_path = recordings_dir / folder_name / "DMT" / "data_trialsmxm_3s.mat"
        if not mat_path.exists():
            print(f"Skipping {subj_id}: {mat_path} not found")
            continue

        print(f"\nProcessing {subj_id} ({folder_name})...")
        try:
            df, inj_time = process_subject(str(mat_path))
            df["subject"] = subj_id
            all_results[subj_id] = df
            injection_offsets[subj_id] = inj_time
            print(f"  Done: {len(df)} LZc values")
        except Exception as e:
            print(f"  ERROR processing {subj_id}: {e}")
            continue

    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        out_path = results_dir / "lzc_results.csv"
        combined.to_csv(out_path, index=False)

        # Save injection time offsets for plotting plasma alignment
        offsets_df = pd.DataFrame(
            list(injection_offsets.items()),
            columns=["subject", "injection_time_min"],
        )
        offsets_df.to_csv(results_dir / "injection_offsets.csv", index=False)

        # Plot LZc values per subject
        import matplotlib.pyplot as plt

        subjects = sorted(combined["subject"].unique())
        n_cols = 3
        n_rows = (len(subjects) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        for i, subj in enumerate(subjects):
            ax = axes[i]
            subj_df = combined[combined["subject"] == subj]
            inj_t = injection_offsets.get(subj, None)
            ax.scatter(subj_df["time_min"], subj_df["lzc_raw"],
                       s=4, alpha=0.5, color="steelblue")
            if inj_t is not None:
                ax.axvline(inj_t, color="red", linestyle="--", linewidth=1)
            ax.set_title(subj)
            ax.set_ylim(0.55, 0.9)

        for j in range(len(subjects), len(axes)):
            axes[j].set_visible(False)
        for ax in axes[n_cols * (n_rows - 1):]:
            if ax.get_visible():
                ax.set_xlabel("Time (min)")
        for r in range(n_rows):
            axes[r * n_cols].set_ylabel("LZc")

        fig.suptitle("Raw LZc per subject (red = injection time)", fontsize=14, y=1.0)
        plt.tight_layout()
        plot_path = results_dir / "lzc_raw_all_subjects.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved LZc plot to {plot_path}")

        print(f"\nSaved results to {out_path}")
        print(f"Subjects processed: {list(all_results.keys())}")
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
