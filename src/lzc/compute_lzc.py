"""
Compute Lempel-Ziv Complexity (LZ76) from EEG data under DMT administration.

Pipeline:
1. Load data_trialsmxm_3s.mat for each subject's DMT session
   - Cell 0: baseline (pre-infusion)
   - Cells 1-3: post-infusion segments
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


def lz76(binary_array):
    """Compute LZ76 complexity of a binary sequence, normalized by n/log2(n)."""
    n = len(binary_array)
    if n <= 1:
        return 0.0
    s = binary_array.tobytes()
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
    return len(sub_strings) / (n / np.log2(n))


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

    Cell 0 = baseline, Cells 1-3 = post-injection.
    Time axis uses sequential trial indexing (3s per trial), which removes
    the gap between baseline and post-injection recordings.

    Returns
    -------
    df : DataFrame with columns ['time_min', 'lzc_normalized', 'injection_time_min']
    """
    print(f"  Loading {mat_path}...")
    d = sio.loadmat(mat_path, squeeze_me=True)
    cells = d["data_trialsmxm_3s"]
    ds_factor = 5  # 1000Hz -> 200Hz

    # --- Baseline (cells 0 + 1) ---
    # Cell 0: pre-injection baseline
    baseline_lzc = []
    for bl_cell_idx in range(1):
        for trial in cells[bl_cell_idx][()]["trial"]:
            trial_ds = decimate(trial, ds_factor, axis=1)  # resampling
            baseline_lzc.append(compute_lzc_trial(trial_ds))
    baseline_lzc = np.array(baseline_lzc)
    baseline_mean = np.mean(baseline_lzc)  # do the mean of all 3s lzc from baseline
    n_baseline = len(cells[0][()]["trial"])  # sequential time offset = cell 0 only
    print(f"  Baseline (cells 0): {len(baseline_lzc)} trials, mean LZc = {baseline_mean:.4f}")

    # --- All cells: sequential trial time ---
    all_lzc = []
    all_times = []
    trial_idx = 0

    for cell_idx in range(len(cells)):
        cell = cells[cell_idx][()]
        trials = cell["trial"]

        for j in range(len(trials)):
            trial_ds = decimate(trials[j], ds_factor, axis=1)
            lzc_val = compute_lzc_trial(trial_ds)
            # Sequential time: each trial is 3 seconds
            t_min = (trial_idx + 0.5) * 3.0 / 60.0  # midpoint of trial
            all_lzc.append(lzc_val)
            all_times.append(t_min)
            trial_idx += 1

    all_lzc = np.array(all_lzc)
    all_times = np.array(all_times)

    # Normalize: percentage change relative to baseline mean
    lzc_normalized = (all_lzc - baseline_mean) / baseline_mean * 100

    # Injection time on the sequential scale (after all baseline trials)
    injection_time_min = n_baseline * 3.0 / 60.0

    print(f"  Total: {len(all_lzc)} trials, injection at {injection_time_min:.1f} min, "
          f"LZc range [{lzc_normalized.min():.1f}, {lzc_normalized.max():.1f}]")

    df = pd.DataFrame({
        "time_min": all_times,
        "lzc_raw": all_lzc,
        "lzc_normalized": lzc_normalized,
        "baseline_mean": baseline_mean,
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

        print(f"\nSaved results to {out_path}")
        print(f"Subjects processed: {list(all_results.keys())}")
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
