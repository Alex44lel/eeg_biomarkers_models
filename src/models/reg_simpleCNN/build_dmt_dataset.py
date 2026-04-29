"""
Build NPZ dataset for DMT plasma concentration regression.

For each subject, loads 3-second EEG trials from data_trialsmxm_3s.mat,
filters to [2, 15] minutes post-injection, and assigns labels from the
PK model's posterior mean y1 curve (plasma DMT in ng/mL).

When --include-baseline is set, also collects ALL pre-injection trials
(t_since_inj < 0) and stores them alongside post-injection trials with
an is_baseline boolean flag and NaN labels. This is used by the linear
subject-adaptation experiment in reg_simpleCNN/{model,train_cv}.py.

Usage (run from project root):
    # Default — post-injection only, output bit-identical to legacy build:
    python -m src.models.reg_simpleCNN.build_dmt_dataset \
        [--trace PATH] [--t-min 2] [--t-max 15]

    # With baseline trials:
    python -m src.models.reg_simpleCNN.build_dmt_dataset \
        --include-baseline \
        --out-name eeg_dmt_regression_with_baseline.npz
"""

import argparse
import numpy as np
import scipy.io as sio
import arviz as az
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
DATA_DIR = PROJECT_ROOT / "data"

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

EXCLUDED_SUBJECTS = ["S03", "S04", "S08", "S09", "S11"]

# EEG channels to keep (exclude ECG, VEOG, EMGfront, EMGtemp at indices 31-34)
EEG_CHANNELS = list(range(31)) + [35]  # 32 EEG channels

# Channel labels for the 32 EEG channels (from the original 36-channel montage)
EEG_CHANNEL_LABELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
    "TP9", "TP10", "POz", "FCz",
]


def compute_y1_curve(trace, subject_idx, t_array):
    """Compute the posterior-mean y1 curve (plasma ng/mL) for one subject.

    Evaluates eq. 16 per posterior sample and then averages — matches
    partial_pooling_model.compute_posterior_predictive_y1. Plugging in the
    posterior means of the parameters first would bias the curve (Jensen's
    inequality: y1 is non-linear in k0, k1, k2, y_init).
    """
    n_subjects = trace.posterior["y_init"].shape[-1]
    k0 = trace.posterior["k0"].values.reshape(-1, n_subjects)[:, subject_idx]       # (S,)
    k1 = trace.posterior["k1"].values.reshape(-1, n_subjects)[:, subject_idx]
    k2 = trace.posterior["k2"].values.reshape(-1, n_subjects)[:, subject_idx]
    y_init = trace.posterior["y_init"].values.reshape(-1, n_subjects)[:, subject_idx]

    s = k0 + k1 + k2
    p = k0 * k2
    disc = np.maximum(s**2 - 4 * p, 1e-12)
    alpha = (s - np.sqrt(disc)) / 2.0       # (S,)
    beta = (s + np.sqrt(disc)) / 2.0        # (S,)

    # Broadcast (S, 1) against (T,) to get per-sample curves of shape (S, T)
    a = alpha[:, None]
    b = beta[:, None]
    k2_ = k2[:, None]
    y0 = y_init[:, None]
    t = t_array[None, :]

    y1_samples = (y0 / (b - a)) * ((k2_ - a) * np.exp(-a * t) - (k2_ - b) * np.exp(-b * t))
    return y1_samples.mean(axis=0)


def process_subject(folder_name, subj_id, trace, pk_subject_idx, t_min, t_max,
                    include_baseline=False):
    """Load EEG trials for one subject, filter by time, assign PK labels.

    Post-injection trials (t_since_inj in [t_min, t_max]) get plasma labels
    from the PK model. When include_baseline is True, ALL pre-injection trials
    (t_since_inj < 0) are also collected with NaN labels and is_baseline=True.
    """
    mat_path = RECORDINGS_DIR / folder_name / "DMT" / "data_trialsmxm_3s.mat"
    if not mat_path.exists():
        print(f"  Skipping {subj_id}: {mat_path} not found")
        return None

    d = sio.loadmat(str(mat_path), squeeze_me=True)
    cells = d["data_trialsmxm_3s"]

    # Count baseline trials to determine injection time
    n_baseline = len(cells[0][()]["trial"])
    injection_time = n_baseline * 3.0 / 60.0

    # Post-injection trials (kept by [t_min, t_max] window)
    post_eeg = []
    post_times = []
    # Pre-injection trials (t_since_inj < 0); only collected when requested
    base_eeg = []
    base_times = []
    trial_idx = 0

    for cell_idx in range(len(cells)):
        cell = cells[cell_idx][()]
        trials = cell["trial"]
        for j in range(len(trials)):
            t_since_inj = (trial_idx + 0.5) * 3.0 / 60.0 - injection_time
            if t_min <= t_since_inj <= t_max:
                post_eeg.append(trials[j][EEG_CHANNELS, :])  # (32, 3000)
                post_times.append(t_since_inj)
            elif include_baseline and t_since_inj < 0:
                base_eeg.append(trials[j][EEG_CHANNELS, :])
                base_times.append(t_since_inj)
            trial_idx += 1

    if not post_eeg:
        print(f"  {subj_id}: no trials in [{t_min}, {t_max}] min")
        return None

    post_eeg_arr = np.stack(post_eeg)
    post_times_arr = np.array(post_times)
    post_labels = compute_y1_curve(trace, pk_subject_idx, post_times_arr)

    msg = (f"  {subj_id}: {len(post_eeg)} post-inj trials, "
           f"t=[{post_times_arr.min():.2f}, {post_times_arr.max():.2f}] min, "
           f"plasma=[{post_labels.min():.1f}, {post_labels.max():.1f}] ng/mL")

    if include_baseline:
        if not base_eeg:
            print(msg + "  |  WARNING: no pre-injection trials found")
            return None
        base_eeg_arr = np.stack(base_eeg)
        base_times_arr = np.array(base_times)
        base_labels = np.full(len(base_eeg), np.nan, dtype=np.float64)

        eeg_array = np.concatenate([base_eeg_arr, post_eeg_arr], axis=0)
        times_array = np.concatenate([base_times_arr, post_times_arr], axis=0)
        labels_array = np.concatenate([base_labels, post_labels], axis=0)
        is_baseline = np.concatenate([
            np.ones(len(base_eeg), dtype=bool),
            np.zeros(len(post_eeg), dtype=bool),
        ])

        print(msg + f"  |  + {len(base_eeg)} baseline trials, "
                    f"t=[{base_times_arr.min():.2f}, {base_times_arr.max():.2f}] min")
    else:
        eeg_array = post_eeg_arr
        times_array = post_times_arr
        labels_array = post_labels
        is_baseline = None  # signal "do not save column"
        print(msg)

    return {
        "eeg": eeg_array,
        "labels": labels_array,
        "times": times_array,
        "subject": np.array([subj_id] * len(eeg_array)),
        "is_baseline": is_baseline,
    }


def main():
    parser = argparse.ArgumentParser(description="Build DMT regression dataset")
    parser.add_argument("--trace", type=str,
                        default="results/paper_model/only_plasma/partial_pooling_trace.nc",
                        help="Path to PK model trace")
    parser.add_argument("--t-min", type=float, default=2.0,
                        help="Start time in minutes post-injection (default: 2)")
    parser.add_argument("--t-max", type=float, default=15.0,
                        help="End time in minutes post-injection (default: 15)")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Also collect all pre-injection trials (t<0) with NaN "
                             "labels and is_baseline=True. Used by the linear "
                             "subject-adaptation experiment.")
    parser.add_argument("--out-name", type=str, default="eeg_dmt_regression.npz",
                        help="Output filename inside data/ (default: eeg_dmt_regression.npz)")
    args = parser.parse_args()

    # Load PK model trace and get subject list
    print(f"Loading PK trace from {args.trace}...")
    trace = az.from_netcdf(args.trace)

    # Get the subject names from prepare_data (same order as the trace)
    from src.paper_model.prepare_data import load_and_prepare
    pk_data = load_and_prepare()
    pk_subjects = pk_data["subject_names"]
    pk_subj_to_idx = {s: i for i, s in enumerate(pk_subjects)}
    print(f"PK model subjects: {pk_subjects}")

    # Process each subject
    all_eeg = []
    all_labels = []
    all_times = []
    all_subjects = []
    all_is_baseline = []

    for folder_name, subj_id in sorted(SUBJECT_FOLDERS.items()):
        if subj_id in EXCLUDED_SUBJECTS:
            continue
        if subj_id not in pk_subj_to_idx:
            print(f"  Skipping {subj_id}: not in PK model")
            continue

        print(f"Processing {subj_id}...")
        result = process_subject(
            folder_name, subj_id, trace,
            pk_subj_to_idx[subj_id], args.t_min, args.t_max,
            include_baseline=args.include_baseline,
        )
        if result is not None:
            all_eeg.append(result["eeg"])
            all_labels.append(result["labels"])
            all_times.append(result["times"])
            all_subjects.append(result["subject"])
            if result["is_baseline"] is not None:
                all_is_baseline.append(result["is_baseline"])

    # Concatenate and save
    eeg_data = np.concatenate(all_eeg, axis=0)     # (N, 32, 3000)
    labels = np.concatenate(all_labels, axis=0)     # (N,) ng/mL (NaN for baseline)
    times = np.concatenate(all_times, axis=0)       # (N,) minutes
    subjects = np.concatenate(all_subjects, axis=0)  # (N,) subject IDs

    save_kwargs = dict(
        eeg_data=eeg_data,
        labels=labels,
        times=times,
        subjects=subjects,
        channel_labels=np.array(EEG_CHANNEL_LABELS),
    )

    if args.include_baseline:
        # All per-subject results carried an is_baseline mask in this branch
        save_kwargs["is_baseline"] = np.concatenate(all_is_baseline, axis=0)

    out_path = DATA_DIR / args.out_name
    np.savez(out_path, **save_kwargs)

    print(f"\nSaved to {out_path}")
    print(f"  Total trials: {len(labels)}")
    print(f"  Subjects: {sorted(set(subjects))}")
    print(f"  EEG shape: {eeg_data.shape}")
    if args.include_baseline:
        is_base = save_kwargs["is_baseline"]
        n_base = int(is_base.sum())
        n_post = int((~is_base).sum())
        post_labels = labels[~is_base]
        print(f"  Baseline trials: {n_base}  Post-inj trials: {n_post}")
        print(f"  Post-inj label range: [{post_labels.min():.1f}, {post_labels.max():.1f}] ng/mL")
    else:
        print(f"  Label range: [{labels.min():.1f}, {labels.max():.1f}] ng/mL")
    print(f"  Time range: [{times.min():.2f}, {times.max():.2f}] min")


if __name__ == "__main__":
    main()
