"""
Build NPZ dataset for DMT plasma regression using a per-subject bi-exponential
fit as the label source (alternative to the PK-model trace approach in
reg_graphTrip/build_dmt_dataset.py).

For each DMT subject we fit the generalised 2-exponential form
    C(t) = A * exp(-alpha * t) + B * exp(-beta * t)
to the 4 post-injection plasma points in data/plasma_clean.csv (time relative
to injection, condition=="dmt", time_point>=1), with B>=0, alpha>beta>0, and
A free to be negative. This admits both monotone-decay (A,B>=0) and
Bateman-style peaked (A<0, B>0) trajectories

The schema of the output .npz matches data/eeg_dmt_regression.npz so the
consumer (EEGDataset) is unchanged; an extra `biexp_params` array of shape
(n_subjects, 5) = [A, alpha, B, beta, ssr] is added for traceability.

A single subjects-overview figure is also saved next to the .npz.

Usage (run from project root):
    python -m src.models.reg_simpleCNN.build_biexp_dataset \
        [--t-min 2] [--t-max 15] [--out data/eeg_dmt_regression_biexp.npz]
"""

from scipy.optimize import curve_fit
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"

SUBJECT_FOLDERS = {
    "S01AS": "S01", "S02WT": "S02", "S03BS": "S03", "S04SG": "S04",
    "S05LM": "S05", "S06ET": "S06", "S07CS": "S07", "S08EK": "S08",
    "S09BB": "S09", "S10DL": "S10", "S11NW": "S11", "S12AI": "S12",
    "S13MBJ": "S13",
}

EXCLUDED_SUBJECTS = ["S03", "S04", "S08", "S09", "S11"]

EEG_CHANNELS = list(range(31)) + [35]  # 32 pure EEG channels
EEG_CHANNEL_LABELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
    "TP9", "TP10", "POz", "FCz",
]


def biexp(t, A, alpha, B, beta):
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

# we do this for each subject


def fit_biexp(t_obs, c_obs):
    """Per-subject 2-exponential fit with α>β>0, B>=0, A free, C(t)>=0 on obs range.
    Multistart over (α, β, sign(A)) so both monotone-decay (A>=0) and Bateman-
    style peaked (A<0) shapes are explored. Returns (params, ssr) where
    params = (A, alpha, B, beta).
    """
    t_obs = np.asarray(t_obs, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    c_max = float(np.max(c_obs))
    U = 10.0 * c_max + 1.0

    # (alpha, beta) seeds in 1/min: fast/slow rates typical for DMT PK.
    # intuition: at any given moment, the concentration is decreasing at a rate of 2x the current value per minute (see derivatives)
    alpha_seeds = [0.3, 1.0, 2.0, 5.0]  # distribution phase (fast)
    beta_seeds = [0.02, 0.05, 0.1, 0.2]  # elimination phase (slow)

    # Bounds: A free in [-U, U], B non-negative. alpha/beta bounds overlap but
    # the α>β admissibility check below rejects non-canonical solutions.
    # (A, α, B, β)
    bounds_lo = [-U, 1e-4, 0.0, 1e-5]
    bounds_hi = [U, 50.0, U, 5.0]

    # Grid over observation window .
    t_check = np.linspace(float(t_obs.min()), float(t_obs.max()), 64)
    neg_tol = -1e-3 * max(c_max, 1.0)  # negative tolerance

    best = None
    for a_s in alpha_seeds:
        for b_s in beta_seeds:
            if a_s <= b_s:
                continue
            for A_sign in (+1.0, -1.0):
                p0 = [A_sign * c_max, a_s, c_max * 0.5, b_s]
                try:
                    popt, _ = curve_fit(
                        biexp, t_obs, c_obs,
                        p0=p0,
                        bounds=(bounds_lo, bounds_hi),
                        maxfev=20000,
                    )
                except (RuntimeError, ValueError):
                    continue
                A, alpha, B, beta = popt
                # check alpha is at least 1% bigger
                if alpha <= beta * 1.01:
                    continue
                # Non-negativity over the observation window.
                if np.any(biexp(t_check, A, alpha, B, beta) < neg_tol):
                    continue
                resid = c_obs - biexp(t_obs, A, alpha, B, beta)
                ssr = float(np.sum(resid ** 2))  # sum of the square residuals
                if best is None or ssr < best[1]:
                    best = ((A, alpha, B, beta), ssr)

    if best is None:
        raise RuntimeError("bi-exponential fit failed for all multistart seeds")
    return best


def process_subject_eeg(folder_name, subj_id, t_min, t_max):
    """Load EEG trials for one subject, filter by time. Returns eeg, times, or None."""
    mat_path = RECORDINGS_DIR / folder_name / "DMT" / "data_trialsmxm_3s.mat"
    if not mat_path.exists():
        print(f"  Skipping {subj_id}: {mat_path} not found")
        return None

    d = sio.loadmat(str(mat_path), squeeze_me=True)
    cells = d["data_trialsmxm_3s"]
    n_baseline = len(cells[0][()]["trial"])
    injection_time = n_baseline * 3.0 / 60.0

    eeg_trials, trial_times = [], []
    trial_idx = 0
    for cell_idx in range(len(cells)):
        cell = cells[cell_idx][()]
        trials = cell["trial"]
        for j in range(len(trials)):
            t_since_inj = (trial_idx + 0.5) * 3.0 / 60.0 - injection_time
            if t_min <= t_since_inj <= t_max:
                eeg_trials.append(trials[j][EEG_CHANNELS, :])
                trial_times.append(t_since_inj)
            trial_idx += 1

    if not eeg_trials:
        print(f"  {subj_id}: no trials in [{t_min}, {t_max}] min")
        return None

    return np.stack(eeg_trials), np.array(trial_times)


def load_plasma_post_inj():
    """Return {subj_id: (t_arr, c_arr)} for post-injection DMT points."""
    pdf = pd.read_csv(DATA_DIR / "plasma_clean.csv")
    pdf = pdf[(pdf["condition"] == "dmt") & (pdf["time_point"] >= 1)]
    pdf = pdf[~pdf["subject"].isin(EXCLUDED_SUBJECTS)]
    out = {}
    for subj, sdf in pdf.groupby("subject"):
        sdf = sdf.sort_values("time_min")
        out[subj] = (
            sdf["time_min"].to_numpy(dtype=float),
            sdf["plasma_conc"].to_numpy(dtype=float),
        )
    return out


def plot_subject_fits(fits, t_min, t_max, out_path):
    """2xN grid (one panel per subject) with observed plasma + fitted curve."""
    subjects = sorted(fits.keys())
    n = len(subjects)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)

    t_dense = np.linspace(0.0, 22.0, 400)
    for ax, subj in zip(axes.flat, subjects):
        t_obs, c_obs, params, ssr = fits[subj]
        A, alpha, B, beta = params
        c_fit = biexp(t_dense, A, alpha, B, beta)
        ax.plot(t_dense, c_fit, color="darkorange",
                label=f"bi-exp fit\nα={alpha:.2f}/min  β={beta:.3f}/min")
        ax.scatter(t_obs, c_obs, color="steelblue", s=35, zorder=3,
                   label="observed")
        ax.axvspan(t_min, t_max, color="gray", alpha=0.12,
                   label=f"label window [{t_min},{t_max}] min")
        ax.set_xlim(0, 22)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time post-injection (min)")
        ax.set_ylabel("plasma DMT (ng/mL)")
        ax.set_title(f"{subj}  SSR={ssr:.1f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("Per-subject bi-exponential fits to post-injection plasma DMT",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved subject-fits plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build bi-exponential-label DMT regression dataset"
    )
    parser.add_argument("--t-min", type=float, default=2.0,
                        help="Start time in minutes post-injection (default: 2)")
    parser.add_argument("--t-max", type=float, default=15.0,
                        help="End time in minutes post-injection (default: 15)")
    parser.add_argument("--out", type=str,
                        default=str(DATA_DIR / "eeg_dmt_regression_biexp.npz"),
                        help="Output .npz path")
    args = parser.parse_args()

    plasma = load_plasma_post_inj()

    # --- per-subject fits ---
    fits = {}
    print("Fitting bi-exponential to post-injection plasma points...")
    for subj, (t_obs, c_obs) in sorted(plasma.items()):
        params, ssr = fit_biexp(t_obs, c_obs)
        A, alpha, B, beta = params
        fits[subj] = (t_obs, c_obs, params, ssr)
        print(f"  {subj}: A={A:8.2f}  alpha={alpha:6.3f}/min  "
              f"B={B:8.2f}  beta={beta:6.3f}/min  SSR={ssr:8.2f}")

    # --- EEG + label assembly ---
    all_eeg, all_labels, all_times, all_subjects = [], [], [], []
    params_by_order = []

    print("\nAssembling EEG trials and evaluating labels...")
    for folder_name, subj_id in sorted(SUBJECT_FOLDERS.items()):
        if subj_id in EXCLUDED_SUBJECTS or subj_id not in fits:
            continue
        print(f"Processing {subj_id}...")
        res = process_subject_eeg(folder_name, subj_id, args.t_min, args.t_max)
        if res is None:
            continue
        eeg_array, times_array = res

        _, _, params, ssr = fits[subj_id]
        A, alpha, B, beta = params
        labels = biexp(times_array, A, alpha, B, beta)

        n_neg = int((labels < 0).sum())
        if n_neg > 0:
            print(f"  WARN {subj_id}: {n_neg} labels < 0 (min={labels.min():.3f}); "
                  f"using raw fit values as requested")

        print(f"  {subj_id}: {len(times_array)} trials  "
              f"t=[{times_array.min():.2f},{times_array.max():.2f}] min  "
              f"label=[{labels.min():.2f},{labels.max():.2f}] ng/mL")

        all_eeg.append(eeg_array)
        all_labels.append(labels.astype(np.float32))
        all_times.append(times_array.astype(np.float32))
        all_subjects.append(np.array([subj_id] * len(times_array)))
        params_by_order.append((subj_id, A, alpha, B, beta, ssr))

    eeg_data = np.concatenate(all_eeg, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    times_all = np.concatenate(all_times, axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    biexp_params_arr = np.array(
        [[A, alpha, B, beta, ssr] for (_, A, alpha, B, beta, ssr) in params_by_order],
        dtype=np.float64,
    )
    biexp_param_subjects = np.array(
        [s for (s, *_rest) in params_by_order]
    )

    out_path = Path(args.out)
    np.savez(
        out_path,
        eeg_data=eeg_data,
        labels=labels_all,
        times=times_all,
        subjects=subjects_all,
        channel_labels=np.array(EEG_CHANNEL_LABELS),
        biexp_params=biexp_params_arr,
        biexp_param_subjects=biexp_param_subjects,
        biexp_param_cols=np.array(["A", "alpha", "B", "beta", "ssr"]),
    )
    print(f"\nSaved to {out_path}")
    print(f"  Total trials: {len(labels_all)}")
    print(f"  Subjects: {sorted(set(subjects_all.tolist()))}")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Label range: [{labels_all.min():.2f}, {labels_all.max():.2f}] ng/mL")
    print(f"  Time range: [{times_all.min():.2f}, {times_all.max():.2f}] min")

    plot_path = out_path.with_suffix("").with_name(out_path.stem + "_fits.png")
    plot_subject_fits(fits, args.t_min, args.t_max, plot_path)


if __name__ == "__main__":
    main()
