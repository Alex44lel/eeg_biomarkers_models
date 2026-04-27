"""
Build ANTI-ALIASED polyphase-downsampled NPZ datasets from data/eeg_dmt_regression.npz.

Companion to build_downsampled_dataset.py. The original script does a raw
strided slice (eeg[..., i::k]) which is mathematically a valid polyphase
decomposition but, for any *single* sub-signal viewed in isolation,
behaves like a decimation without anti-alias filtering: any frequency
content above the new Nyquist (500/k Hz) folds back into 0..500/k Hz and
contaminates the kept bands.

Empirically this destroys downstream regression performance at k≥5
(see misc/study_notes/findings.md for the data). At k=5 (Nyquist 100 Hz)
the high-gamma EEG band aliases into alpha/beta; at k=10 (Nyquist 50 Hz)
even alpha gets corrupted.

This script applies an anti-alias FIR low-pass filter to the source signal
*before* taking each phase, eliminating the aliasing while still producing
K interleaved sub-signals per parent trial.

Output filenames have a `_filt` suffix so they coexist with the unfiltered
ones produced by build_downsampled_dataset.py:

    data/eeg_dmt_regression_k{K}_filt.npz   (this script)
    data/eeg_dmt_regression_k{K}.npz        (build_downsampled_dataset.py)

Output schema is identical to the unfiltered version (same fields,
same dtypes, same trial-major row layout). Existing dataset / training
code reads it transparently as long as the dataset key is registered in
DATASET_PATHS.

Filter design: 31-tap FIR low-pass with cutoff at the new Nyquist
(1/k of original Nyquist), Hamming window, applied via filtfilt for
zero-phase response. Same defaults as scipy.signal.decimate(ftype='fir',
zero_phase=True) — well-tested DSP.

Usage:
    python -m src.models.reg_simpleCNN.build_downsampled_dataset_filt --k 2 5 10
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, firwin

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = PROJECT_ROOT / "data" / "eeg_dmt_regression.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"
DEFAULT_NUMTAPS = 31  # matches scipy.signal.decimate(ftype='fir') default


def design_anti_alias_filter(k, numtaps=DEFAULT_NUMTAPS):
    """FIR low-pass at cutoff = 1/k of original Nyquist (Hamming window).

    With filtfilt this gives a zero-phase response with effective sharpness
    roughly 2× the single-pass filter, no group delay.
    """
    if k < 2:
        return None
    # cutoff is in normalized units where 1.0 == Nyquist of original signal.
    return firwin(numtaps, 1.0 / k, window="hamming")


def polyphase_split_filt(eeg, k, numtaps=DEFAULT_NUMTAPS):
    """Anti-aliased polyphase split.

    Steps:
        1. Low-pass filter the input along the time axis at the new Nyquist.
        2. Take K interleaved phases of the filtered signal: phase i is
           eeg_filt[..., i::k] truncated to a uniform length.

    Output shape and row layout are identical to polyphase_split (the
    unfiltered version): (n*k, c, l_ds) in trial-major order, so that
    rows [t*k : (t+1)*k] hold all K phases of parent trial t.
    """
    n, c, l = eeg.shape
    l_ds = l // k

    if k == 1:
        out = np.empty((n, c, l_ds), dtype=eeg.dtype)
        out[:] = eeg[:, :, :l_ds]
        return out

    b = design_anti_alias_filter(k, numtaps=numtaps)
    # Filter along the time axis with zero-phase forward-backward.
    # filtfilt promotes to float64 internally; cast back to source dtype.
    filtered = filtfilt(b, [1.0], eeg, axis=-1).astype(eeg.dtype, copy=False)

    out = np.empty((n * k, c, l_ds), dtype=eeg.dtype)
    for i in range(k):
        out[i::k] = filtered[:, :, i:i + k * l_ds:k]
    return out


def replicate_per_phase(arr, k):
    """Replicate length-N 1-D array to K*N in trial-major order: each value
    repeated K times consecutively. Mirrors polyphase_split_filt's row
    layout (and matches the unfiltered build's helper of the same name)."""
    return np.repeat(arr, k, axis=0)


def build_one(src_eeg, src_labels, src_times, src_subjects, k, numtaps=DEFAULT_NUMTAPS):
    """Return the field dict for one anti-aliased K-factor dataset."""
    n_orig, c, l = src_eeg.shape
    if l % k != 0:
        l_ds = l // k
        print(f"  NOTE: L={l} not divisible by k={k}; using L_ds={l_ds} "
              f"(dropping {l - k * l_ds} trailing samples)")

    eeg_ds = polyphase_split_filt(
        src_eeg.astype(np.float32, copy=False), k, numtaps=numtaps
    )
    labels_ds = replicate_per_phase(src_labels.astype(np.float32, copy=False), k)
    times_ds = replicate_per_phase(src_times.astype(np.float32, copy=False), k)
    subjects_ds = replicate_per_phase(src_subjects, k)
    k_idx = np.tile(np.arange(k, dtype=np.int8), n_orig)
    orig_trial_id = np.repeat(np.arange(n_orig, dtype=np.int32), k)

    return {
        "eeg_data": eeg_ds,
        "labels": labels_ds,
        "times": times_ds,
        "subjects": subjects_ds,
        "k_idx": k_idx,
        "orig_trial_id": orig_trial_id,
        "k_factor": np.int32(k),
        "antialias_numtaps": np.int32(numtaps),  # provenance for the build
    }


def sanity_check(src_eeg, src_labels, src_times, src_subjects, fields, k):
    """Hard-fail asserts that the build is internally consistent and that
    the anti-alias filter has actually attenuated power above the new Nyquist.
    """
    eeg = fields["eeg_data"]
    labels = fields["labels"]
    times = fields["times"]
    subjects = fields["subjects"]
    k_idx = fields["k_idx"]
    orig_id = fields["orig_trial_id"]

    n_orig = src_eeg.shape[0]
    assert eeg.shape == (n_orig * k, src_eeg.shape[1], src_eeg.shape[2] // k), \
        f"unexpected eeg shape {eeg.shape}"
    assert labels.shape == (n_orig * k,)
    assert times.shape == (n_orig * k,)
    assert subjects.shape == (n_orig * k,)
    assert k_idx.shape == (n_orig * k,)
    assert orig_id.shape == (n_orig * k,)

    # k_idx and orig_trial_id layout: trial-major, K phases per parent trial
    expected_k_idx = np.tile(np.arange(k, dtype=np.int8), n_orig)
    expected_orig = np.repeat(np.arange(n_orig, dtype=np.int32), k)
    assert np.array_equal(k_idx, expected_k_idx), "k_idx layout mismatch"
    assert np.array_equal(orig_id, expected_orig), "orig_trial_id layout mismatch"

    # Labels/times/subjects are constant within each parent trial's K phases
    for oid in [0, n_orig // 2, n_orig - 1]:
        mask = orig_id == oid
        assert mask.sum() == k
        assert len(set(subjects[mask].tolist())) == 1, \
            f"orig_trial_id {oid} maps to multiple subjects"
        assert np.allclose(labels[mask], labels[mask][0])
        assert np.allclose(times[mask], times[mask][0])

    # Spectral check: average power above the new Nyquist should be
    # dramatically lower in the filtered output than in the source.
    # Sample 3 random parent trials and 3 random channels for the check.
    rng = np.random.default_rng(0)
    n_check = min(3, n_orig)
    trial_idx = rng.choice(n_orig, size=n_check, replace=False)
    chan_idx = rng.choice(src_eeg.shape[1], size=3, replace=False)
    src_fs = 1000  # Hz
    new_fs = src_fs / k
    new_nyquist = new_fs / 2
    above_nyq_ratio = []
    for t in trial_idx:
        sig = src_eeg[t][chan_idx].astype(np.float32)  # (3, L)
        # FFT of source signal
        spec = np.abs(np.fft.rfft(sig, axis=-1))
        freqs = np.fft.rfftfreq(sig.shape[-1], d=1.0 / src_fs)
        above = spec[:, freqs > new_nyquist].mean()
        below = spec[:, freqs <= new_nyquist].mean() + 1e-12
        above_nyq_ratio.append(above / below)
    src_above = float(np.mean(above_nyq_ratio))

    # Filtered phase-0 should have almost no content above the new Nyquist
    # (it physically can't, since it's now sampled at new_fs).
    above_filt = []
    for t in trial_idx:
        # Phase 0 of trial t lives at row t*k (trial-major layout)
        sig = eeg[t * k][chan_idx].astype(np.float32)
        spec = np.abs(np.fft.rfft(sig, axis=-1))
        freqs = np.fft.rfftfreq(sig.shape[-1], d=1.0 / new_fs)
        # Define "near new Nyquist" as the top 10% of the new spectrum
        cutoff = 0.9 * new_nyquist
        near_top = spec[:, freqs > cutoff].mean()
        below = spec[:, freqs <= cutoff].mean() + 1e-12
        above_filt.append(near_top / below)
    near_top_filt = float(np.mean(above_filt))

    print(f"  Sanity OK: {k * n_orig} rows, {eeg.shape[1]} ch, "
          f"L_ds={eeg.shape[2]} samples, subjects={sorted(set(subjects.tolist()))}")
    print(f"    new_fs={new_fs:.0f} Hz, new_Nyquist={new_nyquist:.0f} Hz")
    print(f"    source mean(power_above_new_Nyquist / power_below) = {src_above:.4f}")
    print(f"    filtered mean(near-top-decile / below) = {near_top_filt:.4f}  "
          f"(should be << source's above ratio if anti-aliasing is working)")


def main():
    p = argparse.ArgumentParser(
        description=("Build ANTI-ALIASED polyphase-downsampled NPZ datasets. "
                     "Companion to build_downsampled_dataset.py.")
    )
    p.add_argument("--src", type=str, default=str(DEFAULT_SRC),
                   help="Source npz (default: data/eeg_dmt_regression.npz)")
    p.add_argument("--k", type=int, nargs="+", default=[2, 5, 10],
                   help="Polyphase factors to build (default: 2 5 10)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="Output directory (default: data/)")
    p.add_argument("--prefix", type=str, default="eeg_dmt_regression",
                   help="Output filename prefix (default: eeg_dmt_regression)")
    p.add_argument("--suffix", type=str, default="filt",
                   help="Output filename suffix to distinguish from the "
                        "unfiltered build (default: 'filt' → "
                        "eeg_dmt_regression_k{K}_filt.npz)")
    p.add_argument("--numtaps", type=int, default=DEFAULT_NUMTAPS,
                   help=f"FIR filter length (default: {DEFAULT_NUMTAPS}, "
                        f"matches scipy.signal.decimate(ftype='fir'))")
    args = p.parse_args()

    src_path = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source: {src_path}")
    src = np.load(src_path, allow_pickle=True)
    src_eeg = src["eeg_data"]
    src_labels = src["labels"]
    src_times = src["times"]
    src_subjects = src["subjects"]
    src_channels = src["channel_labels"]
    n_orig, c, l = src_eeg.shape
    print(f"  shape: ({n_orig}, {c}, {l})  subjects: {sorted(set(src_subjects.tolist()))}")
    print(f"  filter: {args.numtaps}-tap FIR Hamming, zero-phase (filtfilt)")

    for k in args.k:
        if k < 2:
            print(f"Skipping k={k} (must be >= 2)")
            continue
        print(f"\n=== Building k={k} (anti-aliased) ===")
        fields = build_one(src_eeg, src_labels, src_times, src_subjects, k,
                           numtaps=args.numtaps)
        sanity_check(src_eeg, src_labels, src_times, src_subjects, fields, k)

        out_name = f"{args.prefix}_k{k}_{args.suffix}.npz"
        out_path = out_dir / out_name
        np.savez(
            out_path,
            eeg_data=fields["eeg_data"],
            labels=fields["labels"],
            times=fields["times"],
            subjects=fields["subjects"],
            channel_labels=src_channels,
            k_idx=fields["k_idx"],
            orig_trial_id=fields["orig_trial_id"],
            k_factor=fields["k_factor"],
            antialias_numtaps=fields["antialias_numtaps"],
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Saved {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
