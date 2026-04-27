"""
Build polyphase-downsampled NPZ datasets from data/eeg_dmt_regression.npz.

For each factor K, every original trial of shape (32, L) is split into K
interleaved sub-signals
    sub_i[c, t] = eeg[c, t * K + i]      for i in [0, K)
each of length L // K. The K sub-signals share the same subject ID, label
and (post-injection) time as the parent trial — they are different *views*
of the same EEG epoch at a K-times lower effective sample rate, with the
phase indexed by `k_idx` ∈ [0, K).

For LOSO cross-validation this is safe by construction: subject grouping
is preserved (same `subjects` value), so all K splits of a subject end up
either entirely in train or entirely in val. `orig_trial_id` is also
written so any future stricter grouping (e.g. group-by-trial) is trivial.

Output schema (mirrors eeg_dmt_regression.npz, plus two new fields):
    eeg_data         (K * N_orig, 32, L // K)  float32
    labels           (K * N_orig,)             float32
    times            (K * N_orig,)             float32
    subjects         (K * N_orig,)             <U3
    channel_labels   (32,)                     str
    k_idx            (K * N_orig,)             int8       (polyphase index)
    orig_trial_id    (K * N_orig,)             int32      (parent trial)
    k_factor         scalar int                          (K, for traceability)

Usage (from project root):
    python -m src.models.reg_simpleCNN.build_downsampled_dataset --k 2 3 4
    python -m src.models.reg_simpleCNN.build_downsampled_dataset \
        --src data/eeg_dmt_regression.npz --k 2 --out-dir data/
"""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = PROJECT_ROOT / "data" / "eeg_dmt_regression.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"


def polyphase_split(eeg, k):
    """Split eeg of shape (N, C, L) into K polyphase views.

    Returns an array of shape (K * N, C, L // K). The K copies of trial n
    are at rows [k_idx * N + n for k_idx in range(K)]

    row 0:  trial 0, phase 0
    row 1:  trial 0, phase 1
    ...
    row M-1: trial 0, phase K-1
    row M:   trial 1, phase 0
    """
    n, c, l = eeg.shape
    l_ds = l // k  # lenght after decimation
    # Trial-major output: rows are [(trial 0, k=0), (trial 0, k=1), ...,
    # (trial 0, k=K-1), (trial 1, k=0), ...]. This way trial t occupies
    # rows [t*K : (t+1)*K] which makes leakage checks simple.
    out = np.empty((n * k, c, l_ds), dtype=eeg.dtype)
    for i in range(k):
        # eeg[:, :, i::k] has length ceil((l - i) / k). Truncate to l_ds
        # so all K sub-signals are the same length (drops at most K-1
        # tail samples from phases with longer slices).
        out[i::k] = eeg[:, :, i:i + k * l_ds:k]
        # out[i::k], (start:stop:step)
    return out


def replicate_per_phase(arr, k):
    """Replicate a length-N 1-D array to length K*N in trial-major order:
    [arr[0], arr[0], ..., arr[0], arr[1], ...] (each value repeated K
    times consecutively). Mirrors the row layout produced by
    polyphase_split."""
    return np.repeat(arr, k, axis=0)


def build_one(src_eeg, src_labels, src_times, src_subjects, k):
    """Return the field dict for one K-factor dataset."""
    n_orig, c, l = src_eeg.shape
    if l % k != 0:
        # Dropping at most k-1 trailing samples per phase keeps shapes
        # consistent. Warn so the user is aware; for L=3000 and K∈{2,3,4}
        # only K=4 hits this (3000 / 4 = 750, no remainder; K=3 → 1000;
        # K=2 → 1500). All clean for the current dataset.
        l_ds = l // k
        print(f"  NOTE: L={l} not divisible by k={k}; using L_ds={l_ds} "
              f"(dropping {l - k * l_ds} trailing samples)")

    eeg_ds = polyphase_split(src_eeg.astype(np.float32, copy=False), k)
    labels_ds = replicate_per_phase(src_labels.astype(np.float32, copy=False), k)
    times_ds = replicate_per_phase(src_times.astype(np.float32, copy=False), k)
    subjects_ds = replicate_per_phase(src_subjects, k)

    # k_idx 0..K-1 cycles within each parent trial (trial-major layout)
    k_idx = np.tile(np.arange(k, dtype=np.int8), n_orig)
    # orig_trial_id: each block of K rows shares the same parent index
    orig_trial_id = np.repeat(np.arange(n_orig, dtype=np.int32), k)

    return {
        "eeg_data": eeg_ds,
        "labels": labels_ds,
        "times": times_ds,
        "subjects": subjects_ds,
        "k_idx": k_idx,
        "orig_trial_id": orig_trial_id,
        "k_factor": np.int32(k),
    }


def sanity_check(src_eeg, src_labels, src_times, src_subjects, fields, k):
    """Hard-fail asserts that the polyphase build is internally consistent
    and that subject grouping is preserved. Intended to run on every
    build so a corrupt file is never silently produced."""
    n_orig = src_eeg.shape[0]
    eeg = fields["eeg_data"]
    labels = fields["labels"]
    times = fields["times"]
    subjects = fields["subjects"]
    k_idx = fields["k_idx"]
    orig_id = fields["orig_trial_id"]

    # 1. Counts
    assert eeg.shape[0] == k * n_orig, f"row count mismatch: {eeg.shape[0]} vs {k * n_orig}"
    assert eeg.shape[1] == src_eeg.shape[1], "channel count changed"
    assert eeg.shape[2] == src_eeg.shape[2] // k, "downsampled length wrong"
    assert labels.shape == (k * n_orig,)
    assert times.shape == (k * n_orig,)
    assert subjects.shape == (k * n_orig,)
    assert k_idx.shape == (k * n_orig,)
    assert orig_id.shape == (k * n_orig,)

    # 2. k_idx and orig_trial_id layout (trial-major)
    expected_k_idx = np.tile(np.arange(k, dtype=np.int8), n_orig)
    expected_orig = np.repeat(np.arange(n_orig, dtype=np.int32), k)
    assert np.array_equal(k_idx, expected_k_idx), "k_idx layout wrong"
    assert np.array_equal(orig_id, expected_orig), "orig_trial_id layout wrong"

    # 3. Per-trial: all K rows share subject/label/time with the parent
    for t in [0, n_orig // 2, n_orig - 1]:
        block = slice(t * k, (t + 1) * k)
        assert np.all(subjects[block] == src_subjects[t]), \
            f"subject mismatch at trial {t}"
        assert np.allclose(labels[block], src_labels[t], rtol=1e-5), \
            f"label mismatch at trial {t}"
        assert np.allclose(times[block], src_times[t], rtol=1e-5), \
            f"time mismatch at trial {t}"

    # 4. Polyphase value round-trip on a few random rows
    rng = np.random.default_rng(0)
    test_trials = rng.choice(n_orig, size=min(8, n_orig), replace=False)
    l_ds = src_eeg.shape[2] // k
    for t in test_trials:
        for i in range(k):
            row = t * k + i
            expected = src_eeg[t, :, i:i + k * l_ds:k]
            actual = eeg[row]
            assert np.allclose(actual, expected, rtol=1e-5, atol=1e-5), \
                f"polyphase value mismatch at trial {t}, k_idx {i}"

    # 5. Subject-grouping invariant: every (subject, orig_trial_id) pair
    # has exactly K rows, and orig_trial_id maps to a single subject.
    # This is what guarantees LOSO can keep all splits of a subject
    # together.
    unique_orig = np.unique(orig_id)
    assert len(unique_orig) == n_orig
    for oid in [0, n_orig // 2, n_orig - 1]:
        mask = orig_id == oid
        assert mask.sum() == k, f"orig_trial_id {oid} has {mask.sum()} rows, expected {k}"
        assert len(set(subjects[mask].tolist())) == 1, \
            f"orig_trial_id {oid} maps to multiple subjects"

    print(f"  Sanity OK: {k * n_orig} rows, {eeg.shape[1]} ch, "
          f"L_ds={eeg.shape[2]} samples, subjects={sorted(set(subjects.tolist()))}")


def main():
    p = argparse.ArgumentParser(
        description="Build polyphase-downsampled NPZ datasets for SimpleCNN training"
    )
    p.add_argument("--src", type=str, default=str(DEFAULT_SRC),
                   help="Source npz (default: data/eeg_dmt_regression.npz)")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4],
                   help="Polyphase factors to build (default: 2 3 4)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="Output directory (default: data/)")
    p.add_argument("--prefix", type=str, default="eeg_dmt_regression",
                   help="Output filename prefix (default: eeg_dmt_regression)")
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

    for k in args.k:
        if k < 2:
            print(f"Skipping k={k} (must be >= 2)")
            continue
        print(f"\n=== Building k={k} ===")
        fields = build_one(src_eeg, src_labels, src_times, src_subjects, k)
        sanity_check(src_eeg, src_labels, src_times, src_subjects, fields, k)

        out_path = out_dir / f"{args.prefix}_k{k}.npz"
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
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Saved {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
