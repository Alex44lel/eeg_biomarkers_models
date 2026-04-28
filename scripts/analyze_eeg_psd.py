"""
Power-spectrum audit of the source EEG dataset (data/eeg_dmt_regression.npz).

Settles two questions raised during the polyphase / capacity sweep analysis:
  1. What's actually in the source signal above 100 Hz? (Used to interpret
     the apr27 anti-aliased polyphase result.)
  2. Where does plasma-correlated EEG activity live, by frequency band?

Saves a 2-panel PNG and prints band-power tables to stdout.

Usage (from project root):
    python scripts/analyze_eeg_psd.py
        [--src data/eeg_dmt_regression.npz]
        [--out misc/study_notes/eeg_source_psd.png]
        [--n-trials 200] [--seed 0]
"""

from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# np.trapz was removed in NumPy 2.0 — use trapezoid on new versions.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def trial_psd(signal_2d, fs):
    """signal_2d: (C, L). Returns (freqs, psd) averaged over channels."""
    freqs, psd = welch(signal_2d, fs=fs, nperseg=1000, noverlap=500, axis=-1)
    return freqs, psd.mean(axis=0)


def group_psd(eeg, idx, fs):
    psds = [trial_psd(eeg[i], fs)[1] for i in idx]
    f = welch(eeg[idx[0]], fs=fs, nperseg=1000, noverlap=500, axis=-1)[0]
    return f, np.mean(psds, axis=0), np.std(psds, axis=0)


def band_power(f, psd, lo, hi):
    mask = (f >= lo) & (f <= hi)
    return float(_trapz(psd[mask], f[mask]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=str,
                   default="data/eeg_dmt_regression.npz")
    p.add_argument("--out", type=str,
                   default="misc/study_notes/eeg_source_psd.png")
    p.add_argument("--n-trials", type=int, default=1200,
                   help="How many trials to randomly sample for averaging.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fs", type=int, default=1000,
                   help="Original sampling rate in Hz (default: 1000)")
    args = p.parse_args()

    warnings.filterwarnings("ignore")

    src = np.load(args.src, allow_pickle=True)
    eeg = src["eeg_data"]
    labels = src["labels"]
    fs = args.fs

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_trials, eeg.shape[0])
    sample_idx = rng.choice(eeg.shape[0], size=n_sample, replace=False)

    # Tertile split on plasma value
    q33, q66 = np.quantile(labels, [0.33, 0.66])
    low_idx = sample_idx[labels[sample_idx] <= q33]
    high_idx = sample_idx[labels[sample_idx] >= q66]
    print(f"Low-plasma trials  (≤{q33:.1f} ng/mL): {len(low_idx)} of {n_sample}")
    print(f"High-plasma trials (≥{q66:.1f} ng/mL): {len(high_idx)} of {n_sample}")

    f, all_mean, all_std = group_psd(eeg, sample_idx, fs)
    _, low_mean, _ = group_psd(eeg, low_idx, fs)
    _, high_mean, _ = group_psd(eeg, high_idx, fs)

    BANDS = [
        ("delta", 0.5, 4),
        ("theta", 4, 8),
        ("alpha", 8, 13),
        ("beta", 13, 30),
        ("gamma_low", 30, 50),
        ("gamma_high", 50, 100),
        ("100-150", 100, 150),
        ("150-200", 150, 200),
        ("200-250", 200, 250),
        ("250-500", 250, 500),
    ]

    total = band_power(f, all_mean, 0.5, 100)
    print(f"\n=== Power per band (avg over {n_sample} trials × 32 channels) ===")
    print(f"{'band':<13} {'range Hz':<12} {'power (μV²)':<14} {'% of <100 Hz total'}")
    for name, lo, hi in BANDS:
        bp = band_power(f, all_mean, lo, hi)
        pct = bp / total * 100
        print(f"{name:<13} {f'{lo:>4.1f}-{hi:<5.0f}':<12} {bp:<14.3e} {pct:>6.2f}%")

    print(f"\n=== High/Low plasma PSD ratio per band ===")
    print(f"{'band':<13} {'range':<12} {'ratio'}")
    for name, lo, hi in BANDS:
        pl = band_power(f, low_mean, lo, hi)
        ph = band_power(f, high_mean, lo, hi)
        r = ph / pl if pl > 0 else float("nan")
        flag = "  ←" if abs(np.log(max(r, 1e-12))) > 0.10 else ""
        print(f"{name:<13} {f'{lo:>4.1f}-{hi:<5.0f}':<12} {r:>+6.3f}{flag}")

    # ---------- plot ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    # --- panel 1: log-log PSD ---
    ax = axes[0]
    ax.loglog(f, all_mean, label=f"All ({n_sample} trials, ±1σ shaded)",
              color="black", lw=1.5)
    ax.fill_between(f, all_mean - all_std, all_mean + all_std,
                    alpha=0.15, color="black")
    ax.loglog(f, low_mean,
              label=f"Low plasma (≤{q33:.0f} ng/mL, n={len(low_idx)})",
              color="steelblue", lw=1.2, alpha=0.9)
    ax.loglog(f, high_mean,
              label=f"High plasma (≥{q66:.0f} ng/mL, n={len(high_idx)})",
              color="darkorange", lw=1.2, alpha=0.9)
    # Polyphase Nyquist markers
    for nyq, lab in [(250, "k=2"), (100, "k=5"), (50, "k=10")]:
        ax.axvline(nyq, color="red", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.text(nyq * 1.05, all_mean.max(), f" Nyq {lab}\n {nyq} Hz",
                color="red", fontsize=8, va="top")
    # EEG band shading + letters
    for lo, hi, name, c in [
        (0.5, 4, "δ", "#dcdcfa"),
        (4, 8, "θ", "#c4f0ec"),
        (8, 13, "α", "#d8f5d2"),
        (13, 30, "β", "#fdf3c4"),
        (30, 80, "γ", "#f5d4d4"),
    ]:
        ax.axvspan(lo, hi, alpha=0.25, color=c)
        ax.text(np.sqrt(lo * hi), 1e-3, name,
                ha="center", color="dimgray", fontsize=10)
    ax.set_xlim(0.5, 500)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (μV² / Hz)")
    ax.set_title(f"Average power spectrum — {Path(args.src).name} "
                 f"(fs={fs} Hz, {n_sample} trials × 32 ch)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # --- panel 2: high/low plasma PSD ratio ---
    ax = axes[1]
    ratio = high_mean / np.maximum(low_mean, 1e-12)
    ax.semilogx(f, ratio, color="purple", lw=1.4)
    ax.axhline(1.0, color="black", linestyle=":", lw=1)
    for nyq, lab in [(250, "k=2"), (100, "k=5"), (50, "k=10")]:
        ax.axvline(nyq, color="red", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xlim(0.5, 500)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD ratio: high / low plasma")
    ax.set_title("Plasma-correlation: high-plasma vs low-plasma PSD ratio "
                 "(>1: more power at high plasma; <1: less)")
    ax.grid(True, which="both", alpha=0.3)

    # --- panel 3: log-log PSD zoomed to 0.5–50 Hz ---
    ax = axes[2]
    ax.loglog(f, all_mean, label=f"All ({n_sample} trials, ±1σ shaded)",
              color="black", lw=1.5)
    ax.fill_between(f, all_mean - all_std, all_mean + all_std,
                    alpha=0.15, color="black")
    ax.loglog(f, low_mean,
              label=f"Low plasma (≤{q33:.0f} ng/mL, n={len(low_idx)})",
              color="steelblue", lw=1.2, alpha=0.9)
    ax.loglog(f, high_mean,
              label=f"High plasma (≥{q66:.0f} ng/mL, n={len(high_idx)})",
              color="darkorange", lw=1.2, alpha=0.9)
    ax.axvline(50, color="red", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.text(50 * 1.05, all_mean.max(), f" Nyq k=10\n 50 Hz",
            color="red", fontsize=8, va="top")
    for lo, hi, name, c in [
        (0.5, 4, "δ", "#dcdcfa"),
        (4, 8, "θ", "#c4f0ec"),
        (8, 13, "α", "#d8f5d2"),
        (13, 30, "β", "#fdf3c4"),
        (30, 50, "γ", "#f5d4d4"),
    ]:
        ax.axvspan(lo, hi, alpha=0.25, color=c)
        ax.text(np.sqrt(lo * hi), 1e-3, name,
                ha="center", color="dimgray", fontsize=10)
    ax.set_xlim(0.5, 50)
    ax.set_ylim(1e-3, 1e1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (μV² / Hz)")
    ax.set_title(f"Average power spectrum (zoomed 0.5–50 Hz) — {Path(args.src).name}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # ---------- band-power table PNG ----------
    rows = []
    for name, lo, hi in BANDS:
        bp   = band_power(f, all_mean, lo, hi)
        pct  = bp / total * 100
        pl   = band_power(f, low_mean,  lo, hi)
        ph   = band_power(f, high_mean, lo, hi)
        r    = ph / pl if pl > 0 else float("nan")
        rows.append([name, f"{lo:.1f}–{hi:.0f}", f"{bp:.3e}", f"{pct:.2f}%", f"{r:.3f}"])

    col_labels = ["Band", "Range (Hz)", "Power (μV²/Hz)", "% of <100 Hz", "High/Low ratio"]
    fig_t, ax_t = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.2))
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    ax_t.set_title(f"Band-power summary — {n_sample} trials × 32 ch  (fs={fs} Hz)",
                   fontsize=11, pad=12)
    tbl_path = out_path.with_stem(out_path.stem + "_band_table")
    fig_t.savefig(tbl_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig_t)
    print(f"Saved: {tbl_path}")


if __name__ == "__main__":
    main()
