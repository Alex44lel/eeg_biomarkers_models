"""
Feature extraction for EEG graph construction.

- Node features: spectral band powers (delta, theta, alpha, beta, gamma)
- Edge features: amplitude envelope correlation (AEC)
- Conditional features: standard 10-10 electrode coordinates (optional)
"""

import numpy as np
from scipy.signal import welch, hilbert


# Standard EEG frequency bands (Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Standard 10-10 electrode coordinates (x=right, y=anterior, z=superior)
# Approximate unit-sphere positions from Oostenveld & Praamstra (2001)
ELECTRODE_COORDS_10_10 = {
    "Fp1":  (-0.31,  0.95,  0.05),
    "Fp2":  ( 0.31,  0.95,  0.05),
    "F3":   (-0.55,  0.67,  0.50),
    "F4":   ( 0.55,  0.67,  0.50),
    "C3":   (-0.72,  0.00,  0.69),
    "C4":   ( 0.72,  0.00,  0.69),
    "P3":   (-0.55, -0.67,  0.50),
    "P4":   ( 0.55, -0.67,  0.50),
    "O1":   (-0.31, -0.95,  0.05),
    "O2":   ( 0.31, -0.95,  0.05),
    "F7":   (-0.81,  0.59,  0.00),
    "F8":   ( 0.81,  0.59,  0.00),
    "T7":   (-1.00,  0.00,  0.00),
    "T8":   ( 1.00,  0.00,  0.00),
    "P7":   (-0.81, -0.59,  0.00),
    "P8":   ( 0.81, -0.59,  0.00),
    "Fz":   ( 0.00,  0.71,  0.71),
    "Cz":   ( 0.00,  0.00,  1.00),
    "Pz":   ( 0.00, -0.71,  0.71),
    "Oz":   ( 0.00, -0.95,  0.31),
    "FC1":  (-0.35,  0.35,  0.87),
    "FC2":  ( 0.35,  0.35,  0.87),
    "CP1":  (-0.35, -0.35,  0.87),
    "CP2":  ( 0.35, -0.35,  0.87),
    "FC5":  (-0.71,  0.38,  0.59),
    "FC6":  ( 0.71,  0.38,  0.59),
    "CP5":  (-0.71, -0.38,  0.59),
    "CP6":  ( 0.71, -0.38,  0.59),
    "TP9":  (-0.95, -0.31, -0.05),
    "TP10": ( 0.95, -0.31, -0.05),
    "POz":  ( 0.00, -0.81,  0.59),
    "FCz":  ( 0.00,  0.38,  0.92),
}


def compute_band_powers(eeg, fs=1000):
    """
    Compute log spectral band power for each channel.

    Parameters
    ----------
    eeg : ndarray, shape (n_channels, n_samples)
    fs  : sampling rate in Hz

    Returns
    -------
    powers : ndarray, shape (n_channels, 5)
        Log10 band powers for [delta, theta, alpha, beta, gamma].
    """
    n_channels = eeg.shape[0]
    nperseg = min(eeg.shape[1], fs)  # 1-second windows
    freqs, psd = welch(eeg, fs=fs, nperseg=nperseg, axis=1)  # (n_channels, n_freqs)

    powers = np.zeros((n_channels, len(BANDS)), dtype=np.float32)
    for i, (low, high) in enumerate(BANDS.values()):
        band_mask = (freqs >= low) & (freqs < high)
        # Integrate PSD over band (trapezoidal)
        powers[:, i] = np.trapezoid(psd[:, band_mask], freqs[band_mask], axis=1)

    # Log transform (standard in EEG literature)
    powers = np.log10(powers + 1e-10)
    return powers


def compute_aec(eeg):
    """
    Compute amplitude envelope correlation (AEC) between all channel pairs.

    Broadband approach: Hilbert transform -> amplitude envelope -> Pearson correlation.

    Parameters
    ----------
    eeg : ndarray, shape (n_channels, n_samples)

    Returns
    -------
    aec : ndarray, shape (n_channels, n_channels)
        Symmetric correlation matrix, values in [-1, 1].
    """
    analytic = hilbert(eeg, axis=1)
    envelope = np.abs(analytic)  # (n_channels, n_samples)
    aec = np.corrcoef(envelope)  # (n_channels, n_channels)
    # Clean up NaNs (constant channels)
    aec = np.nan_to_num(aec, nan=0.0)
    return aec.astype(np.float32)


def get_electrode_coords(channel_labels):
    """
    Look up 3D coordinates for a list of channel labels.

    Parameters
    ----------
    channel_labels : list of str

    Returns
    -------
    coords : ndarray, shape (n_channels, 3)
    """
    coords = []
    for ch in channel_labels:
        if ch not in ELECTRODE_COORDS_10_10:
            raise ValueError(f"Unknown electrode '{ch}' — not in 10-10 coordinate table")
        coords.append(ELECTRODE_COORDS_10_10[ch])
    return np.array(coords, dtype=np.float32)
