"""Spectral-input CNN for EEG plasma-DMT regression.

Replaces SimpleCNN's first temporal convolutions with a fixed STFT frontend
that produces a (channels x freq x time) representation, then runs a small 2D
conv stack on top. Same external interface as SimpleCNN — exposes
`feature_dim`, `extract_features`, `forward(x, mu_s=None)`, and an optional
`baseline_lambda` — so train_cv.run_fold can swap models with a factory.

Design choices for the MVP (apr29):
- STFT magnitude (or power), Hann window, configurable n_fft and hop_length.
- Optional log compression (`log(power + eps)`) to tame the heavy-tailed EEG
  power distribution.
- Bins above `f_max` (default 100 Hz) are dropped — the source data is already
  software-low-passed there (see misc/study_notes/eeg_source_psd.png).
- The frontend is parameter-free (fixed Hann window). Future work can swap it
  for a SincConv or learnable filterbank without changing the backbone.
- 2D backbone: 3 Conv2d blocks, channels (C1, C2, C3), strides (1,2), (2,2),
  (2,2). Default channels (64, 128, 256) → ~460k params, similar to the
  SimpleCNN k=2 default champion (436k).
"""

import torch
import torch.nn as nn

from .model import SE


class SpectralFrontend(nn.Module):
    """Fixed STFT magnitude (or power) frontend.

    Forward:
        Input  x: (B, C, L)            time-domain EEG, L samples per trial.
        Output  : (B, C, F, T)         spectrogram per channel.

    F = number of kept frequency bins (= ceil(n_fft/2)+1 truncated to f<=f_max
    if `fs` is given; otherwise the full one-sided spectrum is kept).
    T = number of STFT frames = floor((L - n_fft) / hop_length) + 1 with
        center=False; floor(L / hop_length) + 1 with center=True.

    Args:
        n_fft:        FFT size. Bin width in Hz = fs / n_fft.
        hop_length:   STFT hop in samples.
        fs:           Sampling rate of the input. Used only to truncate bins
                      above f_max. None = keep all bins.
        f_max:        Frequency cutoff in Hz. Bins above this are dropped.
        power:        Exponent on |STFT|. 1 = magnitude, 2 = power.
        log:          If True, apply log(eps + value) compression.
        eps:          Stabilizer for log.
        center:       Pass-through to torch.stft.
    """

    def __init__(self, n_fft=256, hop_length=32, fs=500, f_max=100.0,
                 power=2.0, log=True, eps=1e-6, center=True):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.fs = float(fs) if fs is not None else None
        self.f_max = float(f_max) if f_max is not None else None
        self.power = float(power)
        self.log = bool(log)
        self.eps = float(eps)
        self.center = bool(center)
        self.register_buffer("window",
                             torch.hann_window(self.n_fft), persistent=False)

        n_bins_full = self.n_fft // 2 + 1
        if self.fs is not None and self.f_max is not None:
            bin_hz = self.fs / self.n_fft
            self.n_bins = int(min(n_bins_full,
                                  int(self.f_max / bin_hz) + 1))
        else:
            self.n_bins = n_bins_full

    def forward(self, x):
        # x: (B, C, L) -> reshape so torch.stft treats each channel independently
        B, C, L = x.shape
        x_flat = x.reshape(B * C, L)
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )  # (B*C, F_full, T)
        mag = spec.abs()
        if self.power != 1.0:
            mag = mag.pow(self.power)
        if self.log:
            mag = torch.log(mag + self.eps)
        mag = mag[:, : self.n_bins, :]  # truncate to f<=f_max
        F, T = mag.shape[-2], mag.shape[-1]
        return mag.reshape(B, C, F, T)

    def output_shape(self, signal_length):
        """Return (F, T) for an input of `signal_length` samples."""
        if self.center:
            T = signal_length // self.hop_length + 1
        else:
            T = max(0, (signal_length - self.n_fft) // self.hop_length + 1)
        return self.n_bins, T


class _SE2d(nn.Module):
    """SE block for 2D feature maps (channel-wise re-weighting)."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpectralCNN(nn.Module):
    """STFT frontend + 2D conv backbone for plasma-DMT regression.

    Same external surface as SimpleCNN:
        - feature_dim                       (int, last conv channels)
        - extract_features(x) -> (B, F)     pooled feature vector
        - forward(x, mu_s=None) -> (B,)     scalar prediction
        - baseline_lambda                   (optional learnable scalar)

    Default channels (64, 128, 256) and 3 Conv2d blocks with strides
    (1,2), (2,2), (2,2) along (freq, time).
    """

    def __init__(self, in_channels=32, dropout=0.3,
                 channels=(64, 128, 256), use_se=True,
                 n_fft=256, hop_length=32, fs=500, f_max=100.0,
                 spectral_power=2.0, spectral_log=True,
                 use_baseline_subtraction=False):
        super().__init__()
        self.frontend = SpectralFrontend(
            n_fft=n_fft, hop_length=hop_length, fs=fs, f_max=f_max,
            power=spectral_power, log=spectral_log,
        )
        channels = list(channels)
        assert len(channels) == 3, "MVP backbone uses exactly 3 conv blocks"
        c1, c2, c3 = channels

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=(3, 7),
                      stride=(1, 2), padding=(1, 3)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.se1 = _SE2d(c1) if use_se else nn.Identity()
        self.drop1 = nn.Dropout2d(dropout)

        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(3, 5),
                      stride=(2, 2), padding=(1, 2)),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.se2 = _SE2d(c2) if use_se else nn.Identity()
        self.drop2 = nn.Dropout2d(dropout)

        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.se3 = _SE2d(c3) if use_se else nn.Identity()
        self.drop3 = nn.Dropout2d(dropout)

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.regressor = nn.Linear(c3, 1)

        self.feature_dim = c3
        self.use_baseline_subtraction = use_baseline_subtraction
        if use_baseline_subtraction:
            self.baseline_lambda = nn.Parameter(torch.ones(1))

    def extract_features(self, x):
        z = self.frontend(x)                            # (B, C, F, T)
        z = self.drop1(self.se1(self.block1(z)))
        z = self.drop2(self.se2(self.block2(z)))
        z = self.drop3(self.se3(self.block3(z)))
        return self.pool(z)                             # (B, feature_dim)

    def forward(self, x, mu_s=None):
        feat = self.extract_features(x)
        if self.use_baseline_subtraction and mu_s is not None:
            feat = feat - self.baseline_lambda * mu_s
        return self.regressor(feat).squeeze(-1)
