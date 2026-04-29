import torch
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-and-Excitation block for 1-D signals."""
    # learns to re-weight each channel of your signal based on its global importance.

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, T) -> (B,C,1)
            nn.Flatten(),  # (B,C,1) -> (B,C)
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),  # (B, C)
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)  # (B,C,1)
        return x * w


class Identity(nn.Module):
    def forward(self, x):
        return x


# Channel progression: block 1→64, 2→128, 3+→256
_BLOCK_CHANNELS = [64, 128, 256, 256, 256, 256, 256]
# Default strides when not specified: blocks 1-7
_DEFAULT_STRIDES = [8, 4, 4, 2, 2, 2, 2]


def compute_rf(kernels, strides):
    """Receptive field in samples (= ms @ 1kHz) for a stack of strided convs."""
    rf = kernels[0]
    stride_product = 1
    for i in range(1, len(kernels)):
        stride_product *= strides[i - 1]
        rf += (kernels[i] - 1) * stride_product
    return rf


class SimpleCNN(nn.Module):
    """
    CNN for EEG plasma-DMT regression.
    Input:  [B, in_channels, 3000]
    Output: [B]  (predicted plasma DMT concentration in ng/mL)

    kernels — list/tuple of kernel sizes, one per block (length = n_blocks, max 7).
    strides — list/tuple of strides, one per block. Defaults to _DEFAULT_STRIDES[:n].
    use_se  — if False, SE blocks are replaced with identity.

    Channel progression: 64 / 128 / 256 / 256 / 256 / 256 / 256.
    RF (ms @ 1kHz) = compute_rf(kernels, strides).

    Linear subject adaptation (optional)
    ------------------------------------
    use_baseline_subtraction: if True, the regressor sees
        feat' = feat - baseline_lambda * mu_s
    where mu_s is a per-sample baseline feature passed via forward(x, mu_s=...)
    and baseline_lambda is a learnable scalar (initialised to 1.0).
    The baseline buffer mu_s itself is computed externally (in the training
    loop) by averaging extract_features() over each subject's pre-injection
    trials. mu_s must be detached so gradients don't flow through it.
    """

    def __init__(self, in_channels=32, dropout=0.3,
                 kernels=(15, 7, 7), strides=None, use_se=True,
                 channels=None,
                 use_baseline_subtraction=False):
        n = len(kernels)
        assert 1 <= n <= 7, "kernels must have 1–7 elements"
        super().__init__()
        self.n_blocks = n

        strides = list(strides) if strides is not None else _DEFAULT_STRIDES[:n]
        assert len(strides) == n, "kernels and strides must have the same length"

        channels = list(channels) if channels is not None else _BLOCK_CHANNELS[:n]
        assert len(channels) == n, (
            f"channels must have the same length as kernels (got {len(channels)} vs {n})"
        )
        ch_in = in_channels
        for i, (k, s, ch_out) in enumerate(zip(kernels, strides, channels), start=1):
            setattr(self, f"block{i}", nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=s, padding=k // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(),
            ))
            setattr(self, f"se{i}", SE(ch_out) if use_se else Identity())
            setattr(self, f"drop{i}", nn.Dropout(dropout))
            ch_in = ch_out

        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.regressor = nn.Linear(channels[-1], 1)

        self.feature_dim = channels[-1]
        self.use_baseline_subtraction = use_baseline_subtraction
        if use_baseline_subtraction:
            # Single learnable scalar shared across subjects/features. Init at
            # 1.0 = literal paper subtraction; the model can learn to shrink
            # toward 0 if subtraction hurts.
            self.baseline_lambda = nn.Parameter(torch.ones(1))

    def extract_features(self, x):
        for i in range(1, self.n_blocks + 1):
            x = getattr(self, f"drop{i}")(
                getattr(self, f"se{i}")(
                    getattr(self, f"block{i}")(x)
                )
            )
        return self.pool(x)

    def forward(self, x, mu_s=None):
        feat = self.extract_features(x)
        if self.use_baseline_subtraction and mu_s is not None:
            feat = feat - self.baseline_lambda * mu_s
        return self.regressor(feat).squeeze(-1)
