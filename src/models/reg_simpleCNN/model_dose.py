"""
Dose-aware SimpleCNN variants.

Three modes for injecting the known initial dose (mg):
  - "multiplicative":
        y_hat = h(feat) * dose_mg
        The CNN predicts a dose-normalized "shape factor" (ng/mL per mg),
        multiplied by the known dose to recover absolute concentration.
        Enforces the prior that dose sets the scale of the output.
  - "multiplicative_residual":
        y_hat = h_mul(feat) * dose_mg + h_res(cat(feat, dose_emb))
        Pure multiplicative with an additive residual head that can express
        mild nonlinear dose-dependent corrections.
  - "concat":
        y_hat = h(cat(feat, dose_emb))
        No scale prior. Dose is just another feature concatenated into the
        regressor head. Baseline for comparison.

Dose is normalized (dose_mg / DOSE_SCALE) when fed into a head as a feature;
the multiplicative branch uses raw dose_mg so its semantic units are ng/mL.
"""

import torch
import torch.nn as nn

from .model import SE

DOSE_SCALE = 20.0  # max dose in the cohort; used only for feature normalization


class SimpleCNNDose(nn.Module):
    """SimpleCNN with configurable dose-injection head.

    Input:
        x:    [B, in_channels, 3000] EEG
        dose: [B]                    dose in mg (float)
    Output:
        y:    [B]                    predicted plasma DMT (ng/mL)
    """

    def __init__(self, in_channels=32, dropout=0.3, dose_mode="multiplicative"):
        super().__init__()
        if dose_mode not in {"multiplicative", "multiplicative_residual", "concat"}:
            raise ValueError(f"unknown dose_mode: {dose_mode}")
        self.dose_mode = dose_mode

        c1, c2, c3 = 64, 128, 256
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(c1), nn.ReLU(),
        )
        self.se1 = SE(c1)
        self.drop1 = nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(c2), nn.ReLU(),
        )
        self.se2 = SE(c2)
        self.drop2 = nn.Dropout(dropout)
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(c3), nn.ReLU(),
        )
        self.se3 = SE(c3)
        self.drop3 = nn.Dropout(dropout)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

        if dose_mode == "multiplicative":
            # predicts ng/mL per mg of dose
            self.head_mul = nn.Linear(c3, 1)
        elif dose_mode == "multiplicative_residual":
            self.head_mul = nn.Linear(c3, 1)
            self.head_res = nn.Linear(c3 + 1, 1)
        else:  # concat
            self.head_concat = nn.Linear(c3 + 1, 1)

    def extract_features(self, x):
        x = self.drop1(self.se1(self.block1(x)))
        x = self.drop2(self.se2(self.block2(x)))
        x = self.drop3(self.se3(self.block3(x)))
        return self.pool(x)

    def forward(self, x, dose):
        feat = self.extract_features(x)
        dose = dose.view(-1).float()  # [B]
        dose_norm = (dose / DOSE_SCALE).unsqueeze(-1)  # [B, 1] in ~[0, 1]

        if self.dose_mode == "multiplicative":
            h = self.head_mul(feat).squeeze(-1)  # [B]
            return h * dose

        if self.dose_mode == "multiplicative_residual":
            h_mul = self.head_mul(feat).squeeze(-1)
            h_res = self.head_res(torch.cat([feat, dose_norm], dim=1)).squeeze(-1)
            return h_mul * dose + h_res

        # concat
        return self.head_concat(torch.cat([feat, dose_norm], dim=1)).squeeze(-1)
