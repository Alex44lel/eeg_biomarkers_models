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


class SimpleCNN(nn.Module):
    """
    CNN with SE blocks for EEG plasma-DMT regression.
    Input:  [B, in_channels, 3000]
    Output: [B]  (predicted plasma DMT concentration in ng/mL)
    """

    def __init__(self, in_channels=32, dropout=0.3):
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
        )
        self.se1 = SE(c1)
        self.drop1 = nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
        )
        self.se2 = SE(c2)
        self.drop2 = nn.Dropout(dropout)

        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )
        self.se3 = SE(c3)
        self.drop3 = nn.Dropout(dropout)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.regressor = nn.Linear(c3, 1)

    def extract_features(self, x):
        """Return 128-dim features before the regressor head."""
        x = self.drop1(self.se1(self.block1(x)))
        x = self.drop2(self.se2(self.block2(x)))
        x = self.drop3(self.se3(self.block3(x)))
        return self.pool(x)

    def forward(self, x):
        x = self.extract_features(x)
        return self.regressor(x).squeeze(-1)
