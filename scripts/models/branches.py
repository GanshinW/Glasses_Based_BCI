# models/branches.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

class TimeBranch(nn.Module):
    """
    1D‐CNN + LSTM branch for time‐domain EEG.
    Input: (batch, n_channels, n_samples)
    Output: (batch, hidden_dim)
    """
    def __init__(self, n_channels: int, n_samples: int, hidden_dim: int = 64):
        super().__init__()
        # two 1D‐CNN layers with pooling
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),                # halves time length
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),                # halves again
        )
        # LSTM over the feature sequence
        seq_len = n_samples // 4            # after two pools
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim,
                            batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        x = self.cnn(x)                     # -> (batch, 64, seq_len)
        x = x.permute(0, 2, 1)              # -> (batch, seq_len, 64)
        _, (h_n, _) = self.lstm(x)          # h_n: (1, batch, hidden_dim)
        return h_n.squeeze(0)               # -> (batch, hidden_dim)


class FreqBranch(nn.Module):
    """
    1D‐CNN branch for frequency‐domain band‐power features.
    Input: (batch, n_channels, n_bands)
    Output: (batch, out_dim)
    """
    def __init__(self, n_channels: int, n_bands: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, out_dim, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)         # -> (batch, out_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_bands)
        x = self.net(x)                     # -> (batch, out_dim, 1)
        return x.squeeze(-1)                # -> (batch, out_dim)
    

class ImgBranch(nn.Module):
    """
    2D‐CNN branch for time–frequency images.
    Uses ResNet18 backbone (single‐channel input).
    Input: (batch, 1, H, W)
    Output: (batch, out_dim)
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        # load ResNet18, replace first conv & fc
        self.net = resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        return self.net(x)                  # -> (batch, out_dim)