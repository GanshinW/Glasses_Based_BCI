# models/multimodal_model.py
import torch
import torch.nn as nn

from models.branches import TimeBranch, FreqBranch, ImgBranch

class MultiModalNet(nn.Module):
    """
    Late-fusion multimodal network combining:
      - TimeBranch  (1D‐CNN + LSTM)
      - FreqBranch  (1D‐CNN on band‐power)
      - ImgBranch   (2D‐CNN on spectrograms)
    """
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_bands: int,
        img_out_dim: int,
        hidden_dim: int,
        n_classes: int,
        use_img: bool = True
    ):
        super().__init__()
        self.use_img = use_img

        # instantiate branches
        self.time_branch = TimeBranch(n_channels, n_samples, hidden_dim)
        self.freq_branch = FreqBranch(n_channels, n_bands, hidden_dim)
        if use_img:
            self.img_branch = ImgBranch(img_out_dim)

        # compute fusion dimension
        fuse_dim = hidden_dim + hidden_dim + (img_out_dim if use_img else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, time, freq, img=None):
        t_feat = self.time_branch(time)
        f_feat = self.freq_branch(freq)
        if self.use_img:
            i_feat = self.img_branch(img)
            x = torch.cat([t_feat, f_feat, i_feat], dim=1)
        else:
            x = torch.cat([t_feat, f_feat], dim=1)
        return self.classifier(x)
