# models/cnn_model.py

import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, n_channels=54, n_samples=12800, num_classes=3):
        """
        A simple CNN architecture for emotion classification.
        Input tensor shape should be (batch, 1, n_channels, n_samples).
        Parameters:
            n_channels (int): Number of EEG channels, e.g., 54
            n_samples (int): Number of time points per trial, e.g., 12.5s × 1024Hz ≈ 12800
            num_classes (int): Number of output classes (e.g., 3: Calm/Pos/Neg)
        """
        super(EmotionCNN, self).__init__()
        # First convolution: kernel size (1 × 5) along time dimension
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,5), padding=(0,2))
        self.relu = nn.ReLU()
        # Max-pooling along time dimension (kernel (1 × 2))
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        # Second convolution
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,5), padding=(0,2))
        # Compute time dimension after two poolings
        time_dim_after = n_samples // 2 // 2  # Each pool halves time dimension
        # Fully connected layers
        self.fc1 = nn.Linear(32 * n_channels * time_dim_after, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.
        x shape: (batch, 1, n_channels, n_samples)
        """
        x = self.relu(self.conv1(x))   # -> (batch, 16, n_channels, n_samples)
        x = self.pool(x)               # -> (batch, 16, n_channels, n_samples/2)
        x = self.relu(self.conv2(x))   # -> (batch, 32, n_channels, n_samples/2)
        x = self.pool(x)               # -> (batch, 32, n_channels, n_samples/4)
        x = x.view(x.size(0), -1)      # Flatten -> (batch, 32 * n_channels * time_dim_after)
        x = self.relu(self.fc1(x))     # -> (batch, 128)
        out = self.fc2(x)              # -> (batch, num_classes)
        return out
