import torch
import torch.nn as nn


class SeismicCNN(nn.Module):
    """
    Baseline 1D CNN for binary seismic-event classification.

    Input:
        (batch, samples, 1)

    Output:
        (batch, 1)

    The output is a logit. Sigmoid is applied later when
    converting the output to a probability.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Expected input:
            (batch, samples, 1)

        Internally converted to:
            (batch, 1, samples)
        """

        # PyTorch Conv1d expects:
        # (batch, channels, samples)

        x = x.transpose(1, 2)

        x = self.features(x)

        x = self.classifier(x)

        return x