import numpy as np
import torch
from torch.utils.data import Dataset

from src.modules.labeling import LabeledWindowRecord


class SeismicDataset(Dataset):
    """
    PyTorch adapter for labeled seismic windows.
    """

    def __init__(
        self,
        windows: list[LabeledWindowRecord],
    ):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):

        window = self.windows[index]

        # Waveform samples
        waveform = np.asarray(
            window.data,
            dtype=np.float32,
        )

        # Shape:
        # (samples,) -> (samples, 1)
        waveform = waveform.reshape(-1, 1)

        x = torch.from_numpy(waveform)

        # Binary target
        y = torch.tensor(
            [float(window.label)],
            dtype=torch.float32,
        )

        return x, y