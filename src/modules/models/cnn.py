from src.modules.training import LearnerDetector, RawWindowInput, TorchLearner, TrainConfig

KERNELS = (7, 5, 3)
STRIDE = 2

# channels of the three convolution layers: about 1k / 10k / 100k parameters
PRESETS = {"1k": (8, 12, 12), "10k": (20, 32, 64), "100k": (54, 108, 216)}


def cnn_param_count(channels, kernels=KERNELS) -> int:
    """Conv weights + biases, plus the head Linear(2 * last, 1) on [mean, max] pooling."""

    total, prev = 0, 1

    for c, k in zip(channels, kernels):
        total += prev * c * k + c
        prev = c

    return total + 2 * prev + 1


def cnn_macs(channels, kernels=KERNELS, stride=STRIDE, n_in: int = 398) -> int:
    """Multiply-accumulates of one window (convolutions + head; pooling and ReLU not counted)."""

    total, prev, length = 0, 1, n_in

    for c, k in zip(channels, kernels):
        length = (length - k) // stride + 1
        total += c * length * prev * k
        prev = c

    return total + 2 * prev


def make_cnn(channels=(20, 32, 64), kernels=KERNELS, stride=STRIDE):
    """
    1D CNN on one normalised window, input (B, 1, T), output one logit (B, 1):
    strided Conv1d + ReLU blocks, then [mean, max] pooling over time (mean = energy-like,
    max = peak-like) and a linear head. No normalisation layers: the input is already normalised.
    """

    import torch
    import torch.nn as nn

    class CNN(nn.Module):

        def __init__(self):
            super().__init__()
            layers, prev = [], 1
            for c, k in zip(channels, kernels):
                layers += [nn.Conv1d(prev, c, k, stride=stride), nn.ReLU()]
                prev = c
            self.features = nn.Sequential(*layers)
            self.head = nn.Linear(2 * prev, 1)

        def forward(self, x):
            h = self.features(x)
            return self.head(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1))

    return CNN()


def build_cnn_detector(
    name: str,
    normalizer,
    size: str = "10k",
    config: TrainConfig | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> LearnerDetector:
    """1D CNN on the normalised raw window (variant W); `normalizer` carries N0 / N1 / N2 / N3."""

    channels = PRESETS[size]

    learner = TorchLearner(lambda: make_cnn(channels), lr=lr, weight_decay=weight_decay, device=device)

    return LearnerDetector(name, learner, RawWindowInput(normalizer, "NCT"), config or TrainConfig())