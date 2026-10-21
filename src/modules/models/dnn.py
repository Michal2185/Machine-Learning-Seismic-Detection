from src.modules.training import FeatureInput, LearnerDetector, TorchLearner, TrainConfig

# hidden layer sizes giving about 1k / 10k / 100k parameters on 6 inputs
PRESETS = {"1k": (28, 28), "10k": (96, 96), "100k": (310, 310)}


def mlp_param_count(hidden, n_in: int = 6, n_out: int = 1) -> int:
    """Weights + biases of a fully connected network n_in -> hidden... -> n_out."""

    sizes = [n_in, *hidden, n_out]

    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def make_mlp(hidden=(96, 96), n_in: int = 6, dropout: float = 0.0):
    """Small ReLU MLP on the 6 window features; one logit per window."""

    import torch.nn as nn

    layers, prev = [], n_in

    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, 1))

    return nn.Sequential(*layers)


def build_dnn_detector(
    name: str,
    normalizer,
    fs: float,
    size: str = "10k",
    config: TrainConfig | None = None,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.0,
    device: str = "cpu",
) -> LearnerDetector:
    """
    DNN on the 6 features of the normalised window (log + standardised per
    fold), trained by the harness. `normalizer` carries the variant
    (N0 / N1 / N2 / N3); size is a key of PRESETS.
    """

    hidden = PRESETS[size]

    learner = TorchLearner(
        lambda: make_mlp(hidden, dropout=dropout),
        lr=lr, weight_decay=weight_decay, device=device,
    )

    return LearnerDetector(name, learner, FeatureInput(normalizer, fs), config or TrainConfig())