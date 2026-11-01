from src.modules.training import LearnerDetector, RawWindowInput, TorchLearner, TrainConfig

# Hidden size per budget, chosen so that GRU and CfC are within ~1 % of each other.
# F = input features per step: W (raw window) F = 1, E (envelope sequence) F = 4.
# The CfC has NO backbone (backbone_layers=0): a backbone of 128 units alone costs
# 2 304 + 4 x 2 064 parameters, i.e. 5.4k parameters at h = 8 (ncps default, see probe).
PRESETS = {
    1: {"1k": {"gru": 17, "cfc": 15}, "10k": {"gru": 56, "cfc": 49}, "100k": {"gru": 181, "cfc": 157}},
    4: {"1k": {"gru": 15, "cfc": 13}, "10k": {"gru": 55, "cfc": 48}, "100k": {"gru": 180, "cfc": 156}},
}


def gru_param_count(n_in: int, hidden: int) -> int:
    """nn.GRU (one layer, two bias vectors) + Linear(hidden, 1) head."""
    return 3 * (n_in * hidden + hidden * hidden + 2 * hidden) + hidden + 1


def cfc_param_count(n_in: int, hidden: int) -> int:
    """ncps CfC, mode 'default', backbone_layers=0: ff1, ff2, time_a, time_b = 4 x Linear(F+h, h), plus head."""
    return 4 * (hidden * (n_in + hidden) + hidden) + hidden + 1


def gru_macs(n_in: int, hidden: int, steps: int) -> int:
    """Gate matmuls + head (elementwise gating not counted)."""
    return steps * 3 * (n_in * hidden + hidden * hidden) + hidden


def cfc_macs(n_in: int, hidden: int, steps: int) -> int:
    """Four Linear(F+h, h) per step + head (tanh / sigmoid / interpolation not counted)."""
    return steps * 4 * (n_in + hidden) * hidden + hidden


def make_gru(n_in: int, hidden: int):
    """Input (B, T, F), output one logit (B, 1) from the LAST hidden state."""

    import torch.nn as nn

    class GruClassifier(nn.Module):

        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, h_n = self.rnn(x)
            return self.head(h_n[-1])

    return GruClassifier()


def make_cfc(n_in: int, hidden: int, dt: float | None = None):
    """
    Input (B, T, F), output one logit (B, 1) from the final CfC state.
    dt = None -> ncps default (time step 1 for every step); dt = 1/fs -> physical time step.
    """

    import torch
    import torch.nn as nn
    from ncps.torch import CfC

    class CfcClassifier(nn.Module):

        state_floats = hidden           # read by count_model_resources (see resources patch)

        def __init__(self):
            super().__init__()
            self.rnn = CfC(n_in, hidden, return_sequences=False, batch_first=True, backbone_layers=0)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            ts = None
            if dt is not None:
                # (B, T, h) view, no copy. ncps indexes timespans[:, t] and the cell multiplies it with
                # (B, h) tensors: a (B,) vector after squeeze() fails to broadcast when B != h.
                ts = torch.full((x.shape[0], x.shape[1], 1), float(dt), dtype=x.dtype, device=x.device)
                ts = ts.expand(-1, -1, hidden)
            out, _ = self.rnn(x, timespans=ts)
            return self.head(out)

    return CfcClassifier()


def build_recurrent_detector(
    name: str,
    kind: str,                      # "gru" | "cfc"
    normalizer,
    size: str = "10k",
    config: TrainConfig | None = None,
    dt: float | None = None,        # CfC only
    lr: float = 1e-3,
    weight_decay: float = 1e-3,     # same defaults as the CNN
    device: str = "cpu",
) -> LearnerDetector:
    """GRU / CfC on the normalised raw window, variant W (398 steps, decision at the last step)."""

    hidden = PRESETS[1][size][kind]

    if kind == "gru":
        factory = lambda: make_gru(1, hidden)
    elif kind == "cfc":
        factory = lambda: make_cfc(1, hidden, dt)
    else:
        raise ValueError(kind)

    learner = TorchLearner(factory, lr=lr, weight_decay=weight_decay, device=device)

    return LearnerDetector(name, learner, RawWindowInput(normalizer, "NTC"), config or TrainConfig())