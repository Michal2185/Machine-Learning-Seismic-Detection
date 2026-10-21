from dataclasses import dataclass, field

import numpy as np

# window labels (same values as src.modules.labeling)
POSITIVE = 1
NEGATIVE = 0
IGNORE = -1


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    patience: int = 10                 # epochs without validation improvement
    min_epochs: int = 1
    val_fraction: float = 0.2          # share of TRAINING waveforms held out for early stopping
    val_neg_stride: int = 5            # use every k-th background window of those waveforms
    negatives_per_positive: float = 5.0   # Sampler ratio (new draw every epoch)
    pos_weight: float | None = None    # loss weight of positives; None -> negatives_per_positive
    seed: int = 0
    score_batch: int = 8192            # windows per inference chunk

    def __post_init__(self):

        if self.epochs < 1 or self.batch_size < 1 or self.score_batch < 1:
            raise ValueError("epochs, batch_size and score_batch must be >= 1")

        if self.patience < 1 or self.min_epochs < 1 or self.val_neg_stride < 1:
            raise ValueError("patience, min_epochs and val_neg_stride must be >= 1")

        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")

        if self.negatives_per_positive <= 0:
            raise ValueError("negatives_per_positive must be > 0")

        if self.pos_weight is not None and self.pos_weight <= 0:
            raise ValueError("pos_weight must be > 0")

    @property
    def effective_pos_weight(self) -> float:
        return float(self.negatives_per_positive if self.pos_weight is None else self.pos_weight)


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    val_ap: float            # NaN without early stopping
    best: bool


@dataclass
class FitInfo:
    n_train_waveforms: int
    n_val_waveforms: int
    val_waveforms: list
    early_stopping: bool
    best_epoch: int
    best_val_ap: float
    stopped_early: bool
    history: list = field(default_factory=list)
    note: str = ""


@dataclass
class ResourceReport:
    """Static cost of one detector; see docs/HARNESS_HANDOFF.md for what is (not) counted."""

    params: int
    bytes_fp32: int
    bytes_int8: int
    macs_per_window: int            # multiply-accumulates of one forward pass over one window
    state_floats: int               # recurrent state kept between windows / steps
    input_ops: int = 0              # rough operation count of the feature / normalisation front end
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"params {self.params:,} | fp32 {self.bytes_fp32 / 1024:.1f} KiB | "
            f"int8 {self.bytes_int8 / 1024:.1f} KiB | MACs/window {self.macs_per_window:,} | "
            f"state {self.state_floats} floats | input ops ~{self.input_ops:,}"
        )