from dataclasses import dataclass

from src.modules.labeling import LabeledWindowRecord


@dataclass
class SamplingResult:
    windows: list[LabeledWindowRecord]

    epoch: int
    n_positive: int
    n_negative: int               # sampled negatives
    n_negative_available: int     # negatives in the pool
    n_ignored_dropped: int        # label -1 windows that were excluded

    @property
    def negatives_per_positive(self) -> float:
        return self.n_negative / self.n_positive if self.n_positive else 0.0