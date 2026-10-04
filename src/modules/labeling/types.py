from dataclasses import dataclass, field

import numpy as np

from src.modules.windowing import WindowRecord


# Label values
POSITIVE = 1   # catalogue onset falls inside the window
NEGATIVE = 0   # clean background
IGNORE = -1    # ambiguous: pre-onset margin or coda; excluded from
               # training loss, threshold selection and false-alarm counts


@dataclass(frozen=True)
class LabelConfig:
    """
    pre_onset_margin_s:
        Catalogue onsets are given to the minute (+-60 s). Windows that
        END within this margin before the catalogue onset may already
        contain the true onset, so they are ignored instead of negative.

    coda_duration_s / default_coda_s:
        Windows STARTING after the onset and within this duration are
        coda (real signal) and are ignored instead of negative.
        PLACEHOLDER VALUES: tune from the envelope decay of training
        events (time until the smoothed envelope returns to background).
    """

    pre_onset_margin_s: float = 60.0
    coda_duration_s: dict = field(
        default_factory=lambda: {
            "impact_mq": 1200.0,
            "shallow_mq": 1800.0,
            "deep_mq": 1200.0,
        }
    )
    default_coda_s: float = 1200.0


@dataclass
class LabeledWindowRecord:
    window: WindowRecord
    label: int

    @property
    def data(self) -> np.ndarray:
        return self.window.data

    @property
    def start_time(self) -> float:
        return self.window.start_time

    @property
    def end_time(self) -> float:
        return self.window.end_time

    @property
    def event(self):
        return self.window.event

    @property
    def is_positive(self) -> bool:
        return self.label == POSITIVE

    @property
    def is_ignored(self) -> bool:
        return self.label == IGNORE


@dataclass
class LabelingSummary:
    total: int
    positive: int
    negative: int
    ignored: int
    per_type: dict        # event_type -> {"pos": int, "neg": int, "ign": int}
    events_without_positive: list

    @property
    def ignored_fraction(self) -> float:
        return self.ignored / self.total if self.total else 0.0