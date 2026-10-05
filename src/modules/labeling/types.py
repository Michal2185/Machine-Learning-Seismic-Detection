from dataclasses import dataclass, field

import numpy as np

from src.modules.windowing import WindowRecord


# Label values
POSITIVE = 1   # a catalogue onset falls inside the window
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
        PER-TYPE FALLBACK (placeholders). When Labeling gets an
        EventZoneTable, each event uses its own estimated coda instead,
        capped at max_coda_s.

    gap_ignore_fraction:
        A window that would be negative but lies at least this share
        inside an interpolated gap carries no information -> ignore.

    Onset anchoring (needs an EventZoneTable)
        The catalogue time is only good to about a minute and usually
        precedes the visible onset, so windows that merely contain the
        catalogue time often hold no signal. With anchoring, a window is
        POSITIVE if it contains the VISIBLE onset with at least
        min_signal_s of signal after it (and starts at most
        positive_span_s after the onset). Everything between
        (earliest plausible onset - pre_onset_margin_s) and the coda end
        that is not positive is IGNORE.
        Events without a usable visible onset (none found, or an offset
        outside onset_bounds) get NO positives; their whole onset region
        and coda are ignored, not negative.

    detect_before_s / detect_after_s:
        EVALUATION tolerance around the catalogue time: an alarm in
        [arrival - detect_before_s, arrival + detect_after_s] detects the
        event. Single source of truth for the evaluator; not used for
        window labels.
    """

    pre_onset_margin_s: float = 60.0
    max_coda_s: float = 21600.0          # cap on a per-event coda (6 h)
    gap_ignore_fraction: float = 0.5     # window share inside a gap -> ignore

    # --- onset anchoring (only active when Labeling gets an EventZoneTable)
    anchor_to_onset: bool = True
    min_signal_s: float = 20.0           # signal a positive window must hold
    onset_bounds: tuple = (-120.0, 600.0)  # usable visible-onset offsets (s)
    positive_span_s: float = 0.0         # extra window starts after the onset
                                         # that still count as positive

    # --- evaluation contract (used by the event-level evaluator)
    detect_before_s: float = 120.0       # alarm counts if >= arrival - 120 s
    detect_after_s: float = 600.0        # ... and <= arrival + 600 s
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
    n_events: int = 0     # catalogue events (incl. those sharing a waveform)
    n_waveforms: int = 0  # distinct waveforms

    @property
    def ignored_fraction(self) -> float:
        return self.ignored / self.total if self.total else 0.0