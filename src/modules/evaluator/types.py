from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalConfig:
    """
    refractory_s:
        After an alarm, threshold crossings are ignored for this long
        (one alarm per burst of activity; a long coda is not a stream of
        separate alarms).
    max_fa_per_day:
        False-alarm budget (per 24 SCORED hours) used to pick the
        operating threshold on validation data.
    """

    refractory_s: float = 300.0
    max_fa_per_day: float = 1.0
    n_thresholds: int = 100          # per half of the default threshold grid
    n_bootstrap: int = 1000
    seed: int = 0


@dataclass
class ScoreSeries:
    """
    Detector output on one stream. times are in seconds from the start of
    the file and mean "the detector knew this much": for a window model
    use the END time of each window.
    """

    waveform_id: str
    times: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.scores = np.asarray(self.scores, dtype=np.float64)

        if self.times.shape != self.scores.shape:
            raise ValueError("times and scores must have the same shape")

        if len(self.times) > 1 and np.any(np.diff(self.times) < 0):
            raise ValueError("times must be sorted")


@dataclass
class EventOutcome:
    evid: str
    event_type: str
    waveform_id: str
    has_positive: bool
    peak_ratio: float | None
    detected: bool
    alarm_s: float                  # first alarm inside the detect interval
    delay_s: float                  # alarm - catalogue time (nan if missed)
    delay_onset_s: float            # alarm - visible onset (nan if none)


@dataclass
class StreamResult:
    waveform_id: str
    outcomes: list
    alarms: np.ndarray              # all alarm times
    n_false_alarms: int
    n_neutral: int                  # alarms inside ignore zones / gaps
    background_hours: float         # scored time outside ignore/gap/detect


def in_group(o: EventOutcome, group: str) -> bool:

    if group == "all":
        return True
    if group == "usable":
        return o.has_positive
    if group == "weak":
        return not o.has_positive

    return o.event_type == group


@dataclass
class EvalResult:
    threshold: float
    streams: list

    @property
    def outcomes(self) -> list:
        return [o for s in self.streams for o in s.outcomes]

    def n_events(self, group: str = "all") -> int:
        return sum(in_group(o, group) for o in self.outcomes)

    def recall(self, group: str = "all") -> float:
        sel = [o for o in self.outcomes if in_group(o, group)]
        return float(np.mean([o.detected for o in sel])) if sel else float("nan")

    @property
    def n_false_alarms(self) -> int:
        return sum(s.n_false_alarms for s in self.streams)

    @property
    def n_alarms(self) -> int:
        return sum(len(s.alarms) for s in self.streams)

    @property
    def background_hours(self) -> float:
        return sum(s.background_hours for s in self.streams)

    @property
    def fa_per_day(self) -> float:
        h = self.background_hours
        return 24.0 * self.n_false_alarms / h if h > 0 else float("nan")

    def delays(self, group: str = "all", reference: str = "arrival") -> np.ndarray:

        key = "delay_s" if reference == "arrival" else "delay_onset_s"

        v = np.array(
            [getattr(o, key) for o in self.outcomes
             if in_group(o, group) and o.detected],
            dtype=float,
        )

        return v[~np.isnan(v)]


@dataclass
class Curve:
    thresholds: np.ndarray
    recall_all: np.ndarray
    recall_usable: np.ndarray
    recall_weak: np.ndarray
    fa_per_day: np.ndarray
    n_alarms: np.ndarray
    background_hours: float

    def recall(self, group: str = "all") -> np.ndarray:
        return {
            "all": self.recall_all,
            "usable": self.recall_usable,
            "weak": self.recall_weak,
        }[group]