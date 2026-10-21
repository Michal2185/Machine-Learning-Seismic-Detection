"""
Alarm logic for plots: turns window scores into alarms and compares them with the catalogue
arrival of each event, using the conventions of docs/DATA_BRANCH_HANDOFF.md:

  * alarm time = END time of the window that triggered it
  * a detector alarms at the first window with score >= threshold, then stays silent for
    `refractory_s` (the evaluator's default is 300 s)
  * an alarm inside an event's detect interval [arrival - 120 s, arrival + 600 s] detects it
    (the first one counts, later ones are neutral duplicates)
  * else an alarm inside an ignore interval or an interpolated gap is neutral
  * else it is a false alarm
This re-implements the rule for plotting; the evaluator stays the authority for reported numbers.
"""

from dataclasses import dataclass, field

import numpy as np


def extract_alarms(t_end, scores, threshold: float, refractory_s: float = 300.0) -> np.ndarray:
    """Alarm times (s) from window end times and scores (any order of input)."""

    t = np.asarray(t_end, dtype=float)
    s = np.asarray(scores, dtype=float)

    if t.shape != s.shape:
        raise ValueError("t_end and scores must have the same shape")

    alarms = []
    next_ok = -np.inf

    for i in np.argsort(t, kind="stable"):
        if s[i] >= threshold and t[i] >= next_ok:
            alarms.append(t[i])
            next_ok = t[i] + refractory_s

    return np.array(alarms, dtype=float)


def merge_intervals(intervals) -> list:

    merged = []

    for lo, hi in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    return [tuple(m) for m in merged]


def _inside(t: float, intervals) -> bool:
    return any(lo <= t <= hi for lo, hi in intervals)


@dataclass
class EventOutcome:
    evid: str
    event_type: str
    arrival_s: float
    onset_s: float | None
    detect: tuple
    detected: bool
    alarm_s: float | None = None

    @property
    def delay_arrival_s(self):
        return None if self.alarm_s is None else self.alarm_s - self.arrival_s

    @property
    def delay_onset_s(self):
        return None if (self.alarm_s is None or self.onset_s is None) else self.alarm_s - self.onset_s


@dataclass
class AlarmReport:
    alarm_s: np.ndarray
    labels: list                      # detection / duplicate / neutral / false (one per alarm)
    outcomes: list = field(default_factory=list)
    scored_seconds: float = 0.0

    @property
    def n_false(self) -> int:
        return sum(1 for l in self.labels if l == "false")

    @property
    def n_detected(self) -> int:
        return sum(1 for o in self.outcomes if o.detected)

    @property
    def fa_per_day(self) -> float:
        return float("nan") if self.scored_seconds <= 0 else self.n_false / (self.scored_seconds / 86400.0)


def scored_seconds(duration_s: float, events, gaps) -> float:
    """Stream length minus ignore intervals, interpolated gaps and detect intervals."""

    excluded = [(0.0, 0.0)]
    excluded += [tuple(e.ignore) for e in events] + [tuple(e.detect) for e in events] + list(gaps)

    clipped = [(max(0.0, lo), min(duration_s, hi)) for lo, hi in excluded]

    return float(duration_s - sum(hi - lo for lo, hi in merge_intervals(clipped)))


def classify_alarms(alarm_s, events, gaps, duration_s: float) -> AlarmReport:
    """events: objects with evid, event_type, arrival_s, onset_s, detect, ignore (EventTarget)."""

    alarms = np.sort(np.asarray(alarm_s, dtype=float))
    first = {e.evid: None for e in events}
    labels = []

    for a in alarms:

        hit = [e for e in events if e.detect[0] <= a <= e.detect[1]]

        if hit:
            e = hit[0]
            if first[e.evid] is None:
                first[e.evid] = float(a)
                labels.append("detection")
            else:
                labels.append("duplicate")

        elif any(_inside(a, [tuple(e.ignore)]) for e in events) or _inside(a, gaps):
            labels.append("neutral")

        else:
            labels.append("false")

    outcomes = [
        EventOutcome(
            evid=e.evid,
            event_type=getattr(e, "event_type", ""),
            arrival_s=float(e.arrival_s),
            onset_s=None if e.onset_s is None else float(e.onset_s),
            detect=tuple(e.detect),
            detected=first[e.evid] is not None,
            alarm_s=first[e.evid],
        )
        for e in events
    ]

    return AlarmReport(
        alarm_s=alarms,
        labels=labels,
        outcomes=outcomes,
        scored_seconds=scored_seconds(duration_s, events, gaps),
    )