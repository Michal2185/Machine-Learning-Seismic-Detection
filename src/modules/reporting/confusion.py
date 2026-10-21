"""
Confusion matrices and derived metrics at one threshold.

Two levels, because a detector is judged on events but trained and scored on windows:

  window level  every scored 60 s window with label 1 (event signal) or 0 (background);
                label -1 (ignore) windows are dropped; the window counts as 'alarm' when
                score >= threshold. Windows overlap by 50 s, so counts are not independent,
                and the threshold was chosen for the EVENT-level false-alarm budget, not to
                optimise windows.
  event level   TP = events detected, FN = events missed, FP = false alarms (alarm episodes
                outside every detect / ignore interval / gap). There is no natural TN for a
                continuous stream, so specificity is only reported at window level.

Metric names: 'false negative rate' = FN / (TP + FN) = 1 - recall (the thesis term
'1 - TP/(TP+FN)'; Reviewer 1, point 7).
"""

from dataclasses import dataclass

import numpy as np


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else float("nan")


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int | None          # None when undefined (event level)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + (self.tn or 0)

    @property
    def prevalence(self) -> float:
        return float("nan") if self.tn is None else _div(self.tp + self.fn, self.n)

    @property
    def recall(self) -> float:                       # sensitivity, true positive rate
        return _div(self.tp, self.tp + self.fn)

    @property
    def fnr(self) -> float:                          # false negative rate
        return _div(self.fn, self.tp + self.fn)

    @property
    def specificity(self) -> float:                  # true negative rate
        return float("nan") if self.tn is None else _div(self.tn, self.tn + self.fp)

    @property
    def fpr(self) -> float:                          # false positive rate
        return float("nan") if self.tn is None else _div(self.fp, self.fp + self.tn)

    @property
    def precision(self) -> float:
        return _div(self.tp, self.tp + self.fp)

    @property
    def f1(self) -> float:
        return _div(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def balanced_accuracy(self) -> float:
        r, s = self.recall, self.specificity
        return float("nan") if np.isnan(r) or np.isnan(s) else 0.5 * (r + s)

    @property
    def mcc(self) -> float:
        if self.tn is None:
            return float("nan")
        d = (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        return float("nan") if d <= 0 else (self.tp * self.tn - self.fp * self.fn) / float(np.sqrt(float(d)))

    def __add__(self, o: "Confusion") -> "Confusion":
        tn = None if (self.tn is None or o.tn is None) else self.tn + o.tn
        return Confusion(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn, tn)

    def as_dict(self) -> dict:
        return dict(tp=self.tp, fp=self.fp, fn=self.fn, tn=self.tn, recall=self.recall, fnr=self.fnr,
                    specificity=self.specificity, fpr=self.fpr, precision=self.precision, f1=self.f1,
                    balanced_accuracy=self.balanced_accuracy, mcc=self.mcc)


def window_confusion(labels, scores, threshold: float) -> Confusion:
    """labels 1 / 0 / -1 (ignore, dropped); a window is positive when score >= threshold."""

    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)

    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")

    keep = (y == 0) | (y == 1)
    y, pred = y[keep] == 1, s[keep] >= threshold

    return Confusion(
        tp=int((y & pred).sum()), fp=int((~y & pred).sum()),
        fn=int((y & ~pred).sum()), tn=int((~y & ~pred).sum()),
    )


def event_confusion(report) -> Confusion:
    """Event level from an AlarmReport: TP detected, FN missed, FP false alarms, TN undefined (None)."""

    return Confusion(tp=report.n_detected, fp=report.n_false,
                     fn=len(report.outcomes) - report.n_detected, tn=None)


def event_summary(report) -> dict:
    """Event-level numbers of one AlarmReport (or pooled: see pool_reports)."""

    c = event_confusion(report)

    return dict(
        events=len(report.outcomes), detected=c.tp, missed=c.fn, false_alarms=c.fp,
        event_recall=c.recall, event_fnr=c.fnr, episode_precision=c.precision,
        fa_per_day=report.fa_per_day,
        duplicates=sum(1 for l in report.labels if l == "duplicate"),
        neutral=sum(1 for l in report.labels if l == "neutral"),
    )


def pool_event_summary(reports) -> dict:
    """Event-level numbers pooled over several waveforms (false alarms per day over total scored time)."""

    reports = list(reports)
    events = sum(len(r.outcomes) for r in reports)
    detected = sum(r.n_detected for r in reports)
    false_alarms = sum(r.n_false for r in reports)
    seconds = sum(r.scored_seconds for r in reports)
    c = Confusion(detected, false_alarms, events - detected, None)

    return dict(
        events=events, detected=detected, missed=events - detected, false_alarms=false_alarms,
        event_recall=c.recall, event_fnr=c.fnr, episode_precision=c.precision,
        fa_per_day=float("nan") if seconds <= 0 else false_alarms / (seconds / 86400.0),
        duplicates=sum(1 for r in reports for l in r.labels if l == "duplicate"),
        neutral=sum(1 for r in reports for l in r.labels if l == "neutral"),
    )