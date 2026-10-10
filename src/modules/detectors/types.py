import numpy as np


class Detector:
    """
    Common interface of every detector in the comparison.

    score(windows, streams_by_id) returns ONE score per window (same
    order); higher = more likely an event. The evaluator turns the scores
    into alarms, so a detector never sees thresholds, labels of the
    evaluation data or the other folds.

    windows        : LabeledWindowRecord list (uses .data, .end_time,
                     .event.evid and window.end_sample only)
    streams_by_id  : waveform id -> EvalStream (continuous detectors use
                     it, window-based networks may ignore it)
    """

    name: str = "detector"

    def fit(self, train_windows) -> "Detector":
        """Learn from TRAINING windows only (no-op for classical detectors)."""
        return self

    def score(self, windows, streams_by_id) -> np.ndarray:
        raise NotImplementedError