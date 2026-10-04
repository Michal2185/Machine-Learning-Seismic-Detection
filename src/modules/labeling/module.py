from collections import defaultdict

import numpy as np

from src.modules.windowing import WindowRecord

from .types import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    LabelConfig,
    LabeledWindowRecord,
    LabelingSummary,
)


def label_array(
    starts: np.ndarray,
    ends: np.ndarray,
    arrivals: np.ndarray,
    coda_s: np.ndarray,
    pre_onset_margin_s: float,
) -> np.ndarray:
    """
    Vectorised labeling core (1-D arrays of equal length, seconds).

    Precedence: POSITIVE > IGNORE > NEGATIVE.
    """

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    arrivals = np.asarray(arrivals, dtype=np.float64)
    coda_s = np.asarray(coda_s, dtype=np.float64)

    labels = np.full(starts.shape, NEGATIVE, dtype=np.int8)

    contains = (starts <= arrivals) & (arrivals < ends)

    pre_onset = (
        (ends <= arrivals)
        & (ends > arrivals - pre_onset_margin_s)
    )

    coda = (
        (starts > arrivals)
        & (starts < arrivals + coda_s)
    )

    labels[pre_onset | coda] = IGNORE
    labels[contains] = POSITIVE

    return labels


class Labeling:
    """
    Assigns three-class labels to waveform windows.

        1 = catalogue onset falls inside the window
       -1 = ambiguous (pre-onset margin or coda) -> ignore
        0 = clean background
    """

    def __init__(self, config: LabelConfig | None = None):
        self.config = config or LabelConfig()

    def _coda_for(self, window: WindowRecord) -> float:
        return self.config.coda_duration_s.get(
            window.event.event_type,
            self.config.default_coda_s,
        )

    def label(
        self,
        window: WindowRecord,
    ) -> LabeledWindowRecord:

        label = label_array(
            np.array([window.start_time]),
            np.array([window.end_time]),
            np.array([window.event.time_rel]),
            np.array([self._coda_for(window)]),
            self.config.pre_onset_margin_s,
        )[0]

        return LabeledWindowRecord(
            window=window,
            label=int(label),
        )

    def run(
        self,
        windows: list[WindowRecord],
    ) -> list[LabeledWindowRecord]:

        if not windows:
            return []

        labels = label_array(
            np.fromiter((w.start_time for w in windows), float, len(windows)),
            np.fromiter((w.end_time for w in windows), float, len(windows)),
            np.fromiter(
                (w.event.time_rel for w in windows), float, len(windows)
            ),
            np.fromiter(
                (self._coda_for(w) for w in windows), float, len(windows)
            ),
            self.config.pre_onset_margin_s,
        )

        return [
            LabeledWindowRecord(window=w, label=int(y))
            for w, y in zip(windows, labels)
        ]

    # ------------------------------------------------------------------
    # Helpers for downstream modules
    # ------------------------------------------------------------------

    @staticmethod
    def trainable(
        labeled: list[LabeledWindowRecord],
    ) -> list[LabeledWindowRecord]:
        """Windows usable for training / validation loss (labels 0 and 1)."""
        return [w for w in labeled if w.label != IGNORE]

    @staticmethod
    def summarize(
        labeled: list[LabeledWindowRecord],
    ) -> LabelingSummary:

        per_type = defaultdict(lambda: {"pos": 0, "neg": 0, "ign": 0})
        positives_per_event = defaultdict(int)

        pos = neg = ign = 0

        for w in labeled:
            key = {POSITIVE: "pos", NEGATIVE: "neg", IGNORE: "ign"}[w.label]
            per_type[w.event.event_type][key] += 1
            positives_per_event[w.event.evid] += int(w.label == POSITIVE)

            if w.label == POSITIVE:
                pos += 1
            elif w.label == IGNORE:
                ign += 1
            else:
                neg += 1

        return LabelingSummary(
            total=len(labeled),
            positive=pos,
            negative=neg,
            ignored=ign,
            per_type=dict(per_type),
            events_without_positive=sorted(
                e for e, n in positives_per_event.items() if n == 0
            ),
        )

    @staticmethod
    def print_summary(labeled: list[LabeledWindowRecord]) -> LabelingSummary:

        s = Labeling.summarize(labeled)

        print("=" * 60)
        print("LABELING SUMMARY")
        print("=" * 60)
        print(f"Total windows: {s.total}")
        print(f"Negative (0):  {s.negative}")
        print(f"Positive (1):  {s.positive}")
        print(
            f"Ignore (-1):   {s.ignored} "
            f"({100 * s.ignored_fraction:.2f}% of windows)"
        )
        print()

        for t, c in sorted(s.per_type.items()):
            print(
                f"{t:12s} pos={c['pos']:5d} "
                f"ign={c['ign']:6d} neg={c['neg']:8d}"
            )

        if s.events_without_positive:
            print()
            print(
                f"WARNING: {len(s.events_without_positive)} events have no "
                f"positive window: {s.events_without_positive[:10]}"
            )

        return s