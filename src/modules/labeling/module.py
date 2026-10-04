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
    Vectorised labeling core for ONE event (1-D arrays, seconds).

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

        1 = a catalogue onset falls inside the window
       -1 = ambiguous (pre-onset margin or coda) -> ignore
        0 = clean background

    event_groups (optional):
        primary evid -> ALL catalogue events on that waveform, from
        WaveformGrouper. When several events share one waveform, a
        window is labeled against all of them (positive if any onset
        is inside, ignore if inside any ignore zone). Without it, each
        window is labeled against its own event only (old behaviour).
    """

    def __init__(
        self,
        config: LabelConfig | None = None,
        event_groups: dict | None = None,
    ):
        self.config = config or LabelConfig()
        self.event_groups = event_groups or {}

    # ------------------------------------------------------------------

    def _coda_for(self, event) -> float:
        return self.config.coda_duration_s.get(
            event.event_type,
            self.config.default_coda_s,
        )

    def _events_of(self, event) -> list:
        return self.event_groups.get(event.evid, [event])

    def _label_group(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        events: list,
    ) -> np.ndarray:

        positive = np.zeros(starts.shape, dtype=bool)
        ignore = np.zeros(starts.shape, dtype=bool)

        for e in events:

            y = label_array(
                starts,
                ends,
                np.full(starts.shape, e.time_rel),
                np.full(starts.shape, self._coda_for(e)),
                self.config.pre_onset_margin_s,
            )

            positive |= y == POSITIVE
            ignore |= y == IGNORE

        labels = np.full(starts.shape, NEGATIVE, dtype=np.int8)
        labels[ignore] = IGNORE
        labels[positive] = POSITIVE

        return labels

    # ------------------------------------------------------------------

    def label(
        self,
        window: WindowRecord,
    ) -> LabeledWindowRecord:

        label = self._label_group(
            np.array([window.start_time]),
            np.array([window.end_time]),
            self._events_of(window.event),
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

        by_waveform = defaultdict(list)

        for i, w in enumerate(windows):
            by_waveform[w.event.evid].append(i)

        labels = np.empty(len(windows), dtype=np.int8)

        for indices in by_waveform.values():

            idx = np.asarray(indices)

            starts = np.fromiter(
                (windows[i].start_time for i in idx), float, len(idx)
            )
            ends = np.fromiter(
                (windows[i].end_time for i in idx), float, len(idx)
            )

            labels[idx] = self._label_group(
                starts,
                ends,
                self._events_of(windows[idx[0]].event),
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
        event_groups: dict | None = None,
    ) -> LabelingSummary:
        """
        Positives are attributed to the TYPE OF THE EVENT whose onset
        they contain; ignore / negative windows to the type of the
        waveform's primary event.
        """

        event_groups = event_groups or {}

        per_type = defaultdict(lambda: {"pos": 0, "neg": 0, "ign": 0})
        positives_per_event = defaultdict(int)
        primaries = {}

        pos = neg = ign = 0

        for w in labeled:

            primaries[w.event.evid] = w.event

            if w.label == POSITIVE:

                pos += 1

                for e in event_groups.get(w.event.evid, [w.event]):
                    if w.start_time <= e.time_rel < w.end_time:
                        per_type[e.event_type]["pos"] += 1
                        positives_per_event[e.evid] += 1

            elif w.label == IGNORE:
                ign += 1
                per_type[w.event.event_type]["ign"] += 1

            else:
                neg += 1
                per_type[w.event.event_type]["neg"] += 1

        all_events = {
            e.evid: e
            for evid, ev in primaries.items()
            for e in event_groups.get(evid, [ev])
        }

        return LabelingSummary(
            total=len(labeled),
            positive=pos,
            negative=neg,
            ignored=ign,
            per_type=dict(per_type),
            events_without_positive=sorted(
                e for e in all_events if positives_per_event[e] == 0
            ),
            n_events=len(all_events),
            n_waveforms=len(primaries),
        )

    @staticmethod
    def print_summary(
        labeled: list[LabeledWindowRecord],
        event_groups: dict | None = None,
    ) -> LabelingSummary:

        s = Labeling.summarize(labeled, event_groups)

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
        print(
            f"Catalogue events: {s.n_events} "
            f"on {s.n_waveforms} waveforms"
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