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


# ----------------------------------------------------------------------
# vectorised cores (ONE event, 1-D arrays of window start / end in seconds)
# ----------------------------------------------------------------------

def label_array(
    starts: np.ndarray,
    ends: np.ndarray,
    arrivals: np.ndarray,
    coda_s: np.ndarray,
    pre_onset_margin_s: float,
) -> np.ndarray:
    """
    Catalogue-anchored labeling (old behaviour).

    POSITIVE: window contains the catalogue time.
    IGNORE:   windows ending within the pre-onset margin, and windows
              starting after the onset within the coda.
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


def label_array_anchored(
    starts: np.ndarray,
    ends: np.ndarray,
    arrival: float,
    coda_s: float,
    onset: float,
    config: LabelConfig,
) -> np.ndarray:
    """
    Onset-anchored labeling for ONE event.

    onset: absolute time (s) of the VISIBLE onset, or NaN if the event has
           no usable visible onset (then it gets no positives).

    IGNORE interval: windows ending after (earliest plausible onset -
    pre_onset_margin_s) and starting before (arrival + coda_s).
    POSITIVE (inside it): window starts at most positive_span_s after the
    onset and ends at least min_signal_s after the onset.
    """

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)

    usable = bool(np.isfinite(onset))

    t_early = min(arrival, onset) if usable else arrival

    labels = np.full(starts.shape, NEGATIVE, dtype=np.int8)

    in_zone = (
        (ends > t_early - config.pre_onset_margin_s)
        & (starts < arrival + coda_s)
    )

    labels[in_zone] = IGNORE

    if usable:
        positive = (
            (starts <= onset + config.positive_span_s)
            & (ends >= onset + config.min_signal_s)
            & (starts < arrival + coda_s)
        )
        labels[positive] = POSITIVE

    return labels


# ----------------------------------------------------------------------
# Labeling
# ----------------------------------------------------------------------

class Labeling:
    """
    Assigns three-class labels to waveform windows.

        1 = positive
       -1 = ignore (ambiguous / coda / gap) -> excluded from training
            loss, threshold selection and false-alarm counting
        0 = clean background

    event_groups (optional)
        primary evid -> ALL catalogue events on that waveform
        (WaveformGrouper). A window is labeled against all of them:
        positive if positive for any, else ignore if ignore for any.

    zones (optional)
        EventZoneTable (ZoneEstimator). Enables
          * per-event coda lengths (capped at config.max_coda_s)
          * ignoring negative windows mostly inside interpolated gaps
          * onset anchoring (config.anchor_to_onset): positives are
            windows holding >= min_signal_s of VISIBLE signal after the
            visible onset; events without a usable onset get no positives.
        Without zones: catalogue-anchored positives and per-type coda
        values (old behaviour).
    """

    def __init__(
        self,
        config: LabelConfig | None = None,
        event_groups: dict | None = None,
        zones=None,
    ):
        self.config = config or LabelConfig()
        self.event_groups = event_groups or {}
        self.zones = zones
        self.gap_ignored = 0

    # ------------------------------------------------------------------

    def _coda_for(self, event) -> float:

        if self.zones is not None:
            z = self.zones.get(event.evid)
            if z is not None and np.isfinite(z.coda_end_s):
                return float(min(z.coda_end_s, self.config.max_coda_s))

        return self.config.coda_duration_s.get(
            event.event_type,
            self.config.default_coda_s,
        )

    def _events_of(self, event) -> list:
        return self.event_groups.get(event.evid, [event])

    def anchor(self, event):
        """
        None   -> catalogue-anchored (no zone information / anchoring off)
        NaN    -> anchored, but no usable visible onset (no positives)
        float  -> absolute time (s) of the visible onset
        """

        if self.zones is None or not self.config.anchor_to_onset:
            return None

        z = self.zones.get(event.evid)

        if z is None:
            return None

        lo, hi = self.config.onset_bounds

        if z.onset_status == "ok" and lo <= z.onset_offset_s <= hi:
            return float(event.time_rel + z.onset_offset_s)

        return float("nan")

    def positive_rule(self, event):
        """
        (reference time, min signal after it, span) describing which
        windows are positive for this event, or None if it has none.
        """

        a = self.anchor(event)

        if a is None:
            return event.time_rel, 1e-9, 0.0

        if not np.isfinite(a):
            return None

        return a, self.config.min_signal_s, self.config.positive_span_s

    # ------------------------------------------------------------------
    # Time intervals consistent with the window labels (for the evaluator)
    # ------------------------------------------------------------------

    def detect_interval(self, event) -> tuple:
        """
        Evaluation tolerance: an alarm inside this interval detects the
        event (anchored to the CATALOGUE time, independent of onset
        anchoring).
        """

        t = float(event.time_rel)

        return (
            t - self.config.detect_before_s,
            t + self.config.detect_after_s,
        )

    def ignore_interval(self, event) -> tuple:
        """
        Onset region + coda of the event: no false alarms are counted
        inside it. Windows ending after the start and starting before the
        end of this interval are positive or ignore, never negative.
        """

        arrival = float(event.time_rel)
        a = self.anchor(event)

        t_early = min(arrival, a) if (a is not None and np.isfinite(a)) else arrival

        return (
            t_early - self.config.pre_onset_margin_s,
            arrival + self._coda_for(event),
        )

    # ------------------------------------------------------------------

    def _label_event(self, starts, ends, event) -> np.ndarray:

        a = self.anchor(event)
        coda = self._coda_for(event)

        if a is None:
            return label_array(
                starts,
                ends,
                np.full(starts.shape, event.time_rel),
                np.full(starts.shape, coda),
                self.config.pre_onset_margin_s,
            )

        return label_array_anchored(
            starts, ends, float(event.time_rel), coda, a, self.config
        )

    def _gap_fraction(self, starts, ends, waveform_id) -> np.ndarray:
        """Share of each window lying inside interpolated gaps."""

        frac = np.zeros(starts.shape, dtype=np.float64)

        if self.zones is None:
            return frac

        length = np.maximum(ends - starts, 1e-9)

        for a, b in self.zones.gaps.get(waveform_id, []):
            overlap = np.clip(
                np.minimum(ends, b) - np.maximum(starts, a), 0.0, None
            )
            frac += overlap / length

        return np.minimum(frac, 1.0)

    def _label_group(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        events: list,
        waveform_id: str | None = None,
    ) -> np.ndarray:

        positive = np.zeros(starts.shape, dtype=bool)
        ignore = np.zeros(starts.shape, dtype=bool)

        for e in events:
            y = self._label_event(starts, ends, e)
            positive |= y == POSITIVE
            ignore |= y == IGNORE

        labels = np.full(starts.shape, NEGATIVE, dtype=np.int8)
        labels[ignore] = IGNORE
        labels[positive] = POSITIVE

        # negative windows mostly inside an interpolated gap carry no
        # information (positives are never overridden)
        if self.zones is not None and waveform_id is not None:
            in_gap = (
                (self._gap_fraction(starts, ends, waveform_id)
                 >= self.config.gap_ignore_fraction)
                & (labels == NEGATIVE)
            )
            self.gap_ignored += int(in_gap.sum())
            labels[in_gap] = IGNORE

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
            window.event.evid,
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

        self.gap_ignored = 0

        by_waveform = defaultdict(list)

        for i, w in enumerate(windows):
            by_waveform[w.event.evid].append(i)

        labels = np.empty(len(windows), dtype=np.int8)

        for waveform_id, indices in by_waveform.items():

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
                waveform_id,
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
        labeling: "Labeling | None" = None,
    ) -> LabelingSummary:
        """
        Positives are attributed to the type of the event whose positive
        rule they satisfy (pass the Labeling instance that produced the
        labels so onset anchoring is respected); ignore / negative windows
        to the type of the waveform's primary event.
        """

        event_groups = event_groups or {}

        def rule_of(e):
            if labeling is None:
                return e.time_rel, 1e-9, 0.0
            return labeling.positive_rule(e)

        per_type = defaultdict(lambda: {"pos": 0, "neg": 0, "ign": 0})
        positives_per_event = defaultdict(int)
        primaries = {}

        pos = neg = ign = 0

        for w in labeled:

            primaries[w.event.evid] = w.event

            if w.label == POSITIVE:

                pos += 1

                for e in event_groups.get(w.event.evid, [w.event]):
                    rule = rule_of(e)
                    if rule is None:
                        continue
                    ref, min_sig, span = rule
                    if (
                        w.start_time <= ref + span
                        and w.end_time >= ref + min_sig
                    ):
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
        labeling: "Labeling | None" = None,
    ) -> LabelingSummary:

        s = Labeling.summarize(labeled, event_groups, labeling)

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
                f"Events without a positive window: "
                f"{len(s.events_without_positive)} "
                f"{s.events_without_positive[:10]}"
                f"{' ...' if len(s.events_without_positive) > 10 else ''}"
            )

        return s