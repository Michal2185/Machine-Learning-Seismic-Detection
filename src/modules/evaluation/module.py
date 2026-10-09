from collections import defaultdict

import numpy as np

from .types import (
    Curve,
    EvalConfig,
    EvalResult,
    EventOutcome,
    ScoreSeries,
    StreamResult,
    in_group,
)


# ----------------------------------------------------------------------
# interval helpers
# ----------------------------------------------------------------------

def merge_intervals(intervals) -> np.ndarray:
    """Union of (start, end) pairs as a sorted (n, 2) array."""

    iv = sorted(
        (float(a), float(b)) for a, b in intervals if b > a
    )

    out = []

    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])

    return np.array(out, dtype=np.float64).reshape(-1, 2)


def contains(merged: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Which times fall inside the merged intervals (closed)."""

    t = np.asarray(t, dtype=np.float64)
    res = np.zeros(t.shape, dtype=bool)

    if len(merged) == 0 or len(t) == 0:
        return res

    i = np.searchsorted(merged[:, 0], t, side="right") - 1
    ok = i >= 0
    res[ok] = t[ok] <= merged[i[ok], 1]

    return res


def union_length(merged: np.ndarray, duration: float) -> float:
    """Total length of the merged intervals clipped to [0, duration]."""

    if len(merged) == 0:
        return 0.0

    a = np.clip(merged[:, 0], 0.0, duration)
    b = np.clip(merged[:, 1], 0.0, duration)

    return float(np.sum(b - a))


# ----------------------------------------------------------------------
# helpers for detectors
# ----------------------------------------------------------------------

def build_series(windows, scores) -> dict:
    """
    waveform id -> ScoreSeries from labeled windows and one score per
    window (same order). Time = window END time.
    """

    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) != len(windows):
        raise ValueError("need exactly one score per window")

    by = defaultdict(list)

    for w, s in zip(windows, scores):
        by[w.event.evid].append((w.end_time, s))

    out = {}

    for wid, items in by.items():
        items.sort(key=lambda t: t[0])
        out[wid] = ScoreSeries(
            wid,
            np.array([t for t, _ in items]),
            np.array([s for _, s in items]),
        )

    return out


def average_precision(y_true, y_score) -> float:
    """Window-level PR-AUC (average precision); y_true in {0, 1}."""

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)

    if y.sum() == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y = y[order]

    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)

    return float(np.sum(precision * y) / y.sum())


def pool_results(results: list) -> EvalResult:
    """Combine results of different folds (each at its own threshold)."""

    return EvalResult(
        threshold=float("nan"),
        streams=[s for r in results for s in r.streams],
    )


# ----------------------------------------------------------------------
# evaluator
# ----------------------------------------------------------------------

class Evaluator:
    """
    Event-level evaluation of any detector that produces a score per time.

    Alarm  : first threshold crossing, then `refractory_s` of silence.
    Each alarm is classified in this order:
      1. inside the detect interval of an event -> detects that event
         (later alarms in the same interval are neutral duplicates)
      2. inside an ignore interval or an interpolated gap -> neutral
      3. otherwise -> false alarm
    False alarms are normalised per 24 hours of SCORED time: stream length
    minus ignore intervals, gaps and detect intervals.
    """

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self._geometry = {}

    # ------------------------------------------------------------------

    def geometry(self, stream):

        g = self._geometry.get(stream.waveform_id)

        if g is not None:
            return g

        duration = len(stream.data) / stream.sampling_rate

        neutral = merge_intervals(
            [t.ignore for t in stream.events] + list(stream.gaps)
        )

        everything = merge_intervals(
            [t.ignore for t in stream.events]
            + [t.detect for t in stream.events]
            + list(stream.gaps)
        )

        hours = (duration - union_length(everything, duration)) / 3600.0

        g = (duration, neutral, hours)
        self._geometry[stream.waveform_id] = g

        return g

    def alarms(self, series: ScoreSeries, threshold: float) -> np.ndarray:

        idx = np.flatnonzero(series.scores >= threshold)

        if len(idx) == 0:
            return np.empty(0, dtype=np.float64)

        te = series.times[idx]
        out = []
        i = 0

        while i < len(te):
            out.append(te[i])
            i = int(np.searchsorted(
                te, te[i] + self.config.refractory_s, side="left"
            ))

        return np.array(out, dtype=np.float64)

    # ------------------------------------------------------------------

    def score_stream(self, stream, series: ScoreSeries, threshold: float):

        _, neutral, hours = self.geometry(stream)

        alarms = self.alarms(series, threshold)

        in_detect = np.zeros(alarms.shape, dtype=bool)
        outcomes = []

        for t in stream.events:

            a, b = t.detect
            hit = alarms[(alarms >= a) & (alarms <= b)]
            in_detect |= (alarms >= a) & (alarms <= b)

            detected = len(hit) > 0
            first = float(hit[0]) if detected else float("nan")

            outcomes.append(
                EventOutcome(
                    evid=t.evid,
                    event_type=t.event_type,
                    waveform_id=stream.waveform_id,
                    has_positive=bool(t.has_positive),
                    peak_ratio=t.peak_ratio,
                    detected=detected,
                    alarm_s=first,
                    delay_s=first - t.arrival_s if detected else float("nan"),
                    delay_onset_s=(
                        first - t.onset_s
                        if detected and t.onset_s is not None
                        else float("nan")
                    ),
                )
            )

        is_neutral = contains(neutral, alarms) & ~in_detect
        false_alarm = ~in_detect & ~is_neutral

        return StreamResult(
            waveform_id=stream.waveform_id,
            outcomes=outcomes,
            alarms=alarms,
            n_false_alarms=int(false_alarm.sum()),
            n_neutral=int(is_neutral.sum()),
            background_hours=hours,
        )

    def evaluate(self, streams, series_by_id: dict, threshold: float) -> EvalResult:

        results = []

        for st in streams:

            if st.waveform_id not in series_by_id:
                raise KeyError(f"No scores for waveform {st.waveform_id}")

            results.append(
                self.score_stream(st, series_by_id[st.waveform_id], threshold)
            )

        return EvalResult(threshold=float(threshold), streams=results)

    # ------------------------------------------------------------------

    def default_thresholds(self, series_by_id: dict) -> np.ndarray:

        pooled = np.concatenate([s.scores for s in series_by_id.values()])
        pooled = pooled[~np.isnan(pooled)]

        n = self.config.n_thresholds

        probs = np.concatenate([
            np.linspace(0.0, 0.99, n),
            1.0 - np.logspace(-2, -5, n),
        ])

        return np.unique(np.quantile(pooled, probs))

    def curve(self, streams, series_by_id: dict, thresholds=None) -> Curve:

        if thresholds is None:
            thresholds = self.default_thresholds(series_by_id)

        thresholds = np.asarray(sorted(thresholds), dtype=np.float64)

        rec = {g: [] for g in ("all", "usable", "weak")}
        fa, na = [], []
        hours = 0.0

        for thr in thresholds:

            r = self.evaluate(streams, series_by_id, thr)

            for g in rec:
                rec[g].append(r.recall(g))

            fa.append(r.fa_per_day)
            na.append(r.n_alarms)
            hours = r.background_hours

        return Curve(
            thresholds=thresholds,
            recall_all=np.array(rec["all"]),
            recall_usable=np.array(rec["usable"]),
            recall_weak=np.array(rec["weak"]),
            fa_per_day=np.array(fa),
            n_alarms=np.array(na),
            background_hours=hours,
        )

    def select_threshold(
        self,
        curve: Curve,
        max_fa_per_day: float | None = None,
        group: str = "all",
    ) -> float:
        """
        Highest recall under the false-alarm budget; among equal recalls
        the highest threshold (fewest alarms). If the budget cannot be met
        at all, the largest threshold of the grid.
        """

        budget = (
            self.config.max_fa_per_day
            if max_fa_per_day is None
            else max_fa_per_day
        )

        feasible = np.nan_to_num(curve.fa_per_day, nan=np.inf) <= budget

        if not feasible.any():
            return float(curve.thresholds[-1])

        r = np.nan_to_num(curve.recall(group), nan=-1.0)
        best = r[feasible].max()

        candidates = feasible & (r >= best - 1e-12)

        return float(curve.thresholds[candidates].max())

    # ------------------------------------------------------------------

    def bootstrap(self, result: EvalResult, n_boot: int | None = None) -> dict:
        """
        95 % intervals from resampling WAVEFORMS (events and false alarms
        of one day are not independent). Returns name -> (point, lo, hi)
        for recall (all / usable / weak / per type) and fa_per_day.
        """

        n_boot = self.config.n_bootstrap if n_boot is None else n_boot
        rng = np.random.default_rng(self.config.seed)

        streams = result.streams
        n = len(streams)

        types = sorted({o.event_type for o in result.outcomes})
        groups = ["all", "usable", "weak"] + types

        n_ev = np.zeros((n, len(groups)))
        n_det = np.zeros((n, len(groups)))

        for i, s in enumerate(streams):
            for j, g in enumerate(groups):
                sel = [o for o in s.outcomes if in_group(o, g)]
                n_ev[i, j] = len(sel)
                n_det[i, j] = sum(o.detected for o in sel)

        fa = np.array([s.n_false_alarms for s in streams], dtype=float)
        hrs = np.array([s.background_hours for s in streams], dtype=float)

        def metrics(idx):
            ev = n_ev[idx].sum(axis=0)
            dt = n_det[idx].sum(axis=0)
            rec = np.where(ev > 0, dt / np.maximum(ev, 1), np.nan)
            h = hrs[idx].sum()
            return rec, (24.0 * fa[idx].sum() / h if h > 0 else np.nan)

        point_rec, point_fa = metrics(np.arange(n))

        boots = [
            metrics(rng.integers(0, n, size=n)) for _ in range(n_boot)
        ]

        out = {}

        for j, g in enumerate(groups):
            v = np.array([b[0][j] for b in boots])
            v = v[~np.isnan(v)]
            out[f"recall_{g}"] = (
                float(point_rec[j]),
                float(np.percentile(v, 2.5)) if len(v) else float("nan"),
                float(np.percentile(v, 97.5)) if len(v) else float("nan"),
            )

        v = np.array([b[1] for b in boots])
        v = v[~np.isnan(v)]
        out["fa_per_day"] = (
            float(point_fa),
            float(np.percentile(v, 2.5)) if len(v) else float("nan"),
            float(np.percentile(v, 97.5)) if len(v) else float("nan"),
        )

        return out

    # ------------------------------------------------------------------

    @staticmethod
    def print_result(result: EvalResult, intervals: dict | None = None) -> None:

        print("=" * 60)
        print("EVENT-LEVEL RESULT")
        print("=" * 60)
        print(f"Threshold: {result.threshold:.4g}")
        print(f"Alarms: {result.n_alarms} | false alarms: "
              f"{result.n_false_alarms} over "
              f"{result.background_hours:.1f} scored hours "
              f"-> {result.fa_per_day:.2f} per 24 h")

        types = sorted({o.event_type for o in result.outcomes})

        print()
        print(f"{'group':14s} {'events':>7s} {'detected':>9s} {'recall':>7s}"
              + ("   95% interval" if intervals else ""))

        for g in ["all", "usable", "weak"] + types:
            n = result.n_events(g)
            d = sum(o.detected for o in result.outcomes if in_group(o, g))
            line = f"{g:14s} {n:7d} {d:9d} {result.recall(g):7.2f}"
            if intervals and f"recall_{g}" in intervals:
                _, lo, hi = intervals[f"recall_{g}"]
                line += f"   [{lo:.2f}, {hi:.2f}]"
            print(line)

        if intervals and "fa_per_day" in intervals:
            _, lo, hi = intervals["fa_per_day"]
            print(f"\nFalse alarms per 24 h: {result.fa_per_day:.2f} "
                  f"[{lo:.2f}, {hi:.2f}]")

        for ref, label in (("arrival", "catalogue time"),
                           ("onset", "visible onset")):
            d = result.delays("all", ref)
            if len(d):
                print(f"Delay vs {label} (s): median {np.median(d):.0f} "
                      f"(p25 {np.percentile(d, 25):.0f}, "
                      f"p75 {np.percentile(d, 75):.0f}, n={len(d)})")