"""
Waveform-level bootstrap statistics from saved event outcomes (no evaluator objects needed).

The runner saves, per config / budget / seed, a plain 'tallies' dict (extract_tallies):

    {waveform_id: {"events": [{evid, event_type, has_positive, detected, delay_s, delay_onset_s}],
                   "n_false": int, "hours": float}}

A Panel holds all seeds of ONE configuration at ONE budget (same waveforms, same events). Metrics are
SEED-AVERAGED: recall = mean over seeds of the pooled recall, so the intervals describe uncertainty
over WAVEFORMS (events and false alarms of one day are not independent), not over seeds; the spread over
seeds is reported separately.

bootstrap() and compare() resample waveforms exactly like Evaluator.bootstrap / paired_bootstrap
(rng = default_rng(seed); idx = rng.integers(0, n, n) per resample; waveform ids sorted), so for one seed
compare() reproduces paired_bootstrap() (checked in tests/test_stats.py when the evaluation module is
importable).
"""

import numpy as np

BASE_GROUPS = ("all", "usable", "weak")


def _in_group(o: dict, g: str) -> bool:
    if g == "all":
        return True
    if g == "usable":
        return bool(o["has_positive"])
    if g == "weak":
        return not bool(o["has_positive"])
    return o["event_type"] == g


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def extract_tallies(result) -> dict:
    """Plain, JSON-serialisable outcomes of an EvalResult (pooled over folds)."""

    out = {}

    for s in result.streams:
        out[s.waveform_id] = dict(
            events=[
                dict(
                    evid=o.evid,
                    event_type=o.event_type,
                    has_positive=bool(o.has_positive),
                    detected=bool(o.detected),
                    delay_s=_num(o.delay_s),
                    delay_onset_s=_num(o.delay_onset_s),
                )
                for o in s.outcomes
            ],
            n_false=int(s.n_false_alarms),
            hours=float(s.background_hours),
        )

    return out


class Panel:
    """Outcomes of one configuration at one budget over several seeds."""

    def __init__(self, tallies_list: list):

        if not tallies_list:
            raise ValueError("need at least one seed")

        self.ids = sorted(tallies_list[0])

        for t in tallies_list[1:]:
            if sorted(t) != self.ids:
                raise ValueError("all seeds must cover the same waveforms")

        types = sorted({e["event_type"] for t in tallies_list[0].values() for e in t["events"]})
        self.groups = list(BASE_GROUPS) + types

        n, g, s = len(self.ids), len(self.groups), len(tallies_list)
        self.ev = np.zeros((n, g))
        self.det = np.zeros((s, n, g))
        self.fa = np.zeros((s, n))
        self.hours = np.zeros(n)
        self.events = {}                             # (waveform, evid) -> per-seed outcome dicts

        for k, t in enumerate(tallies_list):
            for i, wid in enumerate(self.ids):
                ev = t[wid]["events"]
                for j, grp in enumerate(self.groups):
                    sel = [e for e in ev if _in_group(e, grp)]
                    if k == 0:
                        self.ev[i, j] = len(sel)
                    elif self.ev[i, j] != len(sel):
                        raise ValueError(f"waveform {wid}: events differ between seeds")
                    self.det[k, i, j] = sum(e["detected"] for e in sel)
                self.fa[k, i] = t[wid]["n_false"]
                if k == 0:
                    self.hours[i] = t[wid]["hours"]
                for e in ev:
                    self.events.setdefault((wid, e["evid"]), []).append(e)

    @property
    def n_seeds(self) -> int:
        return self.det.shape[0]

    def metrics(self, idx) -> tuple:
        """Seed-averaged recall per group and false alarms per 24 h on the waveforms idx."""

        e = self.ev[idx].sum(axis=0)
        rec = np.where(e > 0, self.det[:, idx].sum(axis=1) / np.maximum(e, 1), np.nan).mean(axis=0)
        h = self.hours[idx].sum()
        fa = 24.0 * self.fa[:, idx].sum(axis=1) / h if h > 0 else np.full(self.n_seeds, np.nan)

        return rec, float(np.mean(fa))

    def per_seed_recall(self, group: str = "all") -> np.ndarray:

        j = self.groups.index(group)
        e = self.ev[:, j].sum()

        return self.det[:, :, j].sum(axis=1) / e if e > 0 else np.full(self.n_seeds, np.nan)

    def delays(self, ref: str = "onset") -> dict:
        """Pooled over seeds and detected events: median / quartiles / n of the delay in seconds."""

        key = "delay_onset_s" if ref == "onset" else "delay_s"
        v = [e[key] for es in self.events.values() for e in es if e["detected"] and e[key] is not None]

        if not v:
            return dict(n=0, median=float("nan"), p25=float("nan"), p75=float("nan"))

        return dict(n=len(v), median=float(np.median(v)), p25=float(np.percentile(v, 25)),
                    p75=float(np.percentile(v, 75)))


def _draws(n: int, n_boot: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_boot)]


def _ci(v: np.ndarray) -> tuple:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]

    if len(v) == 0:
        return float("nan"), float("nan")

    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bootstrap(panel: Panel, n_boot: int = 1000, seed: int = 0) -> dict:
    """name -> (point, lo, hi) for recall_<group> and fa_per_day, resampling waveforms."""

    n = len(panel.ids)
    point_r, point_f = panel.metrics(np.arange(n))
    boots = [panel.metrics(idx) for idx in _draws(n, n_boot, seed)]

    out = {}

    for j, g in enumerate(panel.groups):
        lo, hi = _ci([b[0][j] for b in boots])
        out[f"recall_{g}"] = (float(point_r[j]), lo, hi)

    lo, hi = _ci([b[1] for b in boots])
    out["fa_per_day"] = (float(point_f), lo, hi)

    return out


def compare(a: Panel, b: Panel, n_boot: int = 1000, seed: int = 0) -> dict:
    """
    Paired comparison on the same waveforms: name -> (difference b - a, lo, hi, p_b_better).
    p_b_better: share of resamples where b is better (higher recall / fewer false alarms), ties half.
    """

    if a.ids != b.ids:
        raise ValueError("both panels must cover the same waveforms")

    if a.groups != b.groups or not np.array_equal(a.ev, b.ev):
        raise ValueError("both panels must contain the same events")

    n = len(a.ids)
    pa, fa_ = a.metrics(np.arange(n))
    pb, fb_ = b.metrics(np.arange(n))
    boots = []

    for idx in _draws(n, n_boot, seed):
        ra, fa = a.metrics(idx)
        rb, fb = b.metrics(idx)
        boots.append((rb - ra, fb - fa))

    out = {}

    for j, g in enumerate(a.groups):
        v = np.array([x[0][j] for x in boots])
        v = v[~np.isnan(v)]
        lo, hi = _ci(v)
        p = float(np.mean(v > 0) + 0.5 * np.mean(v == 0)) if len(v) else float("nan")
        out[f"recall_{g}"] = (float(pb[j] - pa[j]), lo, hi, p)

    v = np.array([x[1] for x in boots])
    v = v[~np.isnan(v)]
    lo, hi = _ci(v)
    p = float(np.mean(v < 0) + 0.5 * np.mean(v == 0)) if len(v) else float("nan")
    out["fa_per_day"] = (float(fb_ - fa_), lo, hi, p)

    return out


def significant(row: tuple) -> bool:
    """True when the 95 % interval of a (diff, lo, hi, ...) row excludes 0."""

    _, lo, hi = row[:3]

    return not (np.isnan(lo) or np.isnan(hi)) and (lo > 0 or hi < 0)