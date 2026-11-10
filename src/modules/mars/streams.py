"""
InSight snippets -> EvalStream / EventTarget / windows in the format the Apollo evaluator and detectors expect.

Differences from the Apollo data branch, all deliberate and listed in docs/MARS_PREREGISTRATION.md:
  * the files are 1 h snippets around an event, not 24 h days: false alarms per day are NOT estimable from them;
  * the first `trim_start_s` (filter start-up + file taper) and last `trim_end_s` seconds are registered as gaps
    (neutral for false alarms, masked for STA/LTA);
  * event zones (visible onset, coda end) use the Apollo rule (20 s RMS envelope, onset = 3x local background
    sustained 30 s, coda end = 2x background sustained 600 s) restated here with the parameters of the data handoff,
    because the MQS end time is not available; the local background is the pre-event stretch of the same hour.
"""

import csv
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import chain
from .miniseed import read_trace

try:                                            # real classes when running inside the repo
    from src.modules.pipeline.types import EvalStream, EventTarget
except Exception:                               # sandbox / tests: identical fields
    @dataclass
    class EventTarget:
        evid: str
        event_type: str
        arrival_s: float
        detect: tuple
        ignore: tuple
        onset_s: object
        has_positive: bool
        peak_ratio: object

    @dataclass
    class EvalStream:
        waveform_id: str
        data: np.ndarray
        sampling_rate: float
        start_time: object
        events: list
        gaps: list


# ----------------------------------------------------------------------
# catalogue
# ----------------------------------------------------------------------

def read_catalogue(path) -> list:
    """CSV with columns filename, time_abs(%Y-%m-%dT%H:%M:%S.%f), time_rel(sec), evid -> [dict]."""

    out = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keys = {k.split("(")[0].strip(): v for k, v in row.items()}
            out.append(dict(
                evid=keys["evid"].strip(),
                filename=keys["filename"].strip(),
                time_abs=datetime.strptime(keys["time_abs"].strip(), "%Y-%m-%dT%H:%M:%S.%f"),
                time_rel=float(keys["time_rel"]),
            ))

    return out


def find_file(data_dir, evid: str) -> str:

    hits = sorted(glob.glob(os.path.join(str(data_dir), f"*{evid}.mseed")))

    if len(hits) != 1:
        raise FileNotFoundError(f"expected one miniSEED file for {evid} in {data_dir}, found {len(hits)}")

    return hits[0]


# ----------------------------------------------------------------------
# event zone (Apollo rule, restated)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ZoneParams:
    env_s: float = 20.0
    k_on: float = 3.0
    onset_search: tuple = (-900.0, 300.0)
    onset_min_dur_s: float = 30.0
    k_end: float = 2.0
    coda_min_below_s: float = 600.0
    peak_search_s: float = 900.0
    bg_span_s: float = 1500.0          # local background: up to this long, ending bg_margin_s before the arrival
    bg_margin_s: float = 180.0
    min_bg_s: float = 120.0
    usable: tuple = (-120.0, 600.0)    # visible-onset offset that counts as usable
    ignore_before_s: float = 180.0
    detect: tuple = (120.0, 600.0)     # [arrival - 120, arrival + 600]


def envelope(x, fs: float, env_s: float) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)
    m = uniform_filter1d(x * x, size=max(1, int(env_s * fs)), mode="nearest")

    return np.sqrt(np.maximum(m, 0.0))


def first_sustained(mask, n: int):

    mask = np.asarray(mask, dtype=bool)

    if n <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else None

    if len(mask) < n:
        return None

    cs = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    idx = np.flatnonzero(cs[n:] - cs[:-n] == n)

    return int(idx[0]) if len(idx) else None


def estimate_zone(env, fs: float, arrival: float, t_lo: float, t_hi: float, p: ZoneParams | None = None) -> dict:
    """
    Onset / peak / coda of one event inside the usable part [t_lo, t_hi] (seconds) of a stream.
    status: ok | pre_active (envelope already above k_on x background at the start of the search) | none | no_bg
    """

    p = p or ZoneParams()
    I = lambda s: int(round(s * fs))

    lo_bg = max(t_lo, arrival - p.bg_span_s)
    hi_bg = arrival - p.bg_margin_s

    out = dict(status="no_bg", onset_offset_s=float("nan"), bg=float("nan"), peak_ratio=float("nan"),
               peak_delay_s=float("nan"), coda_end_s=float(t_hi), coda_censored=True)

    if hi_bg - lo_bg < p.min_bg_s:
        return out

    bg = float(np.median(env[I(lo_bg):I(hi_bg)]))
    out["bg"] = bg

    s0, s1 = I(max(t_lo, arrival + p.onset_search[0])), I(min(t_hi, arrival + p.onset_search[1]))
    m = env[s0:s1] > p.k_on * bg

    if len(m) and m[0]:
        out["status"] = "pre_active"
    else:
        idx = first_sustained(m, I(p.onset_min_dur_s)) if len(m) else None
        out["status"] = "none" if idx is None else "ok"
        if idx is not None:
            out["onset_offset_s"] = (s0 + idx) / fs - arrival

    a = I(arrival)
    e1 = min(len(env), I(t_hi), a + I(p.peak_search_s))
    pk = a + int(np.argmax(env[a:e1]))

    out["peak_delay_s"] = pk / fs - arrival
    out["peak_ratio"] = float(env[pk] / bg)

    sub = env[pk:I(t_hi)]
    j = first_sustained(sub < p.k_end * bg, I(p.coda_min_below_s))

    if j is not None:
        out["coda_end_s"] = (pk + j) / fs
        out["coda_censored"] = False

    return out


def make_target(evid: str, arrival: float, zone: dict, p: ZoneParams | None = None, event_type: str = "marsquake"):

    p = p or ZoneParams()
    off = zone["onset_offset_s"]
    usable = zone["status"] == "ok" and p.usable[0] <= off <= p.usable[1]
    onset = arrival + off if usable else None
    early = min(arrival, onset) if onset is not None else arrival

    return EventTarget(
        evid=evid,
        event_type=event_type,
        arrival_s=float(arrival),
        detect=(arrival - p.detect[0], arrival + p.detect[1]),
        ignore=(early - p.ignore_before_s, float(zone["coda_end_s"])),
        onset_s=None if onset is None else float(onset),
        has_positive=bool(usable),
        peak_ratio=None if not np.isfinite(zone["peak_ratio"]) else float(zone["peak_ratio"]),
    )


# ----------------------------------------------------------------------
# streams
# ----------------------------------------------------------------------

def build_streams(catalogue, data_dir, arms=("lp1.3", "nolp"), trim_start_s: float = 120.0,
                  trim_end_s: float = 30.0, zone: ZoneParams | None = None):
    """
    Returns ({arm: [EvalStream]}, [info dict per waveform]). Targets (and so detect / ignore intervals) are computed
    ONCE from the highpass-only data and shared by all arms, so the arms differ only in the detector input.
    """

    zone = zone or ZoneParams()
    streams = {arm: [] for arm in arms}
    info = []

    for rec in catalogue:

        tr = read_trace(find_file(data_dir, rec["evid"]))
        hp = chain.preprocess(tr.data, tr.fs)
        fs = chain.FS_APOLLO
        dur = len(hp) / fs
        t_lo, t_hi = trim_start_s, dur - trim_end_s

        # offset of the first sample: the catalogue time_rel is relative to the file start, as for Apollo
        env = envelope(hp, fs, zone.env_s)
        z = estimate_zone(env, fs, rec["time_rel"], t_lo, t_hi, zone)
        target = make_target(rec["evid"], rec["time_rel"], z, zone)

        gaps = [(0.0, t_lo), (t_hi, dur)] + [(a, b) for a, b in tr.gaps_s]

        for arm in arms:
            streams[arm].append(EvalStream(
                waveform_id=rec["evid"],
                data=chain.arm_data(hp, arm),
                sampling_rate=fs,
                start_time=tr.start,
                events=[target],
                gaps=gaps,
            ))

        info.append(dict(evid=rec["evid"], start=str(tr.start), duration_s=dur, zone=z, target=target,
                         input_gaps=tr.gaps_s))

    return streams, info


# ----------------------------------------------------------------------
# windows
# ----------------------------------------------------------------------

@dataclass
class _EventRef:
    evid: str
    event_type: str = "marsquake"


@dataclass
class MarsWindow:
    """Duck-types LabeledWindowRecord for the detectors and the evaluator (.data .label .end_time .end_sample .event)."""

    data: np.ndarray
    label: int
    start_time: float
    end_time: float
    end_sample: int
    event: _EventRef = field(default=None)


def window_grid(n_samples: int, fs: float, length_s: float = 60.0, step_s: float = 10.0):
    """
    (end_sample array, window length in samples). Length and step are ROUNDED to whole samples, as in the Apollo
    pipeline: 60 s x 6.625 Hz -> 398 samples, 10 s x 6.625 Hz = 66.25 -> 66 samples (so the real spacing is 9.96 s).
    Verified against the Apollo count: a 24 h file (572 400 samples) gives 8 667 windows, not 8 635.
    """

    length = int(round(length_s * fs))
    step = int(round(step_s * fs))

    return np.arange(length, n_samples + 1, step, dtype=np.int64), length


def make_windows(stream, length_s: float = 60.0, step_s: float = 10.0, time_rule: str = "samples") -> list:
    """
    time_rule 'samples': end_time = end_sample / fs (consistent with the data positions);
              'nominal': end_time = length_s + k * step_s (drifts from the data by 0.04 s per window).
    run_mars --check-windowing reports which rule the Apollo windows follow.
    """

    fs = float(stream.sampling_rate)
    ends, length = window_grid(len(stream.data), fs, length_s, step_s)
    ref = _EventRef(stream.waveform_id)
    out = []

    for k, e in enumerate(ends):
        t_end = e / fs if time_rule == "samples" else length_s + k * step_s
        t_start = (e - length) / fs if time_rule == "samples" else t_end - length_s
        out.append(MarsWindow(data=np.asarray(stream.data[e - length:e]), label=0, start_time=t_start,
                              end_time=float(t_end), end_sample=int(e), event=ref))

    return out


# ----------------------------------------------------------------------
# scoring policy
# ----------------------------------------------------------------------

def fold_for(waveform_index: int, seed: int, n_folds: int = 5) -> int:
    """Waveform i is scored by fold model (i + seed) mod n_folds: over n_folds seeds every waveform meets every fold."""
    return (int(waveform_index) + int(seed)) % n_folds