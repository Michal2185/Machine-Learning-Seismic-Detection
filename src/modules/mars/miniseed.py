"""
InSight miniSEED -> one continuous float64 trace (dependency-free; obspy is not needed).

read_trace(path) returns MarsTrace(start, fs, data, gaps_s). Records are placed by their time stamps; missing samples
(none in the files seen so far) are filled by linear interpolation and listed in gaps_s as (start_s, end_s), the same
convention as the interpolated stretches of the Apollo zone table.
"""

from dataclasses import dataclass, field

import numpy as np

from .mseed_records import read_mseed


@dataclass
class MarsTrace:
    start: object                 # datetime of the first sample (UTC, naive)
    fs: float
    data: np.ndarray              # float64
    gaps_s: list = field(default_factory=list)


def assemble(recs) -> MarsTrace:
    """recs: [(t0 datetime, fs, nsamples, encoding, data)] in file order."""

    if not recs:
        raise ValueError("no miniSEED records")

    t0, fs = recs[0][0], float(recs[0][1])

    if any(abs(float(r[1]) - fs) > 1e-9 for r in recs):
        raise ValueError("sampling rate changes inside the file")

    pos = [int(round((r[0] - t0).total_seconds() * fs)) for r in recs]
    n = max(p + len(r[4]) for p, r in zip(pos, recs))

    x = np.full(n, np.nan)

    for p, r in zip(pos, recs):
        x[p:p + len(r[4])] = np.asarray(r[4], dtype=np.float64)

    bad = ~np.isfinite(x)
    gaps = []

    if bad.any():
        idx = np.arange(n)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
        gaps = [(float(a / fs), float(b / fs)) for a, b in zip(edges[::2], edges[1::2])]

    return MarsTrace(start=t0, fs=fs, data=x, gaps_s=gaps)


def read_trace(path) -> MarsTrace:
    return assemble(read_mseed(str(path)))