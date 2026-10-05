import numpy as np
from scipy.ndimage import uniform_filter1d

from src.modules.preprocessor import apply_filter
from src.modules.waveform_loader import WaveformRecord

from .types import EventZone, EventZoneTable, ZoneConfig


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def first_sustained(mask, n):
    """Index of the first run of >= n consecutive True values, else None."""
    mask = np.asarray(mask, dtype=bool)
    if n <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else None
    if len(mask) < n:
        return None
    cs = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    idx = np.flatnonzero(cs[n:] - cs[:-n] == n)
    return int(idx[0]) if len(idx) else None


def find_interpolated_runs(x, fs, min_run_s):
    """
    Runs of linearly interpolated samples (zero second difference) that
    last at least `min_run_s`. Returns [(start_s, end_s), ...].
    Operates on the RAW float64 samples.
    """

    x = np.asarray(x, dtype=np.float64)

    nz = x[x != 0]
    ref = float(np.median(np.abs(nz))) if len(nz) else 1.0

    d2 = np.abs(np.diff(x, 2))
    flags = np.concatenate(([False], d2 <= 1e-6 * ref, [False]))

    f = flags.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], f, [0]))))
    starts, ends = edges[::2], edges[1::2]

    min_len = int(round(min_run_s * fs))

    return [
        (float(a / fs), float(b / fs))
        for a, b in zip(starts, ends)
        if b - a >= min_len
    ]


def envelope(x, fs, env_sec):
    x = np.asarray(x, dtype=np.float64)
    m = uniform_filter1d(x * x, size=max(1, int(env_sec * fs)), mode="nearest")
    # running sums can leave tiny negative residuals in near-zero stretches
    return np.sqrt(np.maximum(m, 0.0))


def background_level(values, dropout_rel):
    """Median of the non-dropout values (iterated)."""
    level = float(np.median(values))
    for _ in range(3):
        good = values[values > dropout_rel * level]
        if len(good) == 0:
            break
        level = float(np.median(good))
    return level


def day_reference(env, fs, arrivals, cfg):

    mask = np.ones(len(env), dtype=bool)

    for arr in arrivals:
        a = max(0, int(round((arr + cfg.event_zone[0]) * fs)))
        b = max(0, int(round((arr + cfg.event_zone[1]) * fs)))
        mask[a:b] = False

    if mask.sum() / fs >= 3600:
        return background_level(env[mask], cfg.dropout_rel)

    return background_level(env, cfg.dropout_rel)


# ----------------------------------------------------------------------
# estimator
# ----------------------------------------------------------------------

class ZoneEstimator:
    """
    Per-event onset / peak / coda estimates from the signal envelope,
    plus the interpolated gaps of every waveform.

    Input: the RAW waveforms (WaveformRecord, one per DISTINCT waveform,
    e.g. grouping.waveforms) and event_groups (primary evid -> all events
    on that waveform, sorted by arrival).
    """

    def __init__(self, config: ZoneConfig | None = None):
        self.config = config or ZoneConfig()

    def run(
        self,
        waveforms: list[WaveformRecord],
        event_groups: dict | None = None,
    ) -> EventZoneTable:

        cfg = self.config
        event_groups = event_groups or {}

        zones = {}
        gaps = {}

        for rec in waveforms:

            if len(rec.stream) != 1:
                raise ValueError(
                    f"Expected one trace for {rec.event.evid}, "
                    f"got {len(rec.stream)}"
                )

            primary = rec.event.evid
            trace = rec.stream[0]
            fs = float(trace.stats.sampling_rate)
            raw = np.asarray(trace.data, dtype=np.float64)

            gaps[primary] = find_interpolated_runs(raw, fs, cfg.gap_min_run_s)

            x = apply_filter(raw, fs, cfg.analysis_low_hz, None, cfg.filter_order)
            env = envelope(x, fs, cfg.env_sec)

            events = event_groups.get(primary, [rec.event])

            b_ref = day_reference(env, fs, [e.time_rel for e in events], cfg)

            for k, e in enumerate(events):

                nxt = events[k + 1].time_rel if k + 1 < len(events) else None

                zones[e.evid] = self._analyse(
                    env, fs, e, nxt, b_ref, gaps[primary], primary
                )

        return EventZoneTable(zones=zones, gaps=gaps, config=cfg)

    # ------------------------------------------------------------------

    def _analyse(self, env, fs, event, next_arrival, b_ref, gap_runs, wid):

        cfg = self.config
        n = len(env)
        arrival = float(event.time_rel)

        def I(s):
            return int(round(s * fs))

        limit = n if next_arrival is None else min(n, I(next_arrival - 60))
        a = min(max(I(arrival), 0), n - 1)

        # local background (non-dropout samples only)
        l0 = max(0, I(arrival + cfg.local_ref[0]))
        l1 = max(0, I(arrival + cfg.local_ref[1]))
        loc = env[l0:l1]
        loc = loc[loc > cfg.dropout_rel * b_ref]
        b_loc = float(np.median(loc)) if len(loc) >= I(600) else b_ref

        # visible onset
        s0 = max(0, I(arrival + cfg.onset_search[0]))
        s1 = min(n, I(arrival + cfg.onset_search[1]))
        m = env[s0:s1] > cfg.k_on * b_loc

        onset, status = np.nan, "none"

        if len(m) and m[0]:
            status = "pre_active"
        elif len(m):
            idx = first_sustained(m, I(cfg.onset_min_dur_s))
            if idx is not None:
                onset, status = (s0 + idx) / fs - arrival, "ok"

        # peak
        p1 = max(a + 1, min(limit, a + I(cfg.peak_search_s)))
        peak_idx = a + int(np.argmax(env[a:p1]))
        peak_delay = peak_idx / fs - arrival
        peak_ratio = float(env[peak_idx] / b_ref)

        # coda end (dropouts carry no information and are skipped)
        sub = env[peak_idx:limit]
        valid = np.flatnonzero(sub > cfg.dropout_rel * b_ref)

        idx = None
        if len(valid):
            j = first_sustained(
                sub[valid] < cfg.k_end * b_ref, I(cfg.coda_min_below_s)
            )
            idx = None if j is None else int(valid[j])

        if idx is None:
            coda_end = limit / fs - arrival
            censored = True
            reason = (
                "next_event"
                if next_arrival is not None and I(next_arrival - 60) < n
                else "file_end"
            )
        else:
            coda_end = (peak_idx + idx) / fs - arrival
            censored, reason = False, ""

        # long interpolated gap inside the coda?
        lo = arrival + peak_delay
        hi = arrival + coda_end
        gap_in_coda = any(
            (b - a_) >= cfg.long_gap_s and a_ < hi and b > lo
            for a_, b in gap_runs
        )

        # exponential decay constant
        tau = np.nan
        if not censored and peak_ratio > cfg.k_end:
            tau = (coda_end - peak_delay) / np.log(peak_ratio / cfg.k_end)

        return EventZone(
            evid=event.evid,
            event_type=event.event_type,
            waveform_id=wid,
            arrival_s=arrival,
            onset_offset_s=float(onset),
            onset_status=status,
            peak_delay_s=float(peak_delay),
            peak_ratio=peak_ratio,
            background=float(b_ref),
            local_background_ratio=float(b_loc / b_ref),
            coda_end_s=float(coda_end),
            coda_censored=bool(censored),
            censor_reason=reason,
            gap_in_coda=bool(gap_in_coda),
            tau_s=float(tau),
        )