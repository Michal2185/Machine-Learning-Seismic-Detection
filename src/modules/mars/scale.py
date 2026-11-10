"""N0 recalibration for a target mission: estimate the fixed scale from target background, set it on a loaded detector."""

import numpy as np

try:
    from src.modules.normalization.module import estimate_fixed_scale
except Exception:                                # sandbox / tests: same estimator (median of per-waveform robust std)
    def estimate_fixed_scale(arrays, stat="median_of_waveforms"):
        if stat != "median_of_waveforms":
            raise ValueError(stat)
        lv = []
        for a in arrays:
            a = np.asarray(a, dtype=np.float64).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                lv.append(1.4826 * np.median(np.abs(a - np.median(a))))
        lv = [v for v in lv if np.isfinite(v) and v > 0]
        if not lv:
            raise ValueError("cannot fit a fixed scale: no usable training data")
        return float(np.median(lv))


def background_samples(stream) -> np.ndarray:
    """Samples outside every ignore interval and every gap: the analogue of the label-0 windows N0 sees on Apollo."""

    n = len(stream.data)
    fs = float(stream.sampling_rate)
    keep = np.ones(n, dtype=bool)

    for a, b in list(stream.gaps) + [t.ignore for t in stream.events]:
        i0, i1 = max(0, int(np.floor(a * fs))), min(n, int(np.ceil(b * fs)))
        if i1 > i0:
            keep[i0:i1] = False

    return np.asarray(stream.data, dtype=np.float64)[keep]


def calibrate_n0(streams) -> float:
    """Median over waveforms of the robust std of their background samples (same estimator as the Apollo N0)."""
    return float(estimate_fixed_scale([background_samples(s) for s in streams], "median_of_waveforms"))


def find_normalizers(obj, max_depth: int = 5) -> list:
    """Every Normalizer-like object (has .config.variant and .fixed_scale) reachable from `obj`."""

    found, seen = [], set()

    def walk(o, d):
        if id(o) in seen or d > max_depth or isinstance(o, (str, bytes, int, float, bool, np.ndarray, type(None))):
            return
        seen.add(id(o))
        if hasattr(o, "fixed_scale") and hasattr(getattr(o, "config", None), "variant"):
            found.append(o)
            return
        if isinstance(o, dict):
            items = list(o.values())
        elif isinstance(o, (list, tuple)):
            items = list(o)
        else:
            items = list(getattr(o, "__dict__", {}).values())
        for v in items:
            walk(v, d + 1)

    walk(obj, 0)

    return found


def set_fixed_scale(det, c: float) -> list:
    """
    Overrides the N0 constant of a loaded detector (LearnerDetector, RFC, amplitude control). Feature / envelope
    standardisation is NOT touched. Refuses N2 / N3 and non-positive or non-finite values. Returns the old constants.
    """

    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"fixed scale must be finite and > 0, got {c!r}")

    norms = find_normalizers(det)

    if not norms:
        raise TypeError(f"no normalizer found inside {type(det).__name__}")

    for n in norms:
        if n.config.variant != "N0":
            raise ValueError(f"set_fixed_scale supports N0 only (found {n.config.variant})")

    old = [n.fixed_scale for n in norms]

    for n in norms:
        n.fixed_scale = float(c)

    return old