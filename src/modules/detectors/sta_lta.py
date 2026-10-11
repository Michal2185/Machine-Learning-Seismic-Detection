from collections import defaultdict

import numpy as np

from .types import Detector


def _end_index(w) -> int:
    """Index of the last sample of a window in its stream."""
    end = getattr(w, "end_sample", None)
    if end is None:
        end = w.window.end_sample
    return int(end) - 1


def valid_mask(n: int, fs: float, gaps) -> np.ndarray:
    """1.0 for observed samples, 0.0 inside interpolated gaps."""

    valid = np.ones(n, dtype=np.float64)

    for a, b in gaps:
        i0 = max(0, int(np.floor(a * fs)))
        i1 = min(n, int(np.ceil(b * fs)) + 1)
        valid[i0:i1] = 0.0

    return valid


def sta_lta_at(
    x: np.ndarray,
    valid: np.ndarray,
    fs: float,
    sta_s: float,
    lta_s: float,
    feature: str,
    end_idx: np.ndarray,
    min_lta_s: float = 600.0,
    delay_lta: bool = False,
) -> np.ndarray:
    """
    Causal STA/LTA ratio at the given sample indices.

    STA = mean feature over the last sta_s seconds, LTA = mean over the
    last lta_s seconds (ending at the same sample, or just before the STA
    window if delay_lta). Interpolated samples (valid = 0) are left out of
    both averages. Score 0 where the STA has fewer than half its samples
    observed or the LTA has less than min_lta_s of observed history.
    """

    x = np.asarray(x, dtype=np.float64)

    # the ratio does not depend on the scale; a robust scale keeps the
    # numbers away from denormals (raw values are about 1e-10)
    nz = np.abs(x[valid > 0])
    scale = float(np.median(nz)) if len(nz) else 1.0
    x = x / (scale if scale > 0 else 1.0)

    z = (x * x if feature == "energy" else np.abs(x)) * valid

    ce = np.concatenate(([0.0], np.cumsum(z)))
    cn = np.concatenate(([0.0], np.cumsum(valid)))

    ns = max(1, int(round(sta_s * fs)))
    nl = max(ns + 1, int(round(lta_s * fs)))

    i1 = np.asarray(end_idx, dtype=np.int64) + 1
    s0 = np.maximum(i1 - ns, 0)

    l1 = s0 if delay_lta else i1
    l0 = np.maximum(l1 - nl, 0)

    sta_sum, sta_n = ce[i1] - ce[s0], cn[i1] - cn[s0]
    lta_sum, lta_n = ce[l1] - ce[l0], cn[l1] - cn[l0]

    ok = (sta_n >= 0.5 * ns) & (lta_n >= min_lta_s * fs)

    out = np.zeros(len(i1), dtype=np.float64)

    sta = sta_sum[ok] / sta_n[ok]
    lta = lta_sum[ok] / lta_n[ok]

    out[ok] = sta / np.maximum(lta, 1e-12)

    return out


class StaLtaDetector(Detector):
    """
    Classical STA/LTA on the CONTINUOUS stream (long LTA, can use the
    history before the window). The usual onboard detector.
    """

    def __init__(
        self,
        sta_s: float,
        lta_s: float,
        feature: str = "energy",
        mask_gaps: bool = True,
        delay_lta: bool = False,
        min_lta_s: float = 600.0,
    ):
        if feature not in ("energy", "abs"):
            raise ValueError("feature must be 'energy' or 'abs'")

        self.sta_s = sta_s
        self.lta_s = lta_s
        self.feature = feature
        self.mask_gaps = mask_gaps
        self.delay_lta = delay_lta
        self.min_lta_s = min_lta_s

        self.name = (
            f"sta_lta(sta={sta_s:g}s, lta={lta_s:g}s, {feature}"
            f"{'' if mask_gaps else ', gaps unmasked'})"
        )

    def score(self, windows, streams_by_id) -> np.ndarray:

        scores = np.zeros(len(windows), dtype=np.float64)

        by_stream = defaultdict(list)

        for i, w in enumerate(windows):
            by_stream[w.event.evid].append(i)

        for wid, idx in by_stream.items():

            st = streams_by_id[wid]
            fs = float(st.sampling_rate)
            x = np.asarray(st.data, dtype=np.float64)

            valid = (
                valid_mask(len(x), fs, st.gaps)
                if self.mask_gaps
                else np.ones(len(x))
            )

            end_idx = np.array([_end_index(windows[i]) for i in idx])

            scores[idx] = sta_lta_at(
                x, valid, fs, self.sta_s, self.lta_s, self.feature,
                end_idx, self.min_lta_s, self.delay_lta,
            )

        return scores


class WindowStaLtaDetector(Detector):
    """
    STA/LTA restricted to the 60 s window: mean feature of the last sta_s
    seconds divided by the mean of the rest of the window. Sees exactly
    what a window-based network sees, so it is the fair classical
    comparison for the networks.
    """

    def __init__(self, sta_s: float = 20.0, feature: str = "energy", chunk: int = 20000):

        if feature not in ("energy", "abs"):
            raise ValueError("feature must be 'energy' or 'abs'")

        self.sta_s = sta_s
        self.feature = feature
        self.chunk = chunk
        self.name = f"window_sta_lta(sta={sta_s:g}s, {feature})"

    def score(self, windows, streams_by_id=None) -> np.ndarray:

        scores = np.zeros(len(windows), dtype=np.float64)

        if not windows:
            return scores

        w0 = windows[0]
        n = len(w0.data)
        length = w0.end_time - w0.start_time
        ns = int(np.clip(round(self.sta_s / length * n), 1, n - 1))

        for a in range(0, len(windows), self.chunk):

            block = np.stack(
                [np.asarray(w.data, dtype=np.float64)
                 for w in windows[a:a + self.chunk]]
            )

            z = block * block if self.feature == "energy" else np.abs(block)

            sta = z[:, -ns:].mean(axis=1)
            lta = z[:, :-ns].mean(axis=1)

            eps = 1e-6 * z.mean(axis=1)

            scores[a:a + self.chunk] = sta / (lta + eps + 1e-300)

        return scores


def sta_lta_grid(
    sta=(10.0, 30.0, 60.0, 120.0, 240.0, 480.0),
    lta=(900.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0, 86400.0),
    features=("energy", "abs"),
    **kwargs,
) -> list:
    """
    Candidate detectors for tuning on the validation folds.

    The first grid stopped at STA 120 s / LTA 7200 s and every fold picked
    the edge; the second stopped at LTA 28,800 s and four of five folds
    picked that edge again. An LTA of 86,400 s is the whole day so far
    (cumulative mean), which is the natural upper end for these files.
    A baseline tuned on a grid whose optimum lies outside it flatters
    the models.
    """

    return [
        StaLtaDetector(s, l, f, **kwargs)
        for s in sta
        for l in lta
        for f in features
        if l > s
    ]


def window_sta_lta_grid(sta=(10.0, 20.0, 30.0), features=("energy", "abs")) -> list:
    return [WindowStaLtaDetector(s, f) for s in sta for f in features]