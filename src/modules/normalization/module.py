import numpy as np

from .types import NormConfig, ScaleTrack

MAD_TO_STD = 1.4826


def robust_std(x) -> float:
    """MAD based standard deviation (NaN if there is no finite sample)."""

    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan")

    return float(MAD_TO_STD * np.median(np.abs(x - np.median(x))))


def gap_mask(n_samples: int, fs: float, gaps) -> np.ndarray:
    """Boolean mask of interpolated samples from [(start_s, end_s)] gaps."""

    mask = np.zeros(n_samples, dtype=bool)

    for start_s, end_s in gaps:
        i0 = max(0, int(np.floor(start_s * fs)))
        i1 = min(n_samples, int(np.ceil(end_s * fs)))
        if i1 > i0:
            mask[i0:i1] = True

    return mask


def end_sample_of(w) -> int:
    """
    End sample (exclusive) of a labeled window in its stream. The Detector
    docstring says "window.end_sample"; that is either an attribute of the
    record or of its nested `.window` object, so both are accepted.
    """

    for obj in (w, getattr(w, "window", None)):
        if obj is not None and hasattr(obj, "end_sample"):
            return int(obj.end_sample)

    raise AttributeError(
        f"cannot find end_sample on {type(w).__name__}; "
        f"attributes: {sorted(a for a in dir(w) if not a.startswith('_'))}"
    )


# ----------------------------------------------------------------------
# N0: fixed scale
# ----------------------------------------------------------------------

def estimate_fixed_scale(arrays: list, stat: str = "median_of_waveforms") -> float:
    """
    One constant for a set of TRAINING waveforms (arrays of samples).

    median_of_waveforms : median of the per-waveform robust std (a single
                          very noisy day cannot dominate it)
    pooled              : robust std of all samples together
    """

    if stat == "pooled":
        scale = robust_std(np.concatenate([np.asarray(a).ravel() for a in arrays]))

    elif stat == "median_of_waveforms":
        levels = [robust_std(a) for a in arrays]
        levels = [v for v in levels if np.isfinite(v) and v > 0]
        scale = float(np.median(levels)) if levels else float("nan")

    else:
        raise ValueError(f"unknown stat {stat!r}")

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot fit a fixed scale: no usable training data")

    return scale


def background_arrays_from_windows(train_windows, background_label: int = 0) -> list:
    """
    Per waveform, the concatenated NON-overlapping background windows
    (label 0) found in `train_windows`. Detector.fit receives windows
    only, so this is how N0 sees "the training waveforms".
    """

    by_id = {}

    for w in train_windows:
        if w.label == background_label:
            by_id.setdefault(w.event.evid, []).append(w)

    arrays = []

    for wins in by_id.values():

        wins.sort(key=end_sample_of)

        kept = []
        last_end = None

        for w in wins:
            length = len(w.data)
            end = end_sample_of(w)

            if last_end is None or end - last_end >= length:
                kept.append(np.asarray(w.data, dtype=np.float64))
                last_end = end

        if kept:
            arrays.append(np.concatenate(kept))

    return arrays


# ----------------------------------------------------------------------
# N2 / N3: causal adaptive scale
# ----------------------------------------------------------------------

def adaptive_scale_track(
    data: np.ndarray,
    fs: float,
    gaps,
    config: NormConfig,
    waveform_id: str = "",
    reference: float | None = None,
) -> ScaleTrack:
    """
    Causal background estimate: a trailing quantile of block levels.

    Every block (config.block_s) gets a robust std (interpolated samples
    excluded). The estimate for block i is the `quantile` of the valid
    block levels in the previous `window_s` (blocks i-W .. i-1). There is
    no event gate and no state, so it cannot lock onto a wrong level.

    Why a quantile: dead / quiet stretches sit below it and events above
    it. The estimate equals the background level as long as
        dead share < quantile <= 1 - event share
    of the trailing window (0.65 tolerates 65 % dead blocks or 35 % event
    blocks). For stationary noise it reads about 5 % above the true sigma
    (the 65th percentile of the block-level spread).

    Blocks with too many interpolated samples, flat or invalid blocks are
    excluded. With fewer than min_valid_blocks valid blocks in the window
    there is no estimate: early blocks carry the first one (start-up only),
    later ones the last one. Strictly causal: scale[i] depends on blocks
    < i only.

    v3 rules (only with a `reference` level, normally the N0 fixed scale of the training
    fold): blocks below dead_ratio x reference are dead and excluded from the estimate
    (instrument off / dropout, not seismic background), and the scale never drops below
    floor_ratio x reference, so the first normal window after a quiet stretch cannot be
    divided by a collapsed estimate. A file without any usable block gets scale = reference.
    reference=None gives the plain v2 behaviour.
    """

    data = np.asarray(data)
    n = len(data)
    fs = float(fs)

    block_len = max(1, int(round(config.block_s * fs)))
    n_blocks = int(np.ceil(n / block_len))
    dt = block_len / fs

    mask = gap_mask(n, fs, gaps)

    dead = (config.dead_ratio * reference
            if reference is not None and config.dead_ratio is not None else None)
    floor = (config.floor_ratio * reference
             if reference is not None and config.floor_ratio is not None else 0.0)

    level = np.full(n_blocks, np.nan)
    skipped = np.zeros(n_blocks, dtype=bool)

    for i in range(n_blocks):

        sl = slice(i * block_len, min(n, (i + 1) * block_len))
        m = mask[sl]

        if m.mean() > config.max_gap_fraction:
            skipped[i] = True
            continue

        v = robust_std(data[sl][~m])

        if not np.isfinite(v) or v <= config.min_scale or (dead is not None and v < dead):
            skipped[i] = True
            continue

        level[i] = v

    window = max(config.min_valid_blocks, int(round(config.window_s / dt)))
    scale = np.full(n_blocks, np.nan)

    for i in range(1, n_blocks):

        past = level[max(0, i - window):i]
        past = past[np.isfinite(past)]

        if past.size >= config.min_valid_blocks:
            scale[i] = max(float(np.quantile(past, config.quantile)), config.min_scale, floor)

    valid = np.isfinite(scale)

    if valid.any():
        first = int(np.argmax(valid))
        scale[:first] = scale[first]                 # start-up

        for i in range(first + 1, n_blocks):         # hold the last estimate
            if not valid[i]:
                scale[i] = scale[i - 1]
    else:
        scale[:] = reference if reference is not None else 1.0       # nothing usable

    with np.errstate(invalid="ignore"):
        elevated = level > config.elevated_ratio * scale

    return ScaleTrack(
        waveform_id=waveform_id,
        block_len=block_len,
        n_samples=n,
        scale=scale,
        level=level,
        elevated=elevated,
        skipped=skipped,
    )


class ScaleCache:
    """
    Scale tracks of all waveforms, computed on demand and kept per reference level.

    Without v3 rules (config.needs_reference False) N2 / N3 depend only on the stream, so one set
    of tracks serves all folds and models. With the rules the tracks also depend on the reference
    level (the N0 fixed scale of the training fold): `get(waveform_id, reference)` computes a track
    the first time a (waveform, reference) pair is asked for (about 1.5 s) and reuses it afterwards,
    so five folds cost five sets of tracks once per process.
    Window level lookup: track.at_end(window.end_sample).
    """

    def __init__(self, config: NormConfig):
        self.config = config
        self.streams = {}
        self._store = {}

    @staticmethod
    def _key(reference):
        return None if reference is None else float(f"{reference:.6g}")

    @property
    def tracks(self) -> dict:
        """Tracks without a reference (v2 mode, or tracks injected by a script)."""
        return self._store.setdefault(None, {})

    def add(self, stream) -> None:
        self.streams[stream.waveform_id] = stream

    @classmethod
    def from_streams(cls, streams, config: NormConfig) -> "ScaleCache":

        cache = cls(config)

        for s in streams:
            cache.add(s)

        return cache

    def get(self, waveform_id: str, reference: float | None = None) -> ScaleTrack:

        key = self._key(reference)
        tracks = self._store.setdefault(key, {})

        if waveform_id in tracks:
            return tracks[waveform_id]

        if key is None and self.config.needs_reference:
            raise RuntimeError(
                "this N2 / N3 configuration needs a reference level: fit the Normalizer first "
                "(Normalizer.fit / fit_streams) or use NormConfig(dead_ratio=None, floor_ratio=None)"
            )

        if waveform_id not in self.streams:
            raise KeyError(f"no scale track for waveform {waveform_id!r}")

        s = self.streams[waveform_id]

        tracks[waveform_id] = adaptive_scale_track(
            s.data, s.sampling_rate, s.gaps, self.config, waveform_id=waveform_id,
            reference=reference,
        )

        return tracks[waveform_id]


# ----------------------------------------------------------------------
# Normalizer
# ----------------------------------------------------------------------

class Normalizer:
    """
    Window (and stream) normalisation for all variants.

      fit(train_windows)  N0 learns its constant; N2 / N3 learn the reference level of the v3 rules;
                        N1 is a no-op
      window(w)           one normalised window (float32)
      windows(ws)         (N, T) float32
      stream(s)           continuous normalised stream (N0, N2, N3), for the
                          stateful variant S; N1 is window-only

    Windows are duck-typed: .data, .end_sample, .event.evid, .label.
    """

    def __init__(self, config: NormConfig | None = None, cache: ScaleCache | None = None):

        self.config = config or NormConfig()
        self.cache = cache
        self.fixed_scale = None
        self.reference = None          # N2 / N3 with v3 rules: N0 constant of the training fold

        if self.config.variant in ("N2", "N3") and cache is None:
            raise ValueError("N2 / N3 need a ScaleCache (ScaleCache.from_streams)")

    # ------------------------------------------------------------------

    def fit(self, train_windows) -> "Normalizer":

        v = self.config.variant

        if v == "N0":
            arrays = background_arrays_from_windows(train_windows)
            self.fixed_scale = estimate_fixed_scale(
                arrays, self.config.fixed_scale_stat
            )

        elif v in ("N2", "N3") and self.cache.config.needs_reference:
            arrays = background_arrays_from_windows(train_windows)
            self.reference = estimate_fixed_scale(arrays, self.config.fixed_scale_stat)

        return self

    def fit_streams(self, train_streams) -> "Normalizer":
        """N0 from whole training waveforms (EvalStream.data)."""

        v = self.config.variant

        if v == "N0":
            self.fixed_scale = estimate_fixed_scale(
                [s.data for s in train_streams], self.config.fixed_scale_stat
            )

        elif v in ("N2", "N3") and self.cache.config.needs_reference:
            self.reference = estimate_fixed_scale(
                [s.data for s in train_streams], self.config.fixed_scale_stat
            )

        return self

    # ------------------------------------------------------------------

    def _floor(self, scale: float) -> float:
        return max(float(scale), self.config.min_scale)

    def _apply(self, x: np.ndarray, scale) -> np.ndarray:

        y = x / scale

        if self.config.variant == "N3":
            y = np.arcsinh(y / self.config.asinh_knee)

        return y.astype(np.float32)

    def window(self, w) -> np.ndarray:

        x = np.asarray(w.data, dtype=np.float64)
        v = self.config.variant

        if v == "N0":
            if self.fixed_scale is None:
                raise RuntimeError("N0 is not fitted")
            return self._apply(x, self._floor(self.fixed_scale))

        if v == "N1":
            x = x - x.mean()
            return (x / self._floor(x.std())).astype(np.float32)

        track = self.cache.get(w.event.evid, self.reference)

        return self._apply(x, self._floor(track.at_end(end_sample_of(w))))

    def windows(self, windows) -> np.ndarray:

        windows = list(windows)

        if not windows:
            return np.zeros((0, 0), dtype=np.float32)

        return np.stack([self.window(w) for w in windows])

    def stream(self, stream) -> np.ndarray:

        v = self.config.variant
        x = np.asarray(stream.data, dtype=np.float64)

        if v == "N1":
            raise NotImplementedError("N1 is per window; there is no stream form")

        if v == "N0":
            if self.fixed_scale is None:
                raise RuntimeError("N0 is not fitted")
            return self._apply(x, self._floor(self.fixed_scale))

        track = self.cache.get(stream.waveform_id, self.reference)
        scale = np.maximum(track.per_sample(), self.config.min_scale)

        return self._apply(x, scale)