import dataclasses
import re
import zlib
from dataclasses import dataclass

import numpy as np

MAD_TO_STD = 1.4826


# ----------------------------------------------------------------------
# Perturbations of one continuous (already preprocessed) stream
# ----------------------------------------------------------------------

def _robust_std(x) -> float:
    """MAD based standard deviation; falls back to the plain std for degenerate data."""

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return 0.0

    v = MAD_TO_STD * float(np.median(np.abs(x - np.median(x))))

    return v if v > 0 else float(x.std())


def rng_for(waveform_id: str, name: str) -> np.random.Generator:
    """
    Deterministic generator per (waveform, perturbation): every model sees EXACTLY the same perturbed
    data (crc32 is stable across processes, unlike hash()).
    """

    return np.random.default_rng(zlib.crc32(f"{name}|{waveform_id}".encode()))


def apply_gain(data, gain: float) -> np.ndarray:
    return (np.asarray(data, dtype=np.float64) * float(gain)).astype(np.float32)


def add_noise(data, fs: float, alpha: float, rng, highpass_hz: float = 0.2) -> np.ndarray:
    """
    Adds white noise passed through the same causal 4th order highpass as the preprocessing (so it has
    no energy below 0.2 Hz), scaled to alpha x the robust std of the stream (= its background level).
    """

    from scipy.signal import butter, sosfilt

    x = np.asarray(data, dtype=np.float64)
    noise = sosfilt(butter(4, highpass_hz, "highpass", fs=fs, output="sos"), rng.standard_normal(len(x)))
    sd = noise.std()

    if sd > 0:
        noise /= sd

    return (x + alpha * _robust_std(x) * noise).astype(np.float32)


def _gap_mask(n: int, fs: float, gaps) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)

    for a, b in gaps:
        mask[max(0, int(np.floor(a * fs))):min(n, int(np.ceil(b * fs)))] = True

    return mask


def background_psd(data, fs: float, gaps=(), block_s: float = 60.0, nperseg: int = 256):
    """
    Power spectrum of the stream's BACKGROUND: mean Welch estimate over the blocks whose robust level lies
    within 0.5x-1.5x of the median block level (events and dead stretches fall outside) and that are at
    most 10 % interpolated. Returns (freqs, psd) or None when the stream is too short.
    """

    from scipy.signal import welch

    x = np.asarray(data, dtype=np.float64)
    blen = int(round(block_s * fs))
    nb = len(x) // blen

    if nb < 4 or blen < 16:
        return None

    blocks = x[:nb * blen].reshape(nb, blen)
    masked = _gap_mask(len(x), fs, gaps)[:nb * blen].reshape(nb, blen).mean(axis=1)
    levels = np.array([_robust_std(b) for b in blocks])

    ok = (levels > 0) & (masked <= 0.10)

    if ok.sum() < 3:
        return None

    med = np.median(levels[ok])
    sel = ok & (levels > 0.5 * med) & (levels < 1.5 * med)

    if sel.sum() < 3:
        sel = ok

    freqs, psd = welch(blocks[sel], fs=fs, nperseg=min(nperseg, blen), axis=-1)

    return freqs, psd.mean(axis=0)


def shaped_noise(n: int, fs: float, freqs, psd, rng) -> np.ndarray:
    """Gaussian noise with the given power spectrum (random phases), scaled to unit std."""

    spec = np.fft.rfft(rng.standard_normal(n))
    amp = np.sqrt(np.interp(np.fft.rfftfreq(n, 1.0 / fs), freqs, psd))
    amp[0] = 0.0

    x = np.fft.irfft(spec * amp, n)
    sd = x.std()

    return x / sd if sd > 0 else x


def add_matched_noise(data, fs: float, alpha: float, rng, gaps=()) -> np.ndarray:
    """
    Adds noise with the SAME power spectrum as the stream's own background, at alpha x the robust std of the
    stream (the level `add_noise` uses), so only the spectral shape differs between the two tests. Falls back
    to `add_noise` for streams too short to estimate a spectrum.
    """

    est = background_psd(data, fs, gaps)

    if est is None:
        return add_noise(data, fs, alpha, rng)

    x = np.asarray(data, dtype=np.float64)

    return (x + alpha * _robust_std(x) * shaped_noise(len(x), fs, *est, rng)).astype(np.float32)


def inject_gaps(data, fs: float, fraction: float, rng, min_s: float = 20.0, max_s: float = 120.0):
    """
    Replaces random, non-overlapping stretches of 20-120 s by linear interpolation (like the real
    interpolated dropouts) until `fraction` of the stream is covered. Returns (data, [(start_s, end_s)]).
    """

    out = np.asarray(data, dtype=np.float64).copy()
    n = len(out)
    gaps = []
    mask = np.zeros(n, dtype=bool)
    covered, tries = 0.0, 0
    target = fraction * n / fs

    while covered < target and tries < 20000:

        tries += 1
        length = int(rng.uniform(min_s, max_s) * fs)

        if length < 2 or n < length + 3:
            break

        i0 = int(rng.integers(1, n - length - 1))
        i1 = i0 + length

        if mask[i0 - 1:i1 + 1].any():
            continue

        mask[i0:i1] = True
        out[i0:i1] = np.linspace(out[i0 - 1], out[i1], length + 2)[1:-1]     # straight line between the neighbours
        gaps.append((i0 / fs, i1 / fs))
        covered += length / fs

    return out.astype(np.float32), gaps


def apply_jitter(data, sigma_samples: float, rng) -> np.ndarray:
    """Samples taken at jittered times (sigma in sample intervals), linearly interpolated back to the regular grid."""

    x = np.asarray(data, dtype=np.float64)
    n = len(x)
    grid = np.arange(n, dtype=np.float64)

    return np.interp(grid + rng.normal(0.0, sigma_samples, n), grid, x).astype(np.float32)


def apply_lowpass(data, fs: float, cutoff_hz: float) -> np.ndarray:
    """Causal 4th order Butterworth lowpass: bandwidth limit (proxy of decimation, e.g. to about 2.65 Hz)."""

    from scipy.signal import butter, sosfilt

    sos = butter(4, cutoff_hz, "lowpass", fs=fs, output="sos")

    return sosfilt(sos, np.asarray(data, dtype=np.float64)).astype(np.float32)


@dataclass(frozen=True)
class Perturbation:
    """kind: clean | gain | noise | mnoise | gaps | jitter | lowpass; value: factor, alpha, share, sigma or Hz."""

    name: str
    kind: str
    value: float = 0.0

    def apply(self, stream):
        """A copy of the stream (any dataclass with data / sampling_rate / gaps / waveform_id)."""

        if self.kind == "clean":
            return stream

        fs = float(stream.sampling_rate)
        rng = rng_for(stream.waveform_id, self.name)
        gaps = list(stream.gaps)

        if self.kind == "gain":
            data = apply_gain(stream.data, self.value)

        elif self.kind == "noise":
            data = add_noise(stream.data, fs, self.value, rng)

        elif self.kind == "mnoise":
            data = add_matched_noise(stream.data, fs, self.value, rng, gaps)

        elif self.kind == "gaps":
            data, new = inject_gaps(stream.data, fs, self.value, rng)
            gaps += new                                    # alarms inside them are neutral, like real dropouts

        elif self.kind == "jitter":
            data = apply_jitter(stream.data, self.value, rng)

        elif self.kind == "lowpass":
            data = apply_lowpass(stream.data, fs, self.value)

        else:
            raise ValueError(f"unknown perturbation kind {self.kind!r}")

        return dataclasses.replace(stream, data=data, gaps=gaps)


_NAME = re.compile(r"^(clean|gain|mnoise|noise|gaps|jitter|lp)([0-9.]*)$")


def make_perturbation(name: str) -> Perturbation:
    """'clean', 'gain3', 'gain0.3', 'noise1' (white), 'mnoise1' (background-spectrum noise), 'gaps10' (percent of
    the stream), 'jitter0.25', 'lp1.3' (Hz)."""

    m = _NAME.match(name)

    if not m:
        raise ValueError(f"cannot parse perturbation {name!r}")

    kind, val = m.group(1), m.group(2)

    if kind == "clean":
        return Perturbation(name, "clean")

    if val == "":
        raise ValueError(f"{name!r} needs a value")

    v = float(val)

    if kind == "gaps":
        return Perturbation(name, "gaps", v / 100.0)

    return Perturbation(name, {"lp": "lowpass"}.get(kind, kind), v)


DEFAULT_PERTURBATIONS = (
    "clean", "gain0.3", "gain3", "gain10", "noise0.5", "noise1", "noise2",
    "gaps2", "gaps10", "jitter0.25", "lp1.3", "lp0.8",
)


# ----------------------------------------------------------------------
# Windows of a perturbed stream
# ----------------------------------------------------------------------

def _end_sample(w) -> int:
    """Same lookup as normalization.end_sample_of: attribute of the record or of its nested .window."""

    for obj in (w, getattr(w, "window", None)):
        if obj is not None and hasattr(obj, "end_sample"):
            return int(obj.end_sample)

    raise AttributeError(f"cannot find end_sample on {type(w).__name__}")


class PerturbedWindow:
    """A window record with its samples replaced; every other attribute (label, event, end_time...) is delegated."""

    __slots__ = ("_base", "data")

    def __init__(self, base, data):
        self._base = base
        self.data = data

    def __getattr__(self, name):
        return getattr(self._base, name)


def perturbed_windows(windows, streams_by_id) -> list:
    """
    Re-cuts every window from the PERTURBED stream: samples [end_sample - T, end_sample) with T = len(w.data).
    The window records are not modified.
    """

    windows = list(windows)
    out = [None] * len(windows)
    groups = {}

    for i, w in enumerate(windows):
        groups.setdefault((w.event.evid, len(w.data)), []).append(i)

    for (evid, length), idx in groups.items():

        data = np.asarray(streams_by_id[evid].data)
        ends = np.array([_end_sample(windows[i]) for i in idx], dtype=np.int64)

        if ends.min() - length < 0 or ends.max() > len(data):
            raise ValueError(f"window outside the stream {evid!r}: end_sample range "
                             f"[{ends.min()}, {ends.max()}], stream length {len(data)}")

        view = np.lib.stride_tricks.sliding_window_view(data, length)
        segs = view[ends - length].copy()

        for j, i in enumerate(idx):
            out[i] = PerturbedWindow(windows[i], segs[j])

    return out


def check_window_convention(windows, streams_by_id, n: int = 50) -> None:
    """
    Raises unless re-cutting the CLEAN stream reproduces w.data, i.e. unless end_sample is the exclusive
    end of the window in that stream. Without this a perturbation could silently shift the windows.
    """

    windows = list(windows)
    step = max(1, len(windows) // n)
    sample = windows[::step][:n]

    for w, r in zip(sample, perturbed_windows(sample, streams_by_id)):
        if not np.allclose(np.asarray(w.data), r.data, rtol=1e-5, atol=0.0):
            raise AssertionError(
                "end_sample does not match the window data: windows cannot be re-cut from the stream "
                f"(window of waveform {w.event.evid!r}, end_sample {_end_sample(w)})"
            )


# ----------------------------------------------------------------------
# Frozen detectors on perturbed test streams
# ----------------------------------------------------------------------

@dataclass
class StressRun:
    pooled: dict               # {(perturbation name, budget): pooled result}
    oracle: dict               # same keys, threshold re-chosen on the perturbed test data (diagnostic only)
    thresholds: list           # per fold {budget: threshold}, chosen on CLEAN validation


def stress_cross_validate(
    bundle,
    detector,
    evaluator,
    budgets,
    perturbations,
    build_series,
    pool_results,
    oracle: bool = False,
    select_group: str = "all",
    verbose: bool = True,
) -> StressRun:
    """
    Per fold: fit on the TRAINING windows, choose the threshold of every false-alarm budget on the CLEAN
    validation streams (exactly like cross_validate_budgets with one candidate), then apply the SAME
    frozen detector and threshold to the test streams after each perturbation (the clean one included).
    Nothing is retuned on perturbed data. With oracle=True the threshold is also re-chosen on the
    perturbed test streams: an optimistic diagnostic that separates a calibration shift from a loss of
    separability; never report it as a result.
    """

    budgets = list(budgets)
    perturbations = list(perturbations)

    pooled = {(p.name, b): [] for p in perturbations for b in budgets}
    orc = {(p.name, b): [] for p in perturbations for b in budgets} if oracle else {}
    thresholds = []

    for k in range(bundle.folds.n_folds):

        split = bundle.fold_split(k)

        val_streams = bundle.eval_streams(split.validation_events)
        test_streams = bundle.eval_streams(split.test_events)

        val_by_id = {s.waveform_id: s for s in val_streams}
        test_by_id = {s.waveform_id: s for s in test_streams}

        detector.fit(split.train)

        val_series = build_series(split.validation, detector.score(split.validation, val_by_id))
        curve = evaluator.curve(val_streams, val_series)

        thr = {b: evaluator.select_threshold(curve, max_fa_per_day=b, group=select_group) for b in budgets}
        thresholds.append(thr)

        if any(p.kind != "clean" for p in perturbations):
            check_window_convention(split.test, test_by_id)

        for p in perturbations:

            streams_p = [p.apply(s) for s in test_streams]
            by_id_p = {s.waveform_id: s for s in streams_p}

            windows_p = split.test if p.kind == "clean" else perturbed_windows(split.test, by_id_p)

            series = build_series(windows_p, detector.score(windows_p, by_id_p))

            for b in budgets:

                res = evaluator.evaluate(streams_p, series, thr[b])
                pooled[(p.name, b)].append(res)

                if verbose:
                    print(f"[{detector.name} @ {b:g}/day] fold {k} {p.name}: "
                          f"recall {res.recall(select_group):.2f} FA/day {res.fa_per_day:.2f}")

            if oracle:
                curve_p = evaluator.curve(streams_p, series)
                for b in budgets:
                    t = evaluator.select_threshold(curve_p, max_fa_per_day=b, group=select_group)
                    orc[(p.name, b)].append(evaluator.evaluate(streams_p, series, t))

    return StressRun(
        pooled={key: pool_results(v) for key, v in pooled.items()},
        oracle={key: pool_results(v) for key, v in orc.items()},
        thresholds=thresholds,
    )