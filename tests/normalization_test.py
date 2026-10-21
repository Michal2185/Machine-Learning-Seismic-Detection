"""Synthetic unit tests for the normalisation variants. Run: python tests/test_normalization.py"""

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.normalization import (
    NormConfig,
    Normalizer,
    ScaleCache,
    adaptive_scale_track,
    background_arrays_from_windows,
    estimate_fixed_scale,
    gap_mask,
    robust_std,
)

FS = 6.625
WIN = 398
STEP = 66
H = int(3600 * FS)               # samples per hour


@dataclass
class Ev:
    evid: str


@dataclass
class Win:
    data: np.ndarray
    end_sample: int
    label: int
    event: Ev


@dataclass
class Stream:
    waveform_id: str
    data: np.ndarray
    sampling_rate: float = FS
    gaps: list = None

    def __post_init__(self):
        if self.gaps is None:
            self.gaps = []


def noise(hours, sigma, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, int(hours * H)).astype(np.float32)


def make_windows(stream, labels=None, step=STEP):
    out = []
    for end in range(WIN, len(stream.data) + 1, step):
        lab = 0 if labels is None else labels(end)
        out.append(Win(stream.data[end - WIN:end], end, lab, Ev(stream.waveform_id)))
    return out


def burst(data, start_s, dur_s, gain, seed=1):
    d = data.copy()
    i0, i1 = int(start_s * FS), int((start_s + dur_s) * FS)
    d[i0:i1] = (d[i0:i1] * gain).astype(np.float32)
    return d


# ---------------------------------------------------------------- helpers

def test_robust_std_gaussian_and_outliers():
    x = np.random.default_rng(0).normal(0, 2.0, 100000)
    assert abs(robust_std(x) - 2.0) < 0.05
    x[:100] = 1e6
    assert abs(robust_std(x) - 2.0) < 0.05
    assert np.isnan(robust_std(np.array([np.nan])))


def test_gap_mask():
    m = gap_mask(1000, 10.0, [(10.0, 20.0), (95.0, 200.0)])
    assert m[100:200].all() and not m[:100].any() and not m[200:950].any()
    assert m[950:].all() and m.sum() == 100 + 50


def test_end_sample_nested_and_missing():
    from src.modules.normalization import end_sample_of

    @dataclass
    class Inner:
        end_sample: int

    @dataclass
    class Outer:
        window: Inner

    assert end_sample_of(Outer(Inner(123))) == 123
    try:
        end_sample_of(object())
        raise AssertionError("expected AttributeError")
    except AttributeError:
        pass


# ---------------------------------------------------------------- N0

def test_fixed_scale_median_of_waveforms():
    arrays = [noise(1, s, seed=i) for i, s in enumerate([1.0, 2.0, 100.0])]
    sc = estimate_fixed_scale(arrays, "median_of_waveforms")
    assert abs(sc - 2.0) < 0.1                     # the noisy day does not dominate


def test_fixed_scale_pooled_and_errors():
    arrays = [noise(1, 1.0, 1), noise(1, 3.0, 2)]
    assert 1.0 < estimate_fixed_scale(arrays, "pooled") < 3.0
    for bad in ([], [np.zeros(100)]):
        try:
            estimate_fixed_scale(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_n0_fit_uses_only_nonoverlapping_background_windows():
    s = Stream("a", noise(2, 1.0))
    wins = make_windows(s, labels=lambda e: 0)
    arrays = background_arrays_from_windows(wins)
    assert len(arrays) == 1
    assert len(arrays[0]) <= len(s.data)           # no sample counted twice
    assert len(arrays[0]) > 0.8 * len(s.data)
    # positives / ignore never enter the fit
    loud = Stream("b", noise(1, 1.0, 5) * 1000)
    bad = make_windows(loud, labels=lambda e: 1) + make_windows(loud, labels=lambda e: -1)
    assert background_arrays_from_windows(bad) == []


def test_n0_transform_and_fit():
    s = Stream("a", noise(2, 5.0))
    wins = make_windows(s)
    n = Normalizer(NormConfig("N0")).fit(wins)
    assert abs(n.fixed_scale - 5.0) < 0.2
    y = n.window(wins[10])
    assert y.dtype == np.float32 and y.shape == (WIN,)
    assert np.allclose(y, wins[10].data / n.fixed_scale, rtol=1e-5)


def test_n0_not_fitted_raises():
    n = Normalizer(NormConfig("N0"))
    try:
        n.window(Win(np.ones(WIN), WIN, 0, Ev("a")))
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_n0_keeps_day_to_day_ratio():
    quiet, loud = Stream("q", noise(1, 0.01)), Stream("l", noise(1, 3.0, 2))
    n = Normalizer(NormConfig("N0")).fit_streams([quiet, loud])
    r = n.stream(loud).std() / n.stream(quiet).std()
    assert 250 < r < 350                           # N0 does NOT remove the 300x spread


# ---------------------------------------------------------------- N1

def test_n1_zero_mean_unit_std_and_amplitude_invariance():
    x = np.random.default_rng(0).normal(3.0, 2.0, WIN)
    n = Normalizer(NormConfig("N1"))
    y = n.window(Win(x, WIN, 0, Ev("a")))
    assert abs(y.mean()) < 1e-6 and abs(y.std() - 1.0) < 1e-5
    y2 = n.window(Win(x * 1e-9, WIN, 0, Ev("a")))
    assert np.allclose(y, y2, atol=1e-4)


def test_n1_flat_window_is_finite_zero():
    n = Normalizer(NormConfig("N1"))
    y = n.window(Win(np.full(WIN, 7.0), WIN, 0, Ev("a")))
    assert np.isfinite(y).all() and np.allclose(y, 0.0)


def test_n1_has_no_stream_form():
    try:
        Normalizer(NormConfig("N1")).stream(Stream("a", noise(0.1, 1)))
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


# ---------------------------------------------------------------- N2 track

def rel(a, b):
    return abs(a - b) / b


def test_track_constant_noise():
    t = adaptive_scale_track(noise(8, 3.0), FS, [], NormConfig("N2"))
    # 65th percentile of the block-level spread: a few % above sigma, stable over time
    for frac in (0.3, 0.6, 1.0):
        assert 1.0 < t.scale[int(frac * (len(t.scale) - 1))] / 3.0 < 1.08
    assert t.elevated.mean() < 0.01
    assert not t.skipped.any()


def test_track_is_causal():
    cfg = NormConfig("N2")
    d = noise(4, 1.0)
    t1 = adaptive_scale_track(d, FS, [], cfg)
    k = int(2 * H)
    d2 = d.copy()
    d2[k:] = np.random.default_rng(9).normal(0, 50, len(d) - k)
    t2 = adaptive_scale_track(d2, FS, [], cfg)
    last = k // t1.block_len
    assert np.array_equal(t1.scale[: last + 1], t2.scale[: last + 1])
    assert not np.array_equal(t1.scale, t2.scale)


def test_track_event_does_not_inflate_estimate():
    d = burst(noise(8, 1.0), 4 * 3600, 1800, 20.0)          # 30 min event = 6 % of the window
    t = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    during = t.block_of(int((4 * 3600 + 1500) * FS))
    after = t.block_of(int(6 * 3600 * FS))
    assert t.scale[during] < 1.1 and t.scale[after] < 1.1
    assert t.elevated[t.block_of(int(4.2 * H)): t.block_of(int(4.4 * H))].mean() > 0.9


def test_track_tolerates_long_event_share():
    # an event + coda holding 30 % of the trailing window still leaves the estimate near sigma
    d = burst(noise(10, 1.0), 6 * 3600, 6480, 10.0)         # 1.8 h of 10x
    t = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    assert t.scale[t.block_of(int(7.8 * H))] < 1.25          # event share of window ~ 30 %


def test_track_adopts_new_regime():
    d = np.concatenate([noise(4, 1.0, 1), noise(10, 4.0, 2)])
    t = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    assert rel(t.scale[-1], 4.0 * 1.037) < 0.06
    assert t.scale[t.block_of(int(4.5 * H))] < 1.5           # not yet half an hour after the jump
    assert t.scale[t.block_of(int(8.0 * H))] > 3.5           # adopted within ~2-4 h


def test_track_flagged_gap_is_excluded():
    d = noise(8, 1.0)
    i0, i1 = 2 * H, 3 * H
    d[i0:i1] = np.linspace(0, 1e-3, i1 - i0).astype(np.float32)
    gaps = [(2 * 3600.0, 3 * 3600.0)]
    t = adaptive_scale_track(d, FS, gaps, NormConfig("N2"))
    assert t.skipped[t.block_of(int(2.5 * H))]
    assert rel(t.scale[t.block_of(int(3.1 * H))], 1.0) < 0.08


def test_track_unflagged_dead_stretches_are_ignored():
    # evid00002-like: long quiet / dead stretches that are NOT flagged as gaps
    d = noise(12, 1.0)
    for start_h in (1.0, 3.5, 6.0, 8.5):                     # 4 x 1 h dead of 12 h (33 %)
        d[int(start_h * H):int((start_h + 1.0) * H)] *= 1e-3
    t = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    for h in (5.0, 7.5, 9.5, 11.9):
        assert 0.88 < t.scale[t.block_of(int(h * H))] < 1.1, h   # ~5-8 % low at a 33-40 % dead share


def test_track_file_edge_ramps():
    # level ramps up over the first 40 min (filter / file start), like the real files
    d = noise(10, 1.0)
    n_ramp = int(2400 * FS)
    d[:n_ramp] *= np.linspace(1e-4, 1.0, n_ramp).astype(np.float32)
    t = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    assert rel(t.scale[t.block_of(int(5 * H))], 1.037) < 0.06
    assert t.elevated[t.block_of(int(1.5 * H)):].mean() < 0.01


def test_track_works_at_real_amplitudes():
    d = noise(8, 1.0) * 1e-10
    t = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    t2 = adaptive_scale_track(noise(8, 1.0), FS, [], NormConfig("N2"))
    assert not t.skipped.any()
    assert np.allclose(t.scale / 1e-10, t2.scale, rtol=1e-5)  # scale equivariant


def test_track_starts_and_flat_stream():
    t = adaptive_scale_track(np.zeros(5000, dtype=np.float32), FS, [], NormConfig("N2"))
    assert np.allclose(t.scale, 1.0) and t.skipped.all()
    d = noise(2, 2.0)
    d[: 3 * t.block_len] = 0.0                               # silent start
    t2 = adaptive_scale_track(d, FS, [], NormConfig("N2"))
    assert np.isfinite(t2.scale).all() and 2.0 < t2.scale[0] < 2.3
    # a file shorter than the minimum number of valid blocks has no estimate: falls back to 1
    short = adaptive_scale_track(noise(0.02, 2.0), FS, [], NormConfig("N2"))
    assert np.allclose(short.scale, 1.0)


def test_track_holds_estimate_over_long_gap():
    d = noise(40, 1.0)
    gaps = [(8 * 3600.0, 38 * 3600.0)]                        # longer than the 6 h window
    t = adaptive_scale_track(d, FS, gaps, NormConfig("N2"))
    assert np.isfinite(t.scale).all()
    assert rel(t.scale[t.block_of(int(37 * H))], t.scale[t.block_of(int(14 * H))]) < 1e-9 \
        or t.scale[t.block_of(int(37 * H))] > 0


def test_track_lookup_conventions():
    t = adaptive_scale_track(noise(1, 1.0), FS, [], NormConfig("N2"))
    L = t.block_len
    assert t.at_end(WIN) == t.scale[(WIN - 1) // L]
    assert t.at_end(10 ** 9) == t.scale[-1]                  # clipped, no IndexError
    assert len(t.per_sample()) == t.n_samples
    assert len(t.level) == len(t.scale) == len(t.elevated)


# ---------------------------------------------------------------- N2 / N3 on windows

def make_cache(streams, variant="N2", **kw):
    return ScaleCache.from_streams(streams, NormConfig(variant, **kw))


def test_n2_removes_day_to_day_amplitude():
    quiet, loud = Stream("q", noise(4, 0.01)), Stream("l", noise(4, 3.0, 2))
    cache = make_cache([quiet, loud])
    n = Normalizer(NormConfig("N2"), cache)
    out = []
    for s in (quiet, loud):
        ws = [w for w in make_windows(s) if w.end_sample > H]
        out.append(n.windows(ws).std())
    assert abs(out[0] - 1.0) < 0.1 and abs(out[1] - 1.0) < 0.1


def test_n2_event_window_stands_out():
    base = Stream("a", noise(5, 2.0))
    ev = Stream("a", burst(base.data, 3 * 3600, 600, 10.0))
    cache = make_cache([ev])
    n = Normalizer(NormConfig("N2"), cache)
    end = int((3 * 3600 + 300) * FS)
    quiet_end = int(2.5 * H)
    y_ev = n.window(Win(ev.data[end - WIN:end], end, 1, Ev("a")))
    y_bg = n.window(Win(ev.data[quiet_end - WIN:quiet_end], quiet_end, 0, Ev("a")))
    assert y_ev.std() > 8 * y_bg.std()


def test_n2_window_matches_scalar_division():
    s = Stream("a", noise(3, 2.0))
    cache = make_cache([s])
    n = Normalizer(NormConfig("N2"), cache)
    w = make_windows(s)[500]
    expect = w.data / cache.get("a").at_end(w.end_sample)
    assert np.allclose(n.window(w), expect, rtol=1e-5)


def test_stream_form_equals_per_sample_division():
    s = Stream("a", noise(2, 2.0))
    cache = make_cache([s])
    n = Normalizer(NormConfig("N2"), cache)
    y = n.stream(s)
    assert y.dtype == np.float32 and y.shape == s.data.shape
    assert np.allclose(y, s.data / cache.get("a").per_sample(), rtol=1e-4)


def test_n3_asinh_properties():
    s = Stream("a", burst(noise(4, 1.0), 2 * 3600, 600, 50.0))
    cache = make_cache([s])
    n2 = Normalizer(NormConfig("N2"), cache)
    n3 = Normalizer(NormConfig("N3"), cache)
    w = make_windows(s)[int(2.05 * 3600 * FS / STEP)]
    z, y = n2.window(w), n3.window(w)
    assert np.allclose(y, np.arcsinh(z), atol=1e-5)
    assert np.all(np.sign(y) == np.sign(z)) or np.all(np.abs(z[np.sign(y) != np.sign(z)]) < 1e-6)
    big = np.abs(z) > 20
    assert big.any() and np.all(np.abs(y[big]) < 0.25 * np.abs(z[big]))   # peaks compressed
    small = np.abs(z) < 0.1
    assert np.allclose(y[small], z[small], atol=1e-3)                    # linear near zero


def test_n3_knee():
    s = Stream("a", noise(2, 1.0))
    cache = make_cache([s])
    w = make_windows(s)[300]
    a = Normalizer(NormConfig("N3", asinh_knee=1.0), cache).window(w)
    b = Normalizer(NormConfig("N3", asinh_knee=3.0), cache).window(w)
    z = Normalizer(NormConfig("N2"), cache).window(w)
    assert np.allclose(b, np.arcsinh(z / 3.0), atol=1e-5) and not np.allclose(a, b)


# ---------------------------------------------------------------- config / errors

def test_config_validation():
    for kw in (dict(variant="N9"), dict(quantile=0.0), dict(quantile=1.0), dict(window_s=0),
               dict(min_valid_blocks=0), dict(elevated_ratio=1.0), dict(asinh_knee=0),
               dict(fixed_scale_stat="mean"), dict(max_gap_fraction=0.0)):
        try:
            NormConfig(**kw)
            raise AssertionError(f"expected ValueError for {kw}")
        except ValueError:
            pass


def test_n2_needs_cache_and_known_waveform():
    for v in ("N2", "N3"):
        try:
            Normalizer(NormConfig(v))
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    n = Normalizer(NormConfig("N2"), ScaleCache(NormConfig("N2")))
    try:
        n.window(Win(np.ones(WIN), WIN, 0, Ev("nope")))
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_determinism_and_empty():
    s = Stream("a", noise(2, 1.0))
    c1, c2 = make_cache([s]), make_cache([s])
    assert np.array_equal(c1.get("a").scale, c2.get("a").scale)
    assert Normalizer(NormConfig("N1")).windows([]).shape[0] == 0


# ---------------------------------------------------------------- runner

def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run()