"""
python -m tests.test_stress

Synthetic tests of src/modules/stress: the perturbations, re-cutting windows from a perturbed stream, and the
frozen-threshold loop (with fake bundle / evaluator / detector, so no data and no torch are needed).
"""

import dataclasses
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.stress.module import (
    DEFAULT_PERTURBATIONS, PerturbedWindow, add_matched_noise, add_noise, background_psd, shaped_noise, apply_gain, apply_jitter, apply_lowpass,
    check_window_convention, inject_gaps, make_perturbation, perturbed_windows, rng_for, stress_cross_validate,
)

FS = 6.625


@dataclasses.dataclass
class FakeStream:
    waveform_id: str
    data: np.ndarray
    sampling_rate: float
    gaps: list
    events: list


def make_stream(wid="w1", n=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = rng.normal(0, 1.0, n) + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return FakeStream(wid, x.astype(np.float32), FS, [], [])


def make_windows(stream, length=64, step=16):
    out = []
    for end in range(length, len(stream.data) + 1, step):
        out.append(SimpleNamespace(data=stream.data[end - length:end].copy(), label=0, end_sample=end,
                                   event=SimpleNamespace(evid=stream.waveform_id), end_time=end / FS))
    return out


# ---------------------------------------------------------------- perturbations

def test_gain_exact():
    s = make_stream()
    p = make_perturbation("gain3").apply(s)
    assert np.allclose(p.data, 3 * s.data, rtol=1e-6) and p.data.dtype == np.float32
    assert make_perturbation("gain0.3").value == 0.3


def test_clean_is_identity():
    s = make_stream()
    assert make_perturbation("clean").apply(s) is s


def test_deterministic_per_waveform_and_name():
    s = make_stream()
    a = make_perturbation("noise1").apply(s).data
    b = make_perturbation("noise1").apply(s).data
    c = make_perturbation("noise2").apply(s).data
    d = make_perturbation("noise1").apply(make_stream("w2")).data
    assert np.array_equal(a, b) and not np.allclose(a, c) and not np.allclose(a, d)
    assert rng_for("w1", "x").integers(1 << 30) == rng_for("w1", "x").integers(1 << 30)


def test_noise_level_and_no_input_mutation():
    s = make_stream(n=60000)
    before = s.data.copy()
    out = add_noise(s.data, FS, 1.0, np.random.default_rng(0))
    assert np.array_equal(s.data, before)
    added = out.astype(np.float64) - s.data
    ratio = added.std() / np.std(s.data)
    assert 0.8 < ratio < 1.3, ratio
    assert abs(added.mean()) < 0.05


def test_gaps_cover_fraction_are_linear_and_listed():
    s = make_stream(n=60000)
    data, gaps = inject_gaps(s.data, FS, 0.10, np.random.default_rng(1))
    covered = sum(b - a for a, b in gaps) * FS
    assert 0.09 * len(data) <= covered <= 0.14 * len(data), covered / len(data)
    for a, b in gaps:
        i0, i1 = int(round(a * FS)), int(round(b * FS))
        seg = data[i0 - 1:i1 + 1].astype(np.float64)
        assert np.allclose(np.diff(seg, 2), 0.0, atol=1e-4 * np.abs(seg).max())     # straight line
        assert 18 <= (b - a) <= 121
    flat = sorted(gaps)
    assert all(flat[i][1] <= flat[i + 1][0] for i in range(len(flat) - 1))          # no overlap
    p = make_perturbation("gaps10").apply(s)
    assert len(p.gaps) > 0 and all(b > a for a, b in p.gaps)
    assert s.gaps == []                                                              # original untouched


def test_jitter_small_for_slow_signal_and_exact_for_zero():
    t = np.arange(5000)
    x = np.sin(2 * np.pi * 0.05 * t / FS).astype(np.float32)
    assert np.allclose(apply_jitter(x, 0.0, np.random.default_rng(0)), x, atol=1e-6)
    err = np.abs(apply_jitter(x, 0.25, np.random.default_rng(0)) - x)
    assert err.max() < 0.05


def test_lowpass_attenuates_high_passes_low():
    n = 20000
    t = np.arange(n) / FS
    lo, hi = np.sin(2 * np.pi * 0.3 * t), np.sin(2 * np.pi * 2.5 * t)
    out_lo = apply_lowpass(lo, FS, 1.3)[5000:]
    out_hi = apply_lowpass(hi, FS, 1.3)[5000:]
    assert out_lo.std() > 0.9 * lo.std()
    assert out_hi.std() < 0.2 * hi.std()


def test_make_perturbation_parsing():
    for name in DEFAULT_PERTURBATIONS:
        assert make_perturbation(name).name == name
    assert make_perturbation("gaps10").value == 0.10 and make_perturbation("lp1.3").kind == "lowpass"
    for bad in ("gain", "foo3", "noise-1x"):
        try:
            make_perturbation(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


# ---------------------------------------------------------------- windows

def test_recut_clean_reproduces_windows():
    s = make_stream()
    w = make_windows(s)
    r = perturbed_windows(w, {s.waveform_id: s})
    assert all(np.array_equal(a.data, b.data) for a, b in zip(w, r))
    check_window_convention(w, {s.waveform_id: s})


def test_recut_uses_perturbed_stream_and_delegates():
    s = make_stream()
    w = make_windows(s)
    p = make_perturbation("gain3").apply(s)
    r = perturbed_windows(w, {s.waveform_id: p})
    assert isinstance(r[5], PerturbedWindow)
    assert np.allclose(r[5].data, 3 * w[5].data, rtol=1e-6)
    assert r[5].label == 0 and r[5].event.evid == "w1" and r[5].end_sample == w[5].end_sample
    assert np.array_equal(w[5].data, s.data[w[5].end_sample - 64:w[5].end_sample])   # records untouched


def test_convention_check_catches_shifted_end_sample():
    s = make_stream()
    w = make_windows(s)
    for x in w:
        x.end_sample += 3
    try:
        check_window_convention(w, {s.waveform_id: s})
    except AssertionError:
        return
    raise AssertionError("a shifted end_sample must be detected")


def test_recut_rejects_window_outside_stream():
    s = make_stream(n=1000)
    w = make_windows(s)
    w[-1].end_sample = 5000
    try:
        perturbed_windows(w, {s.waveform_id: s})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------- frozen loop (fakes)

class FakeEval:
    """series = window scores; evaluate counts alarms (score > thr); curve = the score values."""

    def curve(self, streams, series):
        return np.sort(np.asarray(series))

    def select_threshold(self, curve, max_fa_per_day, group="all"):
        # threshold leaving the highest `max_fa_per_day` percent of scores above it
        return float(np.quantile(curve, 1.0 - max_fa_per_day / 100.0))

    def evaluate(self, streams, series, thr):
        n = int((np.asarray(series) > thr).sum())
        return SimpleNamespace(n_alarm=n, thr=thr, mean_data=float(np.mean([np.abs(s.data).mean() for s in streams])),
                               recall=lambda g="all": 0.0, fa_per_day=0.0)


class RmsDetector:
    name = "rms"

    def __init__(self):
        self.fits = 0

    def fit(self, windows):
        self.fits += 1
        return self

    def score(self, windows, streams_by_id):
        return np.array([np.sqrt(np.mean(np.asarray(w.data, dtype=np.float64) ** 2)) for w in windows])


class FakeBundle:

    def __init__(self):
        self.streams = {f"w{i}": make_stream(f"w{i}", n=8000, seed=i) for i in range(4)}
        self.folds = SimpleNamespace(n_folds=2)

    def fold_split(self, k):
        ids = sorted(self.streams)
        test, val, train = [ids[k]], [ids[(k + 1) % 4]], [i for i in ids if i not in (ids[k], ids[(k + 1) % 4])]
        wins = lambda keep: [w for i in keep for w in make_windows(self.streams[i])]
        return SimpleNamespace(train=wins(train), validation=wins(val), test=wins(test),
                               validation_events=val, test_events=test)

    def eval_streams(self, ids):
        return [dataclasses.replace(self.streams[i]) for i in ids]


def run_fake(names, oracle=False):
    det = RmsDetector()
    perts = [make_perturbation(n) for n in names]
    run = stress_cross_validate(FakeBundle(), det, FakeEval(), [1.0, 5.0], perts,
                                build_series=lambda w, s: np.asarray(s), pool_results=lambda rs: rs,
                                oracle=oracle, verbose=False)
    return det, run


def test_loop_structure_fits_once_per_fold_and_freezes_thresholds():
    det, run = run_fake(["clean", "gain3"])
    assert det.fits == 2                                                  # once per fold, not per perturbation
    assert set(run.pooled) == {("clean", 1.0), ("clean", 5.0), ("gain3", 1.0), ("gain3", 5.0)}
    for clean, g3 in zip(run.pooled[("clean", 1.0)], run.pooled[("gain3", 1.0)]):
        assert clean.thr == g3.thr                                        # same frozen threshold
        assert g3.n_alarm > clean.n_alarm                                 # 3x amplitude, same threshold: more alarms
        assert np.isclose(g3.mean_data, 3 * clean.mean_data, rtol=1e-5)   # evaluator got the perturbed streams
    assert len(run.thresholds) == 2 and set(run.thresholds[0]) == {1.0, 5.0}


def test_clean_uses_original_windows_exactly():
    _, run = run_fake(["clean"])
    r = run.pooled[("clean", 5.0)][0]
    bundle = FakeBundle()
    split = bundle.fold_split(0)
    val = RmsDetector().score(split.validation, None)
    thr = FakeEval().select_threshold(np.sort(val), 5.0)
    assert r.thr == thr


def test_oracle_threshold_follows_perturbed_data():
    _, run = run_fake(["gain3"], oracle=True)
    for frozen, orc in zip(run.pooled[("gain3", 5.0)], run.oracle[("gain3", 5.0)]):
        assert orc.thr > frozen.thr                                       # re-chosen on 3x data
        assert orc.n_alarm <= frozen.n_alarm


def test_gaps_listed_for_evaluator():
    seen = {}

    class Spy(FakeEval):
        def evaluate(self, streams, series, thr):
            seen["gaps"] = [len(s.gaps) for s in streams]
            return super().evaluate(streams, series, thr)

    stress_cross_validate(FakeBundle(), RmsDetector(), Spy(), [5.0], [make_perturbation("gaps10")],
                          build_series=lambda w, s: np.asarray(s), pool_results=lambda rs: rs, verbose=False)
    assert all(g > 0 for g in seen["gaps"])


# ---------------------------------------------------------------- matched (background-spectrum) noise

def coloured_stream(wid="c1", n=120000, rho=0.9, seed=0, bursts=True):
    """AR(1) background (strong low-frequency content, like the real noise) with a few loud bursts."""

    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1.0, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    if bursts:
        for c in rng.integers(2000, n - 2000, 12):
            x[c:c + 800] *= 15.0
    return FakeStream(wid, x.astype(np.float32), FS, [], [])


def zero_cross_rate(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return float(((x[1:] >= 0) != (x[:-1] >= 0)).mean())


def band_ratio(x, lo=(0.1, 0.8), hi=(2.0, 3.2)):
    from scipy.signal import welch
    f, p = welch(np.asarray(x, dtype=np.float64), fs=FS, nperseg=512)
    b = lambda r: p[(f >= r[0]) & (f <= r[1])].mean()
    return b(lo) / b(hi)


def test_background_psd_ignores_bursts_and_matches_level():
    s = coloured_stream()
    f, psd = background_psd(s.data, FS, [])
    clean = coloured_stream(bursts=False)
    f2, psd2 = background_psd(clean.data, FS, [])
    # bursts must not inflate the estimate (selection of quiet blocks)
    assert 0.6 < psd.mean() / psd2.mean() < 1.6, psd.mean() / psd2.mean()
    assert f[0] == 0.0 and psd[1] > 20 * psd[-1]                    # low-frequency dominated


def test_shaped_noise_reproduces_the_spectrum_shape():
    s = coloured_stream(bursts=False)
    f, psd = background_psd(s.data, FS, [])
    n = shaped_noise(120000, FS, f, psd, np.random.default_rng(3))
    assert abs(n.std() - 1.0) < 1e-6
    ref, got = band_ratio(s.data), band_ratio(n)
    assert 0.6 < got / ref < 1.7, (ref, got)                          # same low/high balance


def test_matched_noise_level_equals_white_noise_level():
    s = coloured_stream(bursts=False)
    m = add_matched_noise(s.data, FS, 1.0, np.random.default_rng(1)).astype(np.float64) - s.data
    w = add_noise(s.data, FS, 1.0, np.random.default_rng(1)).astype(np.float64) - s.data
    assert 0.9 < m.std() / w.std() < 1.1, (m.std(), w.std())          # same level: only the spectrum differs


def test_matched_noise_keeps_zero_crossing_rate_white_noise_does_not():
    s = coloured_stream(bursts=False)
    base = zero_cross_rate(s.data)
    m = zero_cross_rate(add_matched_noise(s.data, FS, 1.0, np.random.default_rng(2)))
    w = zero_cross_rate(add_noise(s.data, FS, 1.0, np.random.default_rng(2)))
    assert abs(m - base) / base < 0.2, (base, m)
    assert w > 1.5 * base, (base, w)


def test_matched_noise_perturbation_end_to_end():
    s = coloured_stream()
    p = make_perturbation("mnoise1")
    assert p.kind == "mnoise" and p.value == 1.0
    a, b = p.apply(s), p.apply(s)
    assert np.array_equal(a.data, b.data) and a.data.dtype == np.float32
    assert not np.array_equal(a.data, s.data) and a.gaps == s.gaps
    c = make_perturbation("mnoise2").apply(s)
    assert (c.data.astype(np.float64) - s.data).std() > 1.8 * (a.data.astype(np.float64) - s.data).std()


def test_matched_noise_falls_back_on_short_stream():
    s = make_stream(n=500)
    out = add_matched_noise(s.data, FS, 1.0, np.random.default_rng(0))
    assert out.shape == s.data.shape and np.isfinite(out).all()


def test_gap_blocks_are_excluded_from_the_psd():
    s = coloured_stream(bursts=False)
    d = s.data.copy()
    a, b = 600.0, 3000.0                                              # seconds: a long flat (interpolated) stretch
    d[int(a * FS):int(b * FS)] = 0.0
    f, p_gap = background_psd(d, FS, [(a, b)])
    f, p_ref = background_psd(s.data, FS, [])
    assert 0.6 < p_gap.mean() / p_ref.mean() < 1.6


if __name__ == "__main__":
    fails = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn(); print(f"ok   {name}")
        except Exception:
            fails += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)