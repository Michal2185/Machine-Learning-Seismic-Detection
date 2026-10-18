"""Synthetic unit tests for the 6-feature extractor. Run: python tests/test_features.py"""

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.dnn_features import FEATURE_NAMES, N_FEATURES, extract_features

FS = 6.625
N = 398
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def sine(k, amp=1.0, phase=0.3, n=N, fs=FS):
    """Sine with exactly k cycles in the window (bin-centred)."""
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * k * t / n + phase)


def f(x, fs=FS):
    return extract_features(x, fs)


def test_names_and_shape():
    assert FEATURE_NAMES == ("energy", "rms", "ptp", "crest", "dom_freq", "zcr") and N_FEATURES == 6
    assert f(sine(8)).shape == (6,) and f(sine(8)).dtype == np.float32
    assert f(np.stack([sine(8), sine(20)])).shape == (2, 6)


def test_sine_values():
    A, k = 2.0, 8
    v = f(sine(k, A))
    assert abs(v[IDX["energy"]] - A ** 2 * N / 2) / (A ** 2 * N / 2) < 1e-3
    assert abs(v[IDX["rms"]] - A / np.sqrt(2)) < 1e-3
    assert abs(v[IDX["ptp"]] - 2 * A) / (2 * A) < 0.02
    assert abs(v[IDX["crest"]] - np.sqrt(2)) < 0.05
    assert abs(v[IDX["dom_freq"]] - k * FS / N) < 1e-6
    crossings = v[IDX["zcr"]] * N / FS
    assert abs(crossings - 2 * k) <= 1


def test_dominant_frequency_tracks_bin_and_fs():
    for k in (5, 20, 40, 100):
        assert abs(f(sine(k))[IDX["dom_freq"]] - k * FS / N) < 1e-6
    assert abs(f(sine(20), fs=2 * FS)[IDX["dom_freq"]] - 2 * 20 * FS / N) < 1e-6
    # zcr is per second, so it doubles with fs for the same samples
    assert abs(f(sine(20), fs=2 * FS)[IDX["zcr"]] / f(sine(20))[IDX["zcr"]] - 2.0) < 1e-4


def test_dc_is_ignored():
    a = f(sine(8))
    b = f(sine(8) + 1000.0)
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)
    assert abs(b[IDX["dom_freq"]] - 8 * FS / N) < 1e-6


def test_scaling_behaviour():
    x = np.random.default_rng(0).normal(size=N)
    a, b = f(x), f(7.0 * x)
    assert abs(b[IDX["energy"]] / a[IDX["energy"]] - 49.0) < 1e-3
    assert abs(b[IDX["rms"]] / a[IDX["rms"]] - 7.0) < 1e-4
    assert abs(b[IDX["ptp"]] / a[IDX["ptp"]] - 7.0) < 1e-4
    for n in ("crest", "dom_freq", "zcr"):
        assert abs(b[IDX[n]] - a[IDX[n]]) < 1e-4 * max(1.0, abs(a[IDX[n]]))


def test_impulse_crest_factor():
    x = np.zeros(N)
    x[100] = 5.0
    v = f(x)
    assert abs(v[IDX["crest"]] - np.sqrt(N - 1)) < 1e-3
    assert abs(v[IDX["energy"]] - 25.0 * (1 - 1 / N)) < 1e-3


def test_white_noise_statistics():
    x = np.random.default_rng(1).normal(size=(200, N))
    v = f(x)
    assert abs(v[:, IDX["rms"]].mean() - 1.0) < 0.02
    assert abs(v[:, IDX["zcr"]].mean() * N / FS / (N - 1) - 0.5) < 0.02    # ~half of the pairs
    assert 3.0 < v[:, IDX["crest"]].mean() < 4.5


def test_alternating_signal_zcr():
    x = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    assert abs(f(x)[IDX["zcr"]] * N / FS - (N - 1)) < 1e-3
    assert abs(f(x)[IDX["dom_freq"]] - FS / 2) < 0.05


def test_flat_and_zero_windows_are_zero():
    for x in (np.zeros(N), np.full(N, 3.7), np.full(N, 1e-9)):
        v = f(x)
        assert np.isfinite(v).all() and np.allclose(v, 0.0), x[0]


def test_batch_equals_single():
    X = np.random.default_rng(2).normal(size=(5, N)) * np.arange(1, 6)[:, None]
    X[2] = 4.0
    batch = f(X)
    for i in range(5):
        assert np.allclose(batch[i], f(X[i]), rtol=1e-5, atol=1e-6)


def test_n1_makes_energy_and_rms_constant():
    # documents a modelling fact: after per-window standardisation (N1) the
    # energy and RMS features carry no information
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, N)) * rng.uniform(0.1, 50, size=(20, 1))
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    v = f(Z)
    assert np.allclose(v[:, IDX["rms"]], 1.0, atol=1e-4)
    assert np.allclose(v[:, IDX["energy"]], N, rtol=1e-4)


def test_invalid_input():
    for bad, fs in ((np.array([1.0, np.nan] + [0.0] * 10), FS), (np.ones((2, 3)), FS),
                    (np.ones(N), 0.0), (np.ones((2, 2, 8)), FS)):
        try:
            extract_features(bad, fs)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


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