"""
python -m tests.test_envelope_input

Synthetic tests of the variant-E input builder (no torch, no data files). A fake identity normaliser
stands in for N0 so the features can be checked against the thesis definitions in extract_features.
"""

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.dnn_features import extract_features
from src.modules.training.inputs import EnvelopeInput, envelope_features

FS = 6.625
T = 398
N_SUB = 12
L = T // N_SUB            # 33 samples (4.98 s)
USED = N_SUB * L          # 396: the oldest 2 samples are dropped


class IdNorm:
    """Identity 'normaliser' with the Normalizer interface used by the input builders."""

    def fit(self, windows):
        return self

    def windows(self, windows):
        return np.stack([np.asarray(w.data, dtype=np.float32) for w in windows])


def make_windows(n, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return [SimpleNamespace(data=rng.normal(0, scale, T).astype(np.float32), label=0,
                            event=SimpleNamespace(evid="w")) for _ in range(n)]


def X(windows):
    return np.stack([w.data for w in windows]).astype(np.float64)


def test_shape_and_dtype():
    f = envelope_features(X(make_windows(5)), N_SUB, FS)
    assert f.shape == (5, N_SUB, 4) and f.dtype == np.float32 and np.isfinite(f).all()


def test_matches_thesis_feature_definitions():
    x = X(make_windows(4, seed=1))
    f = envelope_features(x, N_SUB, FS)
    for i in range(4):
        for j in range(N_SUB):
            sub = x[i, T - USED + j * L: T - USED + (j + 1) * L]
            ref = extract_features(sub, FS)                    # energy, rms, ptp, crest, dom, zcr
            assert np.isclose(f[i, j, 0], np.log(ref[0] + 1e-6), rtol=1e-4, atol=1e-5), (i, j)
            assert np.isclose(f[i, j, 1], ref[1], rtol=1e-4, atol=1e-6)
            assert np.isclose(f[i, j, 2], ref[2], rtol=1e-4, atol=1e-6)
            assert np.isclose(f[i, j, 3], ref[5], rtol=1e-4, atol=1e-6)


def test_alignment_to_window_end():
    w = make_windows(1, seed=2)
    x = X(w)
    base = envelope_features(x, N_SUB, FS)
    y = x.copy(); y[0, :T - USED] += 100.0                     # oldest 2 samples: ignored
    assert np.allclose(envelope_features(y, N_SUB, FS), base)
    y = x.copy(); y[0, -1] += 5.0                              # newest sample: changes only the last step
    g = envelope_features(y, N_SUB, FS)
    assert not np.allclose(g[0, -1], base[0, -1])
    assert np.allclose(g[0, :-1], base[0, :-1])


def test_burst_in_last_subwindow():
    x = X(make_windows(1, seed=3, scale=0.1))
    base = envelope_features(x, N_SUB, FS)
    y = x.copy(); y[0, -L:] *= 10.0                            # amplitude x10 in the newest step
    g = envelope_features(y, N_SUB, FS)
    assert np.isclose(g[0, -1, 0] - base[0, -1, 0], 2 * np.log(10.0), atol=1e-3)   # log energy + 4.6
    assert np.isclose(g[0, -1, 1] / base[0, -1, 1], 10.0, rtol=1e-4)               # rms x10
    assert np.allclose(g[0, :-1], base[0, :-1])


def test_flat_window_is_finite():
    x = np.zeros((2, T))
    f = envelope_features(x, N_SUB, FS)
    assert np.isfinite(f).all()
    assert np.allclose(f[..., 0], np.log(1e-6), atol=1e-5) and np.allclose(f[..., 1:], 0.0)


def test_log_amp_option():
    x = X(make_windows(3, seed=4))
    a = envelope_features(x, N_SUB, FS)
    b = envelope_features(x, N_SUB, FS, log_amp=True)
    assert np.allclose(b[..., 1], np.log(a[..., 1] + 1e-6), atol=1e-4)
    assert np.allclose(b[..., 2], np.log(a[..., 2] + 1e-6), atol=1e-4)
    assert np.allclose(b[..., 0], a[..., 0]) and np.allclose(b[..., 3], a[..., 3])


def test_standardisation_fitted_on_training_windows():
    train = make_windows(300, seed=5)
    inp = EnvelopeInput(IdNorm(), FS, fit_stride=1).fit(train)
    z = inp.build(train).reshape(-1, 4)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-4) and np.allclose(z.std(axis=0), 1.0, atol=1e-3)
    assert inp.build(make_windows(7, seed=6)).shape == (7, N_SUB, 4)


def test_batch_equals_single():
    train = make_windows(40, seed=7)
    inp = EnvelopeInput(IdNorm(), FS).fit(train)
    batch = inp.build(train[:6])
    for i in range(6):
        assert np.allclose(inp.build([train[i]])[0], batch[i], atol=1e-6)


def test_does_not_modify_windows():
    w = make_windows(3, seed=8)
    before = X(w).copy()
    EnvelopeInput(IdNorm(), FS).fit(w).build(w)
    assert np.array_equal(X(w), before)


def test_unscaled_option_and_not_fitted_error():
    w = make_windows(3, seed=9)
    inp = EnvelopeInput(IdNorm(), FS, scale=False)
    assert np.allclose(inp.build(w), envelope_features(X(w), N_SUB, FS))
    try:
        EnvelopeInput(IdNorm(), FS).build(w)
    except RuntimeError:
        return
    raise AssertionError("build before fit must raise")


def test_step_and_ops():
    inp = EnvelopeInput(IdNorm(), FS).fit(make_windows(20, seed=10))
    assert np.isclose(inp.step_s, L / FS) and 4.9 < inp.step_s < 5.1
    assert 0 < inp.ops_streaming_per_10s() < inp.ops_per_window(T)


def test_rejects_bad_input():
    for bad in (np.zeros(T), np.full((1, T), np.nan), np.zeros((1, 10))):
        try:
            envelope_features(bad, N_SUB, FS)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    try:
        envelope_features(np.zeros((1, T)), N_SUB, 0.0)
    except ValueError:
        return
    raise AssertionError("fs <= 0 must raise")


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