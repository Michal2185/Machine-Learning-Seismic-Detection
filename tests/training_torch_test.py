"""
Torch learner + resource-hook tests. Skipped (exit 0) when torch is not installed.
Run: python tests/test_training_torch.py
NOTE: written without a torch runtime at hand; the first run on your machine is its first execution.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("SKIPPED: torch is not installed")
    sys.exit(0)

import numpy as np

from synthetic_windows import FS, WIN, make_dataset

from src.modules.normalization import NormConfig, Normalizer, ScaleCache
from src.modules.training import (
    FeatureInput,
    LearnerDetector,
    TorchLearner,
    TrainConfig,
    average_precision,
    conv1d_macs,
    count_model_resources,
    gru_macs,
    linear_macs,
)


def mlp():
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1))


class ConvGru(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 5)
        self.gru = nn.GRU(4, 8, batch_first=True)
        self.head = nn.Linear(8, 1)

    def forward(self, x):                       # x: (B, 1, T)
        h = torch.relu(self.conv(x)).transpose(1, 2)
        _, last = self.gru(h)
        return self.head(last[-1])


def separable(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.float32)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    x[:, 0] += 3.0 * y
    return x, y


def test_learner_learns_separable_data():
    x, y = separable()
    lr = TorchLearner(mlp, lr=1e-2)
    lr.reset(0)
    for _ in range(60):
        for i in range(0, len(y), 64):
            lr.train_step(x[i:i + 64], y[i:i + 64], 1.0)
    s = lr.predict(x)
    assert s.shape == (len(y),) and s.dtype == np.float64
    assert average_precision(y, s) > 0.9


def test_reset_is_deterministic_and_state_roundtrip():
    x, y = separable()
    a, b = TorchLearner(mlp), TorchLearner(mlp)
    a.reset(3), b.reset(3)
    assert np.allclose(a.predict(x), b.predict(x))
    a.reset(4)
    assert not np.allclose(a.predict(x), b.predict(x))
    state = a.get_state()
    before = a.predict(x)
    for _ in range(20):
        a.train_step(x[:64], y[:64], 1.0)
    assert not np.allclose(a.predict(x), before)
    a.set_state(state)
    assert np.allclose(a.predict(x), before)


def test_resources_mlp():
    lr = TorchLearner(mlp)
    lr.reset(0)
    r = lr.resources(np.zeros((1, 6), dtype=np.float32))
    assert r.params == 6 * 8 + 8 + 8 + 1
    assert r.macs_per_window == linear_macs(6, 8) + linear_macs(8, 1)
    assert r.bytes_fp32 == 4 * r.params and r.bytes_int8 == r.params + 4 * 4
    assert r.state_floats == 0 and r.notes == []


def test_resources_conv_gru():
    lr = TorchLearner(ConvGru)
    lr.reset(0)
    r = lr.resources(np.zeros((1, 1, WIN), dtype=np.float32))
    expect = conv1d_macs(1, 4, 5, WIN - 4) + gru_macs(4, 8, WIN - 4) + linear_macs(8, 1)
    assert r.macs_per_window == expect
    assert r.state_floats == 8


def test_resources_flags_unsupported_layers():
    lr = TorchLearner(lambda: nn.Sequential(nn.Linear(6, 8), nn.LayerNorm(8), nn.Linear(8, 1)))
    lr.reset(0)
    r = lr.resources(np.zeros((1, 6), dtype=np.float32))
    assert any("LayerNorm" in n for n in r.notes)
    assert r.macs_per_window == linear_macs(6, 8) + linear_macs(8, 1)


def test_detector_with_torch_learner():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    cache = ScaleCache.from_streams(streams, NormConfig("N2"))
    inputs = FeatureInput(Normalizer(NormConfig("N2"), cache), FS)
    det = LearnerDetector("mlp", TorchLearner(mlp, lr=1e-2), inputs,
                          TrainConfig(epochs=20, patience=6, seed=0))
    train = [w for w in wins if w.event.evid in ("w00", "w01", "w02", "w03", "w06")]
    test = [w for w in wins if w.event.evid in ("w04", "w05", "w07")]
    det.fit(train)
    s = det.score(test, {})
    lab = np.array([w.label for w in test])
    keep = lab != -1
    assert average_precision(lab[keep] == 1, s[keep]) > 0.5
    rep = det.resource_report(test[0])
    assert rep.params == 65 and rep.macs_per_window == 56


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