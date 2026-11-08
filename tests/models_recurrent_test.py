"""
python -m tests.test_models_recurrent_torch

Needs torch and ncps. Checks parameter counts against the closed forms, output shape, determinism,
the effect of the CfC time step, MAC accounting against the closed form, and that both models learn
a synthetic "burst at the end of the window" task.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.modules.models.recurrent import (
    PRESETS, cfc_macs, cfc_param_count, gru_macs, gru_param_count, make_cfc, make_gru,
)
from src.modules.training.learners import TorchLearner
from src.modules.training.resources import count_model_resources


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_param_counts_match_formulas():
    for F, table in PRESETS.items():
        for size, hs in table.items():
            if size == "100k" and F == 4:
                pass
            assert n_params(make_gru(F, hs["gru"])) == gru_param_count(F, hs["gru"]), (F, size, "gru")
            assert n_params(make_cfc(F, hs["cfc"])) == cfc_param_count(F, hs["cfc"]), (F, size, "cfc")


def test_budgets_are_matched():
    for F, table in PRESETS.items():
        for size, hs in table.items():
            g, c = gru_param_count(F, hs["gru"]), cfc_param_count(F, hs["cfc"])
            assert abs(g - c) / g < 0.02, (F, size, g, c)


def test_known_values():
    assert gru_param_count(1, 17) == 1038 and gru_param_count(1, 56) == 9969
    assert cfc_param_count(1, 15) == 1036 and cfc_param_count(1, 49) == 10046


def test_output_shape():
    x = torch.randn(3, 50, 1)
    assert make_gru(1, 17)(x).shape == (3, 1)
    assert make_cfc(1, 15)(x).shape == (3, 1)
    assert make_cfc(1, 15, dt=0.151)(x).shape == (3, 1)
    assert make_cfc(1, 15, dt=0.151)(x[:1]).shape == (1, 1)       # batch of one (squeeze in ncps)


def test_determinism_under_seed():
    x = torch.randn(4, 30, 1)
    for f in (lambda: make_gru(1, 17), lambda: make_cfc(1, 15)):
        torch.manual_seed(7); a = f()(x)
        torch.manual_seed(7); b = f()(x)
        assert torch.allclose(a, b)


def test_cfc_timestep_changes_output():
    x = torch.randn(4, 30, 1)
    torch.manual_seed(1); a = make_cfc(1, 15, dt=None)(x)
    torch.manual_seed(1); b = make_cfc(1, 15, dt=0.151)(x)
    assert not torch.allclose(a, b, atol=1e-4)


def test_macs_match_closed_form_and_no_notes():
    x = np.zeros((1, 398, 1), dtype=np.float32)
    r = count_model_resources(make_gru(1, 17), x)
    assert r.macs_per_window == gru_macs(1, 17, 398) and not r.notes and r.state_floats == 17
    r = count_model_resources(make_cfc(1, 15), x)
    assert r.macs_per_window == cfc_macs(1, 15, 398), (r.macs_per_window, cfc_macs(1, 15, 398))
    assert not r.notes, r.notes
    assert r.state_floats == 15, "apply the state_floats patch to resources.py"


def test_envelope_variant_shapes_and_macs():
    x = torch.randn(3, 12, 4)
    for size, hs in PRESETS[4].items():
        assert make_gru(4, hs["gru"])(x).shape == (3, 1)
        assert make_cfc(4, hs["cfc"], dt=4.98)(x).shape == (3, 1)
    xs = np.zeros((1, 12, 4), dtype=np.float32)
    r = count_model_resources(make_gru(4, 15), xs)
    assert r.macs_per_window == gru_macs(4, 15, 12) and not r.notes
    r = count_model_resources(make_cfc(4, 13), xs)
    assert r.macs_per_window == cfc_macs(4, 13, 12) and not r.notes and r.state_floats == 13


def _burst_data(n, T, rng):
    y = (np.arange(n) % 2).astype(np.float32)
    x = rng.normal(0, 0.3, (n, T, 1)).astype(np.float32)
    x[y == 1, -T // 3:, 0] *= 8.0
    return x, y


def _learns(factory):
    rng = np.random.default_rng(0)
    learner = TorchLearner(factory, lr=5e-3, weight_decay=0.0)
    learner.reset(0)
    for _ in range(120):
        x, y = _burst_data(64, 60, rng)
        learner.train_step(x, y, 1.0)
    x, y = _burst_data(256, 60, np.random.default_rng(1))
    acc = float(((learner.predict(x) > 0) == (y > 0.5)).mean())
    assert acc > 0.9, acc


def test_gru_learns_burst():
    _learns(lambda: make_gru(1, 17))


def test_cfc_learns_burst():
    _learns(lambda: make_cfc(1, 15, dt=0.151))


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