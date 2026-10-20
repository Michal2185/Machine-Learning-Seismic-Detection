"""DNN model tests that need torch. Skipped (exit 0) without torch. Run: python tests/test_models_torch.py"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch  # noqa: F401
except ImportError:
    print("SKIPPED: torch is not installed")
    sys.exit(0)

import numpy as np

from synthetic_windows import FS, make_dataset

from src.modules.models import build_dnn_detector, make_mlp, mlp_param_count
from src.modules.models.dnn import PRESETS
from src.modules.normalization import NormConfig, Normalizer, ScaleCache
from src.modules.training import TrainConfig, average_precision, linear_macs


def test_make_mlp_matches_formula():
    for key in PRESETS:
        n = sum(p.numel() for p in make_mlp(PRESETS[key]).parameters())
        assert n == mlp_param_count(PRESETS[key])
    assert make_mlp((8,), dropout=0.2)[2].__class__.__name__ == "Dropout"


def test_dnn_detector_end_to_end_and_resources():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    norm = Normalizer(NormConfig("N2"), ScaleCache.from_streams(streams, NormConfig("N2")))
    det = build_dnn_detector("dnn", norm, FS, size="1k", config=TrainConfig(epochs=25, patience=8, seed=0))
    train = [w for w in wins if w.event.evid in ("w00", "w01", "w02", "w03", "w06")]
    test = [w for w in wins if w.event.evid in ("w04", "w05", "w07")]
    det.fit(train)
    s = det.score(test, {})
    lab = np.array([w.label for w in test])
    keep = lab != -1
    assert average_precision(lab[keep] == 1, s[keep]) > 0.5
    r = det.resource_report(test[0])
    h1, h2 = PRESETS["1k"]
    assert r.params == mlp_param_count(PRESETS["1k"])
    assert r.macs_per_window == linear_macs(6, h1) + linear_macs(h1, h2) + linear_macs(h2, 1)
    assert r.notes == [] and len(det.fit_infos) == 1


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