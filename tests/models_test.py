"""Tests of the DNN / RFC model code that need no torch. Run: python tests/test_models.py"""

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_windows import FS, WIN, make_dataset

from src.modules.dnn_features import FEATURE_NAMES
from src.modules.models import RandomForestDetector, build_rfc_detector, mlp_param_count
from src.modules.models.dnn import PRESETS
from src.modules.normalization import NormConfig, Normalizer, ScaleCache
from src.modules.training import average_precision


def n2_norm(streams):
    return Normalizer(NormConfig("N2"), ScaleCache.from_streams(streams, NormConfig("N2")))


def split(wins):
    train = [w for w in wins if w.event.evid in ("w00", "w01", "w02", "w03", "w06")]
    test = [w for w in wins if w.event.evid in ("w04", "w05", "w07")]
    return train, test


def test_mlp_param_count_and_presets():
    assert mlp_param_count((8,)) == 6 * 8 + 8 + 8 + 1
    assert mlp_param_count((4, 4), n_in=2, n_out=1) == (2 * 4 + 4) + (4 * 4 + 4) + (4 + 1)
    for key, target in (("1k", 1_000), ("10k", 10_000), ("100k", 100_000)):
        n = mlp_param_count(PRESETS[key])
        assert abs(n - target) / target < 0.1, (key, n)
    assert mlp_param_count(PRESETS["100k"]) * 4 < 500_000          # fp32 bytes, far below the 105 MB DNN


def test_rfc_learns_on_unseen_waveforms():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    train, test = split(wins)
    det = build_rfc_detector("rfc", n2_norm(streams), FS, size="small", seed=0).fit(train)
    s = det.score(test, {})
    lab = np.array([w.label for w in test])
    keep = lab != -1
    assert s.shape == (len(test),) and np.isfinite(s).all()
    assert average_precision(lab[keep] == 1, s[keep]) > 0.5
    assert det.score([], {}).shape == (0,)


def test_rfc_importances_amplitude_features_dominate():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    train, _ = split(wins)
    det = build_rfc_detector("rfc", n2_norm(streams), FS, seed=0).fit(train)
    imp = det.importances()
    assert set(imp) == set(FEATURE_NAMES) and abs(sum(imp.values()) - 1.0) < 1e-6
    amp = imp["energy"] + imp["rms"] + imp["ptp"]
    assert amp > 0.6, imp                                          # synthetic events are amplitude bursts


def test_rfc_determinism_and_refit_resets():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    f0 = [w for w in wins if w.event.evid in ("w00", "w01", "w02", "w06")]
    f1 = [w for w in wins if w.event.evid in ("w03", "w04", "w05", "w07")]
    norm = n2_norm(streams)
    a = build_rfc_detector("a", norm, FS, seed=1).fit(f0).score(wins[:300], {})
    b = build_rfc_detector("b", norm, FS, seed=1).fit(f0).score(wins[:300], {})
    c = build_rfc_detector("c", norm, FS, seed=2).fit(f0).score(wins[:300], {})
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    det = build_rfc_detector("d", norm, FS, seed=1).fit(f0)
    det.fit(f1)
    fresh = build_rfc_detector("e", norm, FS, seed=1).fit(f1)
    assert np.array_equal(det.score(wins[:300], {}), fresh.score(wins[:300], {}))


def test_rfc_never_trains_on_ignore_windows():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    train, _ = split(wins)
    seen = []

    class SpyBuilder:
        def __init__(self, inner): self.inner = inner
        def fit(self, w): self.inner.fit(w); return self
        def build(self, w):
            seen.extend(x.label for x in w)
            return self.inner.build(w)
        def ops_per_window(self, n): return self.inner.ops_per_window(n)

    det = build_rfc_detector("rfc", n2_norm(streams), FS, size="small", seed=0)
    det.inputs = SpyBuilder(det.inputs)
    det.fit(train)
    assert set(seen) == {0, 1} and seen.count(1) > 0


def test_rfc_resource_report_is_small_and_flagged():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    train, _ = split(wins)
    det = build_rfc_detector("rfc", n2_norm(streams), FS, size="medium", seed=0).fit(train)
    r = det.resource_report(wins[0])
    assert 0 < r.params <= 100 * (2 ** 9 - 1)                      # depth 8 -> at most 511 nodes per tree
    assert r.bytes_fp32 < 400_000 and r.bytes_int8 < r.bytes_fp32
    assert 0 < r.macs_per_window <= 100 * 8 and r.state_floats == 0
    assert r.notes and "estimates" in r.notes[0]


def test_rfc_errors():
    streams, wins = make_dataset()
    det = build_rfc_detector("rfc", n2_norm(streams), FS)
    for fn in (lambda: det.score(wins[:3], {}), det.importances, lambda: det.resource_report(wins[0])):
        try:
            fn()
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass


def test_summarise_and_table():
    from src.modules.models.report import format_table, summarise
    from src.modules.training import ResourceReport

    recs = []
    for seed, (rc, fa) in enumerate([(0.6, 0.2), (0.7, 0.3), (0.8, 0.4)]):
        recs.append(dict(config="dnn-10k-N2", budget=0.25, seed=seed, recall=rc, fa_per_day=fa, test_ap=0.3))
        recs.append(dict(config="dnn-10k-N2", budget=2.0, seed=seed, recall=rc + 0.1, fa_per_day=fa, test_ap=0.3))
    recs.append(dict(config="rfc-N0", budget=0.25, seed=0, recall=0.5, fa_per_day=0.5, test_ap=float("nan")))
    rows = summarise(recs)
    a = [r for r in rows if r["config"] == "dnn-10k-N2" and r["budget"] == 0.25][0]
    assert a["n_seeds"] == 3 and abs(a["recall_mean"] - 0.7) < 1e-12 and abs(a["recall_sd"] - 0.1) < 1e-12
    b = [r for r in rows if r["config"] == "rfc-N0"][0]
    assert b["n_seeds"] == 1 and b["recall_sd"] == 0.0
    res = {"dnn-10k-N2": ResourceReport(10081, 40324, 10113, 10000, 0)}
    text = format_table(rows, res)
    assert "STA/LTA: recall 0.67" in text and "10,081" in text and "rfc-N0" in text


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