"""
python -m tests.test_checkpoints

Round trips of fitted detectors through checkpoints (src/modules/checkpoints), with fake data and the numpy test
learner; the torch round trip runs only when torch is installed. Also checks the fold callback of
cross_validate_budgets and the load / save hooks of the stress loop.
"""

import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.checkpoints import (
    FORMAT, checkpoint_path, detector_state, load_checkpoint, load_detector_state, save_checkpoint, verify_roundtrip,
)
from src.modules.training.inputs import EnvelopeInput
from src.modules.training.learners import NumpyLogisticLearner

FS = 6.625
T = 398


# ---------------------------------------------------------------- fakes

class FakeNormalizer:
    """The attributes of Normalizer that the checkpoint code touches (identity scaling by `fixed_scale`)."""

    def __init__(self, variant="N0"):
        self.config = SimpleNamespace(variant=variant)
        self.fixed_scale = None

    def fit(self, windows):
        self.fixed_scale = 2.0
        return self

    def windows(self, windows):
        return np.stack([np.asarray(w.data, dtype=np.float32) / (self.fixed_scale or 1.0) for w in windows])


class FakeLearnerDetector:
    """Same shape as LearnerDetector: learner + input builder + config + _fitted + score()."""

    def __init__(self, name="fake", variant="N0", seed=3):
        self.name = name
        self.learner = NumpyLogisticLearner(lr=0.05)
        self.inputs = EnvelopeInput(FakeNormalizer(variant), FS)
        self.config = SimpleNamespace(seed=seed)
        self._fitted = False

    def fit(self, windows):
        self.learner.reset(self.config.seed)
        self.inputs.fit(windows)
        x = self.inputs.build(windows)
        y = np.array([w.label for w in windows], dtype=np.float32)
        for _ in range(30):
            self.learner.train_step(x, y, 3.0)
        self._fitted = True
        return self

    def score(self, windows, streams_by_id=None):
        return self.learner.predict(self.inputs.build(list(windows)))


def make_windows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lab = i % 2
        d = rng.normal(0, 1.0 + 4.0 * lab, T).astype(np.float32)
        out.append(SimpleNamespace(data=d, label=lab, event=SimpleNamespace(evid="w"), end_time=i))
    return out


def fitted(seed=3):
    d = FakeLearnerDetector(seed=seed)
    d.fit(make_windows(80, seed=seed))
    return d


# ---------------------------------------------------------------- state round trips

def test_learner_detector_roundtrip_reproduces_scores():
    live = fitted()
    state = detector_state(live, {0.25: 1.5, 2.0: 0.7}, {"note": "x"})
    fresh = FakeLearnerDetector()
    thr, meta = load_detector_state(fresh, pickle.loads(pickle.dumps(state)))
    w = make_windows(30, seed=9)
    assert np.allclose(live.score(w), fresh.score(w))
    assert thr == {0.25: 1.5, 2.0: 0.7} and meta["note"] == "x" and fresh._fitted


def test_input_state_is_restored_exactly():
    live = fitted()
    fresh = FakeLearnerDetector()
    load_detector_state(fresh, detector_state(live))
    assert fresh.inputs.normalizer.fixed_scale == 2.0
    assert np.array_equal(fresh.inputs.mean_, live.inputs.mean_) and np.array_equal(fresh.inputs.std_, live.inputs.std_)
    assert fresh.inputs.step_s == live.inputs.step_s
    w = make_windows(5, seed=2)
    assert np.array_equal(fresh.inputs.build(w), live.inputs.build(w))


def test_unfitted_detector_is_refused():
    try:
        detector_state(FakeLearnerDetector())
    except RuntimeError:
        return
    raise AssertionError("an unfitted detector must not be saved")


def test_n2_n3_are_not_supported():
    d = fitted()
    d.inputs.normalizer.config.variant = "N2"
    try:
        detector_state(d)
    except NotImplementedError:
        return
    raise AssertionError("N2 must be refused")


def test_mismatches_are_detected():
    state = detector_state(fitted())
    other = FakeLearnerDetector()
    other.inputs.normalizer.config.variant = "N1"
    for bad in (other,):
        try:
            load_detector_state(bad, state)
        except ValueError:
            continue
        raise AssertionError("variant mismatch must raise")
    d = FakeLearnerDetector()
    d.inputs.n_sub = 6
    try:
        load_detector_state(d, state)
    except ValueError:
        pass
    else:
        raise AssertionError("n_sub mismatch must raise")
    d = FakeLearnerDetector()
    d.inputs = SimpleNamespace(normalizer=FakeNormalizer(), layout="NT")
    try:
        load_detector_state(d, state)
    except ValueError:
        pass
    else:
        raise AssertionError("input class mismatch must raise")
    state["format"] = FORMAT + 1
    try:
        load_detector_state(FakeLearnerDetector(), state)
    except ValueError:
        return
    raise AssertionError("unknown format must raise")


class ThresholdDetector:
    """Not a learner detector: exercises the pickle fallback (like the amplitude control / RFC)."""

    name = "thr"

    def __init__(self):
        self.cut = None

    def fit(self, windows):
        self.cut = float(np.median([np.abs(w.data).mean() for w in windows]))
        return self

    def score(self, windows, streams_by_id=None):
        return np.array([np.abs(w.data).mean() - self.cut for w in windows])


def test_pickle_fallback_roundtrip():
    live = ThresholdDetector().fit(make_windows(20))
    fresh = ThresholdDetector()
    load_detector_state(fresh, pickle.loads(pickle.dumps(detector_state(live, {1.0: 0.2}))))
    w = make_windows(8, seed=4)
    assert np.allclose(live.score(w), fresh.score(w)) and fresh.cut == live.cut


def test_unpicklable_detector_gives_a_clear_error():
    d = ThresholdDetector().fit(make_windows(10))
    d.hook = lambda x: x
    try:
        detector_state(d)
    except TypeError as e:
        assert "cannot checkpoint" in str(e)
        return
    raise AssertionError("expected TypeError")


# ---------------------------------------------------------------- files

def test_files_layout_atomic_write_and_missing():
    with tempfile.TemporaryDirectory() as root:
        live = fitted()
        p = save_checkpoint(root, "cfcE-1k-N0-long", 2, 4, live, {0.25: 1.0})
        assert p == checkpoint_path(root, "cfcE-1k-N0-long", 2, 4)
        assert p.parts[-3:] == ("cfcE-1k-N0-long", "s2", "fold4.pkl") and p.exists() and not list(p.parent.glob("*.tmp"))
        thr, meta = load_checkpoint(root, "cfcE-1k-N0-long", 2, 4, FakeLearnerDetector())
        assert thr == {0.25: 1.0} and meta["config"] == "cfcE-1k-N0-long" and meta["seed"] == 2 and meta["fold"] == 4
        try:
            load_checkpoint(root, "cfcE-1k-N0-long", 2, 0, FakeLearnerDetector())
        except FileNotFoundError:
            return
        raise AssertionError("missing checkpoint must raise")


def test_verify_roundtrip_passes_and_catches_a_changed_detector():
    with tempfile.TemporaryDirectory() as root:
        live = fitted()
        save_checkpoint(root, "m", 0, 0, live, {1.0: 0.0})
        w = make_windows(20, seed=5)
        assert verify_roundtrip(live, FakeLearnerDetector(), root, "m", 0, 0, w) == 0.0
        live.learner.w = live.learner.w + 0.5            # live detector drifts from what was saved
        try:
            verify_roundtrip(live, FakeLearnerDetector(), root, "m", 0, 0, w)
        except AssertionError:
            return
        raise AssertionError("a differing detector must be caught")


def test_torch_learner_roundtrip_if_available():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("     (torch not installed: skipped)")
        return

    from src.modules.training.learners import TorchLearner

    def make():
        return TorchLearner(lambda: nn.Sequential(nn.Flatten(), nn.Linear(48, 1)), lr=1e-2)

    live = FakeLearnerDetector()
    live.learner = make()
    live.fit(make_windows(80, seed=6))
    fresh = FakeLearnerDetector()
    fresh.learner = make()
    load_detector_state(fresh, pickle.loads(pickle.dumps(detector_state(live, {1.0: 0.1}))))
    w = make_windows(16, seed=7)
    assert np.allclose(live.score(w), fresh.score(w), atol=1e-6)


# ---------------------------------------------------------------- hooks in cv.py and the stress loop

def test_cv_fold_callback_gets_the_fold_state():
    from src.modules.evaluation import cv as cvmod

    seen = []
    calls = dict(fit=0)

    class Det:
        name = "d"

        def fit(self, windows):
            calls["fit"] += 1
            self.fold_marker = calls["fit"]
            return self

        def score(self, windows, streams):
            return np.zeros(len(windows))

    class Ev:
        def curve(self, streams, series):
            return None

        def select_threshold(self, curve, max_fa_per_day, group="all"):
            return 10.0 * max_fa_per_day

        def evaluate(self, streams, series, thr):
            return SimpleNamespace(recall=lambda g="all": 1.0, fa_per_day=0.0)

    win = SimpleNamespace(label=0)
    bundle = SimpleNamespace(
        folds=SimpleNamespace(n_folds=3),
        fold_split=lambda k: SimpleNamespace(train=[win], validation=[win], test=[win],
                                             validation_events=[], test_events=[]),
        eval_streams=lambda ids: [],
    )

    old = (cvmod.build_series, cvmod.average_precision, cvmod.pool_results)
    cvmod.build_series = lambda w, s: s
    cvmod.average_precision = lambda y, s: 0.5
    cvmod.pool_results = lambda rs: rs

    try:
        det = Det()
        cvmod.cross_validate_budgets(bundle, [det], Ev(), [0.25, 2.0], verbose=False,
                                     fold_callback=lambda k, chosen: seen.append(
                                         (k, det.fold_marker, {b: t for b, (c, t) in chosen.items()},
                                          {id(c) for c, _ in chosen.values()} == {id(det)})))
    finally:
        cvmod.build_series, cvmod.average_precision, cvmod.pool_results = old

    assert [s[0] for s in seen] == [0, 1, 2]
    assert [s[1] for s in seen] == [1, 2, 3]                       # called while that fold's fit is in place
    assert all(s[2] == {0.25: 2.5, 2.0: 20.0} and s[3] for s in seen)


def test_stress_loop_load_and_save_hooks():
    from src.modules.stress.module import make_perturbation, stress_cross_validate
    from tests.stress_test import FakeBundle, FakeEval, RmsDetector

    perts = [make_perturbation("clean")]
    kw = dict(build_series=lambda w, s: np.asarray(s), pool_results=lambda rs: rs, verbose=False)

    saved = []
    det = RmsDetector()
    run = stress_cross_validate(FakeBundle(), det, FakeEval(), [1.0, 5.0], perts,
                                save_fold=lambda k, d, thr: saved.append((k, dict(thr))), **kw)
    assert det.fits == 2 and [s[0] for s in saved] == [0, 1]
    assert saved[0][1] == run.thresholds[0]

    loaded = []
    det2 = RmsDetector()

    def load_fold(k, d):
        loaded.append(k)
        return dict(saved[k][1])

    run2 = stress_cross_validate(FakeBundle(), det2, FakeEval(), [1.0, 5.0], perts, load_fold=load_fold,
                                 save_fold=lambda *a: (_ for _ in ()).throw(AssertionError("no save after a load")),
                                 **kw)
    assert det2.fits == 0 and loaded == [0, 1]                       # nothing trained
    assert run2.thresholds == run.thresholds
    assert [r.n_alarm for r in run2.pooled[("clean", 5.0)]] == [r.n_alarm for r in run.pooled[("clean", 5.0)]]

    # a budget missing from the checkpoint is chosen on the clean validation streams
    det3 = RmsDetector()
    run3 = stress_cross_validate(FakeBundle(), det3, FakeEval(), [1.0, 5.0], perts,
                                 load_fold=lambda k, d: {1.0: saved[k][1][1.0]}, **kw)
    assert det3.fits == 0 and run3.thresholds == run.thresholds


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