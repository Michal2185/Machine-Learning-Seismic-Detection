"""Unit tests for the training harness (no torch needed). Run: python tests/test_training.py"""

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_windows import FS, WIN, Ev, Win, make_dataset, make_stream, make_windows

from src.modules.dnn_features import FEATURE_NAMES, FeatureScaler, extract_features
from src.modules.normalization import NormConfig, Normalizer, ScaleCache
from src.modules.training import (
    FeatureInput,
    LearnerDetector,
    NumpyLogisticLearner,
    RawWindowInput,
    TrainConfig,
    average_precision,
    conv1d_macs,
    feature_ops,
    gru_macs,
    linear_macs,
    lstm_macs,
    model_bytes,
    rnn_state_floats,
    split_by_waveform,
)


def expect_error(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def n2_inputs(streams, layout=None):
    cache = ScaleCache.from_streams(streams, NormConfig("N2"))
    norm = Normalizer(NormConfig("N2"), cache)
    return FeatureInput(norm, FS) if layout is None else RawWindowInput(norm, layout)


# ---------------------------------------------------------------- metrics

def test_average_precision_known_values():
    assert average_precision([1, 0, 0, 0], [0.9, 0.5, 0.4, 0.1]) == 1.0
    # positives ranked 1st and 3rd: AP = (1/1 + 2/3) / 2
    assert abs(average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) - (1 + 2 / 3) / 2) < 1e-12
    assert np.isnan(average_precision([0, 0, 0], [1, 2, 3]))


def test_average_precision_ties_and_sklearn():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.9])
    try:
        from sklearn.metrics import average_precision_score
        assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-12
        rng = np.random.default_rng(0)
        y2 = rng.random(500) < 0.1
        s2 = rng.normal(size=500) + 2 * y2
        assert abs(average_precision(y2, s2) - average_precision_score(y2, s2)) < 1e-12
    except ImportError:
        pass
    # random scores -> AP near prevalence
    rng = np.random.default_rng(1)
    y3 = rng.random(20000) < 0.05
    assert abs(average_precision(y3, rng.random(20000)) - 0.05) < 0.02


# ---------------------------------------------------------------- split

def test_split_disjoint_complete_deterministic():
    _, wins = make_dataset()
    a = split_by_waveform(wins, 0.25, seed=3)
    b = split_by_waveform(wins, 0.25, seed=3)
    tr, va, ids = a
    assert len(tr) + len(va) == len(wins)
    assert {w.event.evid for w in tr}.isdisjoint({w.event.evid for w in va})
    assert ids == b[2] and ids == sorted({w.event.evid for w in va})
    assert split_by_waveform(wins, 0.25, seed=4)[2] != ids or True   # may coincide; no crash
    # whole waveforms only
    for evid in ids:
        assert sum(w.event.evid == evid for w in tr) == 0


def test_split_keeps_positives_on_both_sides_and_in_train():
    _, wins = make_dataset(n_event=5, n_quiet=3)
    for seed in range(10):
        tr, va, ids = split_by_waveform(wins, 0.25, seed)
        assert any(w.label == 1 for w in tr)
        assert len(ids) == 2
        assert any(w.label == 1 for w in va), seed        # 2 of 8 held out, one with events


def test_split_edge_cases():
    _, wins = make_dataset()
    tr, va, ids = split_by_waveform(wins, 0.0, 0)
    assert len(tr) == len(wins) and va == [] and ids == []
    one = [w for w in wins if w.event.evid == "w00"]
    tr, va, ids = split_by_waveform(one, 0.5, 0)
    assert len(tr) == len(one) and va == []
    # only one waveform has positives: it stays in training, validation has none
    pos_one = [w for w in wins if w.event.evid in ("w00", "w05", "w06", "w07")]
    tr, va, ids = split_by_waveform(pos_one, 0.5, 0)
    assert any(w.label == 1 for w in tr) and not any(w.label == 1 for w in va)


# ---------------------------------------------------------------- feature scaler

def test_feature_scaler_standardises_log_features():
    rng = np.random.default_rng(0)
    f = np.abs(rng.normal(3.0, 0.7, size=(500, 6))).astype(np.float32)
    sc = FeatureScaler().fit(f)
    z = sc.transform(f)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-4)
    idx = FEATURE_NAMES.index("energy")
    assert abs(np.corrcoef(z[:, idx], np.log(f[:, idx]))[0, 1] - 1.0) < 1e-6   # log is monotone-linear
    zi = FEATURE_NAMES.index("zcr")
    assert abs(np.corrcoef(z[:, zi], f[:, zi])[0, 1] - 1.0) < 1e-6            # zcr is not logged


def test_feature_scaler_flat_windows_clip_and_errors():
    f = np.abs(np.random.default_rng(1).normal(1, 0.3, size=(200, 6))) + 0.1
    sc = FeatureScaler(clip=5.0).fit(f)
    flat = np.zeros((1, 6))
    z = sc.transform(flat)
    assert np.isfinite(z).all() and np.abs(z).max() <= 5.0
    expect_error(lambda: FeatureScaler().transform(f), RuntimeError)
    expect_error(lambda: FeatureScaler().fit(f[:1]))
    const = np.ones((10, 6))
    assert np.isfinite(FeatureScaler().fit(const).transform(const)).all()      # std 0 guarded


def test_feature_scaler_is_fixed_after_fit():
    f = np.abs(np.random.default_rng(2).normal(1, 0.3, size=(100, 6))) + 0.1
    sc = FeatureScaler().fit(f)
    m = sc.mean.copy()
    sc.transform(f * 100)
    assert np.array_equal(sc.mean, m)


# ---------------------------------------------------------------- inputs

def test_raw_input_layouts():
    streams, wins = make_dataset(n_event=1, n_quiet=1, hours=1.0)
    for layout, shape in (("NT", (5, WIN)), ("NCT", (5, 1, WIN)), ("NTC", (5, WIN, 1))):
        x = n2_inputs(streams, layout).fit(wins).build(wins[10:15])
        assert x.shape == shape and x.dtype == np.float32
    expect_error(lambda: RawWindowInput(None, "TN"))


def test_feature_input_shape_and_refit_resets():
    streams, wins = make_dataset(n_event=2, n_quiet=2, hours=1.0)
    norm = Normalizer(NormConfig("N0"))
    inp = FeatureInput(norm, FS)
    inp.fit([w for w in wins if w.event.evid in ("w00", "w01")])
    m1, s1 = inp.scaler.mean.copy(), norm.fixed_scale
    x = inp.build(wins[:7])
    assert x.shape == (7, 6) and x.dtype == np.float32 and np.isfinite(x).all()
    inp.fit([w for w in wins if w.event.evid in ("w02", "w03")])        # next fold
    assert not np.allclose(inp.scaler.mean, m1) and norm.fixed_scale != s1


def test_feature_input_unscaled_equals_extractor():
    streams, wins = make_dataset(n_event=1, n_quiet=1, hours=1.0)
    cache = ScaleCache.from_streams(streams, NormConfig("N2"))
    norm = Normalizer(NormConfig("N2"), cache)
    inp = FeatureInput(norm, FS, scale=False).fit(wins)
    assert np.allclose(inp.build(wins[:5]), extract_features(norm.windows(wins[:5]), FS))


# ---------------------------------------------------------------- config

def test_train_config_validation_and_pos_weight():
    for kw in (dict(epochs=0), dict(batch_size=0), dict(patience=0), dict(val_fraction=1.0),
               dict(val_fraction=-0.1), dict(negatives_per_positive=0), dict(pos_weight=0)):
        expect_error(lambda kw=kw: TrainConfig(**kw))
    assert TrainConfig(negatives_per_positive=5).effective_pos_weight == 5.0
    assert TrainConfig(pos_weight=2.0).effective_pos_weight == 2.0


# ---------------------------------------------------------------- harness

class SpySampler:
    """Wraps the real Sampler, records what every epoch was given and drew."""

    def __init__(self, cfg):
        from src.modules.sampler.module import Sampler
        self.inner = Sampler(cfg.negatives_per_positive, cfg.seed)
        self.given_ids, self.drawn = [], []

    def run(self, windows, epoch=0):
        self.given_ids.append({w.event.evid for w in windows})
        r = self.inner.run(windows, epoch)
        self.drawn.append([id(w) for w in r.windows if w.label == 0])
        self.n_pos = sum(w.label == 1 for w in r.windows)
        return r


def fit_detector(streams, wins, cfg=None, learner=None, inputs=None, name="t", sampler_factory=None):
    cfg = cfg or TrainConfig(epochs=25, patience=8, seed=1)
    det = LearnerDetector(
        name, learner or NumpyLogisticLearner(),
        inputs or n2_inputs(streams), cfg, sampler_factory=sampler_factory,
    )
    return det.fit(wins)


def test_detector_learns_and_scores_in_order():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    train = [w for w in wins if w.event.evid in ("w00", "w01", "w02", "w03", "w06")]
    test = [w for w in wins if w.event.evid in ("w04", "w05", "w07")]
    det = fit_detector(streams, train)
    s = det.score(test, {})
    assert s.shape == (len(test),) and np.isfinite(s).all()
    lab = np.array([w.label for w in test])
    keep = lab != -1
    ap = average_precision(lab[keep] == 1, s[keep])
    prevalence = (lab[keep] == 1).mean()
    assert ap > 20 * prevalence and ap > 0.5, (ap, prevalence)          # unseen waveforms
    assert det.score([], {}).shape == (0,)
    # order: scoring a reversed list reverses the scores
    assert np.allclose(det.score(test[::-1], {}), s[::-1])


def test_detector_determinism_and_seed():
    streams, wins = make_dataset()
    a = fit_detector(streams, wins).score(wins[:200], {})
    b = fit_detector(streams, wins).score(wins[:200], {})
    c = fit_detector(streams, wins, TrainConfig(epochs=25, patience=8, seed=2)).score(wins[:200], {})
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_refit_same_object_resets_between_folds():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    f0 = [w for w in wins if w.event.evid in ("w00", "w01", "w02", "w06")]
    f1 = [w for w in wins if w.event.evid in ("w03", "w04", "w05", "w07")]
    det = fit_detector(streams, f0)
    s_first = det.score(wins[:100], {})
    det.fit(f1)
    s_second = det.score(wins[:100], {})
    fresh = fit_detector(streams, f1).score(wins[:100], {})
    assert not np.allclose(s_first, s_second)
    assert np.array_equal(s_second, fresh)                               # nothing carried over


def test_validation_waveforms_never_trained_on_and_negatives_change_each_epoch():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    spies = []

    def factory(cfg):
        spies.append(SpySampler(cfg))
        return spies[-1]

    det = fit_detector(streams, wins, TrainConfig(epochs=6, patience=6, seed=3, val_fraction=0.25),
                       sampler_factory=factory)
    spy = spies[0]
    val_ids = set(det.info.val_waveforms)
    assert len(val_ids) == 2 and det.info.early_stopping
    for given in spy.given_ids:
        assert given.isdisjoint(val_ids)                                 # no leakage into training
    assert len(spy.drawn) == 6
    assert len({tuple(d) for d in spy.drawn}) == 6                       # a new draw every epoch
    assert spy.n_pos > 0


def test_positives_always_present_and_ignore_never_trained():
    streams, wins = make_dataset(n_event=4, n_quiet=2)
    seen = []

    class RecordingLearner(NumpyLogisticLearner):
        def train_step(self, x, y, pos_weight):
            seen.append(np.asarray(y))
            return super().train_step(x, y, pos_weight)

    fit_detector(streams, wins, TrainConfig(epochs=3, patience=3, seed=0), RecordingLearner())
    y = np.concatenate(seen)
    assert set(np.unique(y)) <= {0.0, 1.0}                               # label -1 never reaches the learner
    assert y.sum() > 0


class ScriptedLearner:
    """Validation quality per epoch is scripted: predict() call k returns quality[k]."""

    def __init__(self, quality):
        self.quality = quality
        self.calls = 0
        self.restored = None

    def reset(self, seed):
        self.calls = 0
        self.restored = None

    def train_step(self, x, y, pos_weight):
        return 0.0

    def predict(self, x):
        q = self.quality[min(self.calls, len(self.quality) - 1)]
        self.calls += 1
        x = np.asarray(x)[:, 0]
        return x * q + (np.arange(len(x)) * 0.618) % 1.0                   # label * q + jitter in [0, 1)

    def get_state(self):
        return self.calls

    def set_state(self, s):
        self.restored = s


class LabelInput:
    def fit(self, w):
        return self

    def build(self, w):
        return np.array([[float(x.label)] for x in w], dtype=np.float32)

    def ops_per_window(self, n):
        return 0


def test_early_stopping_restores_best_epoch():
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    learner = ScriptedLearner([0.2, 0.6, 1.5, 0.9, 0.5, 0.3, 0.2, 0.1])
    det = fit_detector(streams, wins, TrainConfig(epochs=8, patience=2, seed=0),
                       learner, LabelInput())
    assert det.info.best_epoch == 2 and det.info.stopped_early
    assert len(det.info.history) == 5                                    # stopped at epoch 4
    assert learner.restored == 3                                         # state saved after epoch 2's predict
    assert [h.best for h in det.info.history] == [True, True, True, False, False]


def test_no_early_stopping_runs_all_epochs():
    streams, wins = make_dataset()
    det = fit_detector(streams, wins, TrainConfig(epochs=4, val_fraction=0.0, seed=0))
    assert not det.info.early_stopping and len(det.info.history) == 4
    assert det.info.best_epoch == 3 and det.info.val_waveforms == []
    assert np.isnan(det.info.best_val_ap)


def test_validation_without_positives_disables_early_stopping_with_note():
    streams, wins = make_dataset(n_event=1, n_quiet=4)                   # one positive waveform only
    det = fit_detector(streams, wins, TrainConfig(epochs=3, val_fraction=0.4, seed=0))
    assert not det.info.early_stopping and "no positives" in det.info.note
    assert det.info.n_val_waveforms == 0


def test_score_before_fit_and_no_positives_error():
    streams, wins = make_dataset()
    det = LearnerDetector("x", NumpyLogisticLearner(), n2_inputs(streams))
    expect_error(lambda: det.score(wins[:5], {}), RuntimeError)
    expect_error(lambda: det.resource_report(wins[0]), RuntimeError)
    quiet = [w for w in wins if w.label == 0][:500]
    expect_error(lambda: det.fit(quiet))                                 # Sampler: no positive windows


def test_fits_through_cross_validate_budgets_style_loop():
    # cv.py: for each fold the SAME candidate is fitted, then scores validation then test
    streams, wins = make_dataset(n_event=6, n_quiet=2)
    det = LearnerDetector("cand", NumpyLogisticLearner(), n2_inputs(streams),
                          TrainConfig(epochs=5, patience=5, seed=0))
    folds = [("w00", "w01", "w02", "w06"), ("w03", "w04", "w05", "w07")]
    for ids in folds:
        det.fit([w for w in wins if w.event.evid in ids])
        out = det.score([w for w in wins if w.event.evid not in ids][:300], {})
        assert out.shape == (300,)
    assert det.name == "cand"
    assert len(det.fit_infos) == 2 and det.fit_infos[-1] is det.info


# ---------------------------------------------------------------- resources

def test_formulas():
    assert linear_macs(6, 16) == 96 and linear_macs(16, 1) == 16
    assert linear_macs(8, 4, positions=10) == 320
    assert conv1d_macs(1, 8, 5, 394) == 8 * 394 * 5
    assert conv1d_macs(8, 8, 3, 100, groups=8) == 8 * 100 * 1 * 3
    assert gru_macs(1, 8, 398) == 3 * (8 + 64) * 398
    assert gru_macs(4, 8, 10, layers=2) == 10 * 3 * (4 * 8 + 64) + 10 * 3 * (64 + 64)
    assert gru_macs(1, 8, 10, bidirectional=True) == 2 * gru_macs(1, 8, 10)
    assert lstm_macs(1, 8, 10) * 3 == gru_macs(1, 8, 10) * 4
    assert rnn_state_floats(8) == 8 and rnn_state_floats(8, 2, True, True) == 64


def test_model_bytes_and_feature_ops():
    assert model_bytes(1000, 4, 32) == 4000 and model_bytes(1000, 4, 8) == 1016
    expect_error(lambda: model_bytes(10, 1, 16))
    assert feature_ops(398) > feature_ops(100) > 0
    assert 5_000 < feature_ops(398) < 60_000


def test_detector_resource_report_numpy():
    streams, wins = make_dataset(n_event=3, n_quiet=1, hours=1.5)
    det = fit_detector(streams, wins, TrainConfig(epochs=2, patience=2, seed=0))
    r = det.resource_report(wins[0])
    assert r.params == 7 and r.macs_per_window == 6 and r.state_floats == 0
    assert r.bytes_fp32 == 28 and r.input_ops == feature_ops(WIN)
    assert "params 7" in r.summary()


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