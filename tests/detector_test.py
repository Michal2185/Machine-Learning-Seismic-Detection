"""
Tests of the STA/LTA detectors and of the cross-validation runner on
synthetic data with known answers. No data needed:
    python tests/test_detectors.py
"""

from types import SimpleNamespace as NS

import numpy as np

from src.modules.detectors import (
    StaLtaDetector,
    WindowStaLtaDetector,
    sta_lta_at,
    sta_lta_grid,
    valid_mask,
    window_sta_lta_grid,
)
from src.modules.evaluation import EvalConfig, Evaluator, cross_validate, cross_validate_budgets

FS = 6.625


def noise_with_step(n_seconds=10800, step_at=7200.0, amp=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(n_seconds * FS)
    x = rng.normal(size=n)
    x[int(step_at * FS):] *= amp
    return x


# ---------------------------------------------------------------- sta_lta_at

def test_ratio_on_noise_and_after_a_step():
    x = noise_with_step()
    valid = np.ones(len(x))

    before = sta_lta_at(x, valid, FS, 30, 3600, "energy", np.array([int(7000 * FS)]))[0]
    after = sta_lta_at(x, valid, FS, 30, 3600, "energy", np.array([int(7430 * FS)]))[0]

    # STA fully inside the event: energy 100; LTA = (3370 * 1 + 230 * 100) / 3600 = 7.33
    assert 0.7 < before < 1.4, before
    assert 11.0 < after < 16.5, after


def test_scale_invariance():
    x = noise_with_step()
    idx = np.array([int(7430 * FS)])
    a = sta_lta_at(x, np.ones(len(x)), FS, 30, 3600, "energy", idx)
    b = sta_lta_at(x * 1e-10, np.ones(len(x)), FS, 30, 3600, "energy", idx)
    assert abs(a[0] - b[0]) / a[0] < 1e-9


def test_warm_up_gives_zero():
    x = noise_with_step()
    s = sta_lta_at(x, np.ones(len(x)), FS, 30, 3600, "energy",
                   np.array([int(300 * FS), int(700 * FS)]), min_lta_s=600)
    assert s[0] == 0.0 and s[1] > 0.0


def test_gap_masking():
    rng = np.random.default_rng(1)
    x = rng.normal(size=int(10000 * FS))
    gap = (5000.0, 6000.0)
    x[int(gap[0] * FS):int(gap[1] * FS)] = 0.0           # filtered interpolated stretch ~ 0
    idx = np.array([int(6030 * FS)])

    masked = sta_lta_at(x, valid_mask(len(x), FS, [gap]), FS, 30, 3600, "energy", idx)[0]
    unmasked = sta_lta_at(x, np.ones(len(x)), FS, 30, 3600, "energy", idx)[0]

    assert 0.8 < masked < 1.25, masked                    # LTA only uses observed samples
    assert unmasked > 1.2 * masked, (masked, unmasked)    # an unmasked gap inflates the ratio

    inside = sta_lta_at(x, valid_mask(len(x), FS, [gap]), FS, 30, 3600, "energy",
                        np.array([int(5500 * FS)]))[0]
    assert inside == 0.0                                  # STA window mostly missing -> no decision


def test_abs_feature_differs_from_energy():
    x = noise_with_step()
    idx = np.array([int(7430 * FS)])
    e = sta_lta_at(x, np.ones(len(x)), FS, 30, 3600, "energy", idx)[0]
    a = sta_lta_at(x, np.ones(len(x)), FS, 30, 3600, "abs", idx)[0]
    assert a < e and a > 2


# ---------------------------------------------------------------- detector classes

def fake_window(wid, x, end_sample, length=398):
    s = end_sample - length
    return NS(event=NS(evid=wid), data=x[s:end_sample], start_sample=s, end_sample=end_sample,
              start_time=s / FS, end_time=end_sample / FS, label=0)


def test_sta_lta_detector_matches_function_and_checks_names():
    x = noise_with_step()
    stream = NS(data=x, sampling_rate=FS, gaps=[])
    ends = [int(t * FS) for t in (7000, 7430, 9000)]
    wins = [fake_window("w", x, e) for e in ends]

    det = StaLtaDetector(30, 3600, "energy")
    got = det.score(wins, {"w": stream})
    want = sta_lta_at(x, np.ones(len(x)), FS, 30, 3600, "energy", np.array(ends) - 1)
    np.testing.assert_allclose(got, want)
    assert "sta=30s" in det.name and "lta=3600s" in det.name
    assert det.fit([]) is det

    try:
        StaLtaDetector(30, 3600, "median")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_window_sta_lta():
    rng = np.random.default_rng(2)
    n = 398
    quiet = rng.normal(size=n)
    event = rng.normal(size=n)
    event[int(n * 40 / 60):] *= 10                    # last 20 s are 10x stronger
    wins = [NS(data=quiet, start_time=0.0, end_time=60.08),
            NS(data=event, start_time=0.0, end_time=60.08)]

    s = WindowStaLtaDetector(20.0, "energy").score(wins)
    assert 0.5 < s[0] < 2.0 and s[1] > 50

    s2 = WindowStaLtaDetector(20.0, "abs").score(wins)
    assert s2[1] > 5 and s2[1] > s2[0] * 5

    zero = WindowStaLtaDetector(20.0).score([NS(data=np.zeros(n), start_time=0.0, end_time=60.08)])
    assert zero[0] == 0.0                              # an all-zero window is not an event


def test_grids():
    g = sta_lta_grid()
    assert len(g) == 84 and len({d.name for d in g}) == 84          # 6 STA x 7 LTA x 2 features
    old = sta_lta_grid(sta=(10.0, 30.0, 60.0, 120.0), lta=(900.0, 3600.0, 7200.0))
    assert len(old) == 24
    assert max(d.lta_s for d in g) == 86400 and max(d.sta_s for d in g) == 480   # wider than the earlier grids
    assert len(window_sta_lta_grid()) == 6
    assert all(d.lta_s > d.sta_s for d in g)


# ---------------------------------------------------------------- cross-validation runner

def target(evid, arrival, coda=2500.0):
    return NS(evid=evid, event_type="impact_mq", arrival_s=float(arrival),
              detect=(arrival - 120.0, arrival + 600.0),
              ignore=(arrival - 60.0, arrival + coda),
              onset_s=arrival + 20.0, has_positive=True, peak_ratio=15.0)


def make_world(n_streams=6, hours=3, amp=15.0):

    rng = np.random.default_rng(3)
    n = int(hours * 3600 * FS)
    t = np.arange(n) / FS

    streams, windows = {}, {}

    for i in range(n_streams):

        wid = f"w{i}"
        arrival = 4000.0 + 700.0 * i
        x = rng.normal(size=n)

        a = np.zeros(n)
        ramp, tau = 200.0, 800.0
        r = (t >= arrival) & (t < arrival + ramp)
        a[r] = amp * (t[r] - arrival) / ramp
        d = t >= arrival + ramp
        a[d] = amp * np.exp(-(t[d] - arrival - ramp) / tau)
        x = x + a * rng.normal(size=n)

        tg = target(wid, arrival)

        streams[wid] = NS(waveform_id=wid, data=x, sampling_rate=FS, start_time=None,
                          events=[tg], gaps=[])

        ws = []
        for e in range(398, n, 66):
            w = fake_window(wid, x, e)
            if tg.ignore[0] < w.end_time and w.start_time < tg.ignore[1]:
                w.label = 1 if (w.start_time <= tg.onset_s and w.end_time >= tg.onset_s + 20) else -1
            ws.append(w)
        windows[wid] = ws

    return streams, windows


class FakeBundle:

    def __init__(self, streams, windows, n_folds=3):
        self.streams, self.windows = streams, windows
        self.ids = sorted(streams)
        self.folds = NS(n_folds=n_folds)

    def fold_split(self, k):
        n = self.folds.n_folds
        test = [w for i, w in enumerate(self.ids) if i % n == k]
        val = [w for i, w in enumerate(self.ids) if i % n == (k + 1) % n]
        train = [w for w in self.ids if w not in test and w not in val]
        collect = lambda ids: [w for i in ids for w in self.windows[i]]
        return NS(train=collect(train), validation=collect(val), test=collect(test),
                  train_events=train, validation_events=val, test_events=test)

    def eval_streams(self, ids):
        return [self.streams[i] for i in ids]


def test_cross_validate_runs_and_finds_strong_events():
    streams, windows = make_world()
    bundle = FakeBundle(streams, windows)
    ev = Evaluator(EvalConfig(n_thresholds=30, n_bootstrap=100))

    candidates = (sta_lta_grid(sta=(30.0,), lta=(900.0,), features=("energy",))
                  + window_sta_lta_grid(sta=(20.0,), features=("energy",)))

    res = cross_validate(bundle, candidates, ev, name="synthetic", verbose=False)

    assert len(res.folds) == 3
    tested = [o.waveform_id for o in res.pooled.outcomes]
    assert sorted(tested) == bundle.ids and len(set(tested)) == len(tested)   # each waveform tested once
    assert res.pooled.recall("all") >= 0.8, res.pooled.recall("all")
    assert all(not np.isnan(f.test_ap) for f in res.folds)
    assert all(f.candidate in {c.name for c in candidates} for f in res.folds)
    res.print_summary(ev)


def test_cross_validate_budgets():
    streams, windows = make_world()
    bundle = FakeBundle(streams, windows)
    ev = Evaluator(EvalConfig(n_thresholds=30, n_bootstrap=50))
    cands = sta_lta_grid(sta=(30.0,), lta=(900.0, 3600.0), features=("energy",))

    out = cross_validate_budgets(bundle, cands, ev, [0.05, 5.0], name="synthetic", verbose=False)

    assert sorted(out) == [0.05, 5.0]
    for res in out.values():
        tested = sorted(o.waveform_id for o in res.pooled.outcomes)
        assert tested == bundle.ids                      # each waveform tested once per budget
    # a larger budget never lowers the validation recall the thresholds were chosen for
    lo = np.mean([f.val_recall for f in out[0.05].folds])
    hi = np.mean([f.val_recall for f in out[5.0].folds])
    assert hi >= lo - 1e-9
    # single-budget wrapper agrees with the multi-budget runner
    single = cross_validate(bundle, cands, Evaluator(EvalConfig(n_thresholds=30, max_fa_per_day=5.0)), verbose=False)
    assert [f.threshold for f in single.folds] == [f.threshold for f in out[5.0].folds]


if __name__ == "__main__":

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    for t in tests:
        t()
        print("ok", t.__name__)

    print()
    print(f"ALL {len(tests)} TESTS PASSED")