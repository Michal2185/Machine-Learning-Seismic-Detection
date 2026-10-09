"""
Unit tests of the event-level evaluator on synthetic streams with known
answers. No data needed:  python tests/test_evaluator.py
"""

from types import SimpleNamespace as NS

import numpy as np

from src.modules.evaluation import (
    EvalConfig,
    Evaluator,
    ScoreSeries,
    average_precision,
    build_series,
    contains,
    merge_intervals,
    pool_results,
    union_length,
)


DURATION = 86400


def target(evid, arrival, usable=True, etype="impact_mq", coda=2000.0, onset=None):
    return NS(
        evid=evid,
        event_type=etype,
        arrival_s=float(arrival),
        detect=(arrival - 120.0, arrival + 600.0),
        ignore=(arrival - 60.0, arrival + coda),
        onset_s=onset,
        has_positive=usable,
        peak_ratio=10.0,
    )


def stream(wid, events, gaps=()):
    return NS(
        waveform_id=wid,
        data=np.zeros(DURATION, dtype=np.float32),
        sampling_rate=1.0,
        events=list(events),
        gaps=list(gaps),
    )


def series(wid, hits, step=10.0):
    """Scores 0 everywhere, hits = {time: score}."""
    t = np.arange(60.0, DURATION, step)
    s = np.zeros_like(t)
    for time, value in hits.items():
        s[np.argmin(np.abs(t - time))] = value
    return ScoreSeries(wid, t, s)


def run(st, hits, threshold=0.5, **cfg):
    ev = Evaluator(EvalConfig(**cfg))
    return ev, ev.evaluate([st], {st.waveform_id: series(st.waveform_id, hits)}, threshold)


# ---------------------------------------------------------------- helpers

def test_interval_helpers():
    m = merge_intervals([(10, 20), (15, 30), (40, 50), (50, 60), (70, 70)])
    assert m.tolist() == [[10, 30], [40, 60]]
    t = np.array([5, 10, 25, 30, 35, 40, 60, 61])
    assert contains(m, t).tolist() == [False, True, True, True, False, True, True, False]
    assert union_length(m, 100) == 40 and union_length(m, 50) == 30   # clipped to the stream
    assert len(merge_intervals([])) == 0 and not contains(merge_intervals([]), t).any()


# ---------------------------------------------------------------- detection logic

def test_perfect_detection():
    st = stream("w", [target("e1", 10000)])
    _, r = run(st, {10050: 1.0, 10100: 1.0})
    o = r.outcomes[0]
    assert o.detected and o.alarm_s == 10050 and o.delay_s == 50
    assert r.n_alarms == 1 and r.n_false_alarms == 0 and r.recall() == 1.0


def test_late_alarm_misses_the_event_but_is_not_a_false_alarm():
    st = stream("w", [target("e1", 10000)])
    _, r = run(st, {10700: 1.0})                  # after detect end (10600), inside the ignore zone
    assert not r.outcomes[0].detected and np.isnan(r.outcomes[0].delay_s)
    assert r.n_false_alarms == 0 and r.streams[0].n_neutral == 1


def test_early_alarm_inside_detect_interval_has_negative_delay():
    st = stream("w", [target("e1", 10000)])
    _, r = run(st, {9900: 1.0})
    assert r.outcomes[0].detected and r.outcomes[0].delay_s == -100


def test_alarm_before_detect_interval_is_a_false_alarm():
    st = stream("w", [target("e1", 10000, coda=2000.0)])
    _, r = run(st, {9800: 1.0})                    # before ignore (9940) and detect (9880)
    assert not r.outcomes[0].detected and r.n_false_alarms == 1


def test_false_alarm_rate_uses_scored_hours():
    st = stream("w", [target("e1", 10000)])
    _, r = run(st, {50000: 1.0})
    # scored time = 86400 - |ignore U detect| = 86400 - (12000 - 9880)
    assert abs(r.background_hours - (DURATION - 2120) / 3600) < 1e-9
    assert r.n_false_alarms == 1
    assert abs(r.fa_per_day - 24 / r.background_hours) < 1e-9


def test_alarm_inside_gap_is_neutral_and_gap_time_is_not_scored():
    st = stream("w", [], gaps=[(30000.0, 31000.0)])
    _, r = run(st, {30500: 1.0})
    assert r.n_false_alarms == 0 and r.streams[0].n_neutral == 1
    assert abs(r.background_hours - (DURATION - 1000) / 3600) < 1e-9


def test_refractory_period():
    st = stream("w", [])
    hits = {t: 1.0 for t in range(50000, 51001, 10)}
    _, r = run(st, hits, refractory_s=300.0)
    assert r.streams[0].alarms.tolist() == [50000, 50300, 50600, 50900]
    assert r.n_false_alarms == 4
    _, r2 = run(st, hits, refractory_s=2000.0)
    assert r2.n_alarms == 1


def test_duplicate_alarm_in_a_detect_interval_is_neutral():
    st = stream("w", [target("e1", 10000)])
    _, r = run(st, {10050: 1.0, 10400: 1.0}, refractory_s=300.0)
    assert r.n_alarms == 2 and r.n_false_alarms == 0 and r.outcomes[0].alarm_s == 10050


def test_one_alarm_can_detect_two_events():
    st = stream("w", [target("e1", 20000), target("e2", 20300)])
    _, r = run(st, {20250: 1.0})
    assert all(o.detected for o in r.outcomes) and r.n_false_alarms == 0 and r.n_alarms == 1


def test_groups_and_types():
    st = stream("w", [target("e1", 10000, usable=True, etype="impact_mq"),
                      target("e2", 40000, usable=False, etype="deep_mq")])
    _, r = run(st, {10100: 1.0})
    assert r.recall("all") == 0.5 and r.recall("usable") == 1.0 and r.recall("weak") == 0.0
    assert r.recall("impact_mq") == 1.0 and r.recall("deep_mq") == 0.0
    assert np.isnan(r.recall("shallow_mq"))


def test_delay_relative_to_visible_onset():
    st = stream("w", [target("e1", 10000, onset=10040.0)])
    _, r = run(st, {10050: 1.0})
    assert r.delays("all", "arrival").tolist() == [50.0]
    assert r.delays("all", "onset").tolist() == [10.0]


def test_missing_scores_raise():
    ev = Evaluator()
    try:
        ev.evaluate([stream("w", [])], {}, 0.5)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


# ---------------------------------------------------------------- curve, threshold, bootstrap

def make_curve_case():
    st = stream("w", [target("A", 10000), target("B", 40000)])
    hits = {10100: 0.9, 40100: 0.5, 70000: 0.7}       # detect A, detect B, false alarm
    return st, {"w": series("w", hits)}


def test_curve_and_threshold_selection():
    st, ser = make_curve_case()
    ev = Evaluator()
    c = ev.curve([st], ser, thresholds=[0.4, 0.6, 0.8, 0.95])
    assert c.recall_all.tolist() == [1.0, 0.5, 0.5, 0.0]
    fa = c.fa_per_day
    assert fa[0] > 0.9 and fa[1] > 0.9 and fa[2] == 0 and fa[3] == 0
    assert ev.select_threshold(c, max_fa_per_day=0.5) == 0.8      # best recall without false alarms
    assert ev.select_threshold(c, max_fa_per_day=2.0) == 0.4      # budget allows the false alarm
    assert ev.select_threshold(c, max_fa_per_day=-1.0) == 0.95    # budget impossible -> strictest


def test_default_thresholds_are_sorted_and_in_range():
    st, ser = make_curve_case()
    thr = Evaluator().default_thresholds(ser)
    assert np.all(np.diff(thr) > 0) and thr.min() >= 0 and thr.max() <= 0.9


def test_bootstrap():
    ev = Evaluator(EvalConfig(n_bootstrap=300, seed=1))
    # six streams, everything detected, no false alarms
    sts = [stream(f"w{i}", [target(f"e{i}", 10000)]) for i in range(6)]
    ser = {s.waveform_id: series(s.waveform_id, {10100: 1.0}) for s in sts}
    r = ev.evaluate(sts, ser, 0.5)
    iv = ev.bootstrap(r)
    assert iv["recall_all"] == (1.0, 1.0, 1.0) and iv["fa_per_day"] == (0.0, 0.0, 0.0)
    # half detected -> interval brackets the point estimate and is reproducible
    ser2 = {s.waveform_id: series(s.waveform_id, {10100: 1.0} if i % 2 else {}) for i, s in enumerate(sts)}
    r2 = ev.evaluate(sts, ser2, 0.5)
    a, b = ev.bootstrap(r2), ev.bootstrap(r2)
    p, lo, hi = a["recall_all"]
    assert p == 0.5 and lo <= p <= hi and lo < hi
    # no weak events in this case -> recall_weak is (nan, nan, nan); NaN != NaN,
    # so compare with a NaN-aware check
    assert np.isnan(a["recall_weak"]).all()
    np.testing.assert_equal(a, b)                       # reproducible with the same seed


def test_pooling_folds():
    sts = [stream(f"w{i}", [target(f"e{i}", 10000)]) for i in range(4)]
    ev = Evaluator()
    r1 = ev.evaluate(sts[:2], {s.waveform_id: series(s.waveform_id, {10100: 1.0}) for s in sts[:2]}, 0.5)
    r2 = ev.evaluate(sts[2:], {s.waveform_id: series(s.waveform_id, {}) for s in sts[2:]}, 0.5)
    pooled = pool_results([r1, r2])
    assert pooled.n_events() == 4 and pooled.recall() == 0.5


# ---------------------------------------------------------------- window-level helpers

def test_average_precision():
    assert abs(average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) - (1 + 2 / 3) / 2) < 1e-12
    assert average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))


def test_build_series():
    w = lambda wid, end: NS(event=NS(evid=wid), end_time=end)
    windows = [w("a", 30.0), w("b", 10.0), w("a", 10.0), w("a", 20.0)]
    out = build_series(windows, [3, 9, 1, 2])
    assert out["a"].times.tolist() == [10.0, 20.0, 30.0] and out["a"].scores.tolist() == [1, 2, 3]
    assert out["b"].scores.tolist() == [9]
    try:
        build_series(windows, [1, 2])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_series_validation():
    try:
        ScoreSeries("w", [3.0, 1.0], [0.0, 0.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsorted times")


if __name__ == "__main__":

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    for t in tests:
        t()
        print("ok", t.__name__)

    print()
    print(f"ALL {len(tests)} TESTS PASSED")