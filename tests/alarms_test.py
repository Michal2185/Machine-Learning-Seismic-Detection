"""Unit tests for the alarm / arrival comparison logic. Run: python tests/test_alarms.py"""

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.reporting import classify_alarms, extract_alarms, merge_intervals, scored_seconds


@dataclass
class E:
    evid: str
    arrival_s: float
    onset_s: float | None = None
    event_type: str = "impact_mq"
    detect: tuple = None
    ignore: tuple = None

    def __post_init__(self):
        if self.detect is None:
            self.detect = (self.arrival_s - 120, self.arrival_s + 600)
        if self.ignore is None:
            self.ignore = (self.arrival_s - 200, self.arrival_s + 3000)


def test_extract_alarms_refractory_and_order():
    t = np.arange(0, 2000, 10.0)
    s = np.zeros_like(t)
    s[(t >= 500) & (t <= 900)] = 1.0                    # stays above threshold for 400 s
    a = extract_alarms(t, s, 0.5, refractory_s=300)
    assert list(a) == [500.0, 800.0]                    # first crossing, silence, then re-arms
    assert list(extract_alarms(t[::-1], s[::-1], 0.5, 300)) == [500.0, 800.0]   # input order irrelevant
    assert len(extract_alarms(t, s, 2.0)) == 0
    assert list(extract_alarms(t, s, 1.0, 300)) == [500.0, 800.0]               # score == threshold fires


def test_extract_alarms_shape_check():
    try:
        extract_alarms([1, 2], [1], 0.5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_merge_and_scored_seconds():
    assert merge_intervals([(0, 10), (5, 20), (30, 40)]) == [(0.0, 20.0), (30.0, 40.0)]
    ev = [E("a", 1000.0)]                                # ignore (800, 4000), detect (880, 1600)
    gaps = [(5000.0, 5500.0), (3900.0, 4200.0)]
    # excluded = [800, 4200] u [5000, 5500] = 3400 + 500
    assert scored_seconds(10000.0, ev, gaps) == 10000 - 3400 - 500
    assert scored_seconds(10000.0, [], []) == 10000.0
    near_start = [E("b", 100.0)]                         # ignore starts before the file: clipped at 0
    assert scored_seconds(5000.0, near_start, []) == 5000 - 3100


def test_classification_labels():
    ev = [E("e1", 10000.0, onset_s=10090.0)]
    alarms = [2000.0,                 # background -> false
              9900.0,                 # 100 s before arrival: inside detect -> detection
              10200.0,                # second alarm inside detect -> duplicate
              11000.0,                # inside ignore (coda), outside detect -> neutral
              20000.0]                # background -> false
    r = classify_alarms(alarms, ev, gaps=[], duration_s=86400.0)
    assert r.labels == ["false", "detection", "duplicate", "neutral", "false"]
    assert r.n_false == 2 and r.n_detected == 1
    o = r.outcomes[0]
    assert o.detected and o.alarm_s == 9900.0
    assert o.delay_arrival_s == -100.0 and o.delay_onset_s == -190.0


def test_gap_alarm_is_neutral_and_missed_event():
    ev = [E("e1", 10000.0)]
    r = classify_alarms([50000.0, 3000.0], ev, gaps=[(49000.0, 51000.0)], duration_s=86400.0)
    assert r.labels == ["false", "neutral"] or r.labels == ["false", "neutral"][::-1]
    assert dict(zip(r.alarm_s.tolist(), r.labels))[50000.0] == "neutral"
    assert not r.outcomes[0].detected and r.outcomes[0].delay_arrival_s is None
    assert r.n_detected == 0


def test_two_events_one_day_and_fa_rate():
    ev = [E("a", 10000.0), E("b", 40000.0)]
    r = classify_alarms([10300.0, 40500.0, 70000.0, 80000.0], ev, [], 86400.0)
    assert r.labels == ["detection", "detection", "false", "false"]
    assert r.n_detected == 2
    expected_scored = scored_seconds(86400.0, ev, [])
    assert abs(r.fa_per_day - 2 / (expected_scored / 86400.0)) < 1e-9
    assert r.outcomes[1].delay_arrival_s == 500.0 and r.outcomes[1].delay_onset_s is None


def test_no_alarms():
    r = classify_alarms([], [E("a", 10000.0)], [], 86400.0)
    assert r.labels == [] and r.n_false == 0 and r.fa_per_day == 0.0
    assert not r.outcomes[0].detected


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