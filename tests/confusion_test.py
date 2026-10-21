"""Unit tests for the confusion metrics. Run: python tests/test_confusion.py"""

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alarms_test import E

from src.modules.reporting import (
    Confusion,
    classify_alarms,
    event_confusion,
    event_summary,
    pool_event_summary,
    window_confusion,
)


def test_counts_and_rates():
    c = Confusion(tp=8, fp=2, fn=2, tn=88)
    assert c.n == 100 and abs(c.prevalence - 0.1) < 1e-12
    assert abs(c.recall - 0.8) < 1e-12 and abs(c.fnr - 0.2) < 1e-12
    assert abs(c.recall + c.fnr - 1.0) < 1e-12                     # fnr is 1 - recall
    assert abs(c.specificity - 88 / 90) < 1e-12 and abs(c.fpr - 2 / 90) < 1e-12
    assert abs(c.specificity + c.fpr - 1.0) < 1e-12
    assert abs(c.precision - 0.8) < 1e-12 and abs(c.f1 - 0.8) < 1e-12
    assert abs(c.balanced_accuracy - 0.5 * (0.8 + 88 / 90)) < 1e-12
    expected_mcc = (8 * 88 - 2 * 2) / np.sqrt(10 * 10 * 90 * 90)
    assert abs(c.mcc - expected_mcc) < 1e-12


def test_degenerate_cases_are_nan_not_errors():
    assert np.isnan(Confusion(0, 0, 0, 10).recall) and np.isnan(Confusion(0, 0, 0, 10).precision)
    assert np.isnan(Confusion(5, 0, 0, 0).specificity) and np.isnan(Confusion(5, 0, 0, 0).mcc)
    assert Confusion(5, 0, 0, 0).recall == 1.0
    assert np.isnan(Confusion(0, 0, 0, 0).prevalence)
    assert np.isnan(Confusion(5, 0, 0, 0).balanced_accuracy)


def test_addition_pools_counts():
    a, b = Confusion(1, 2, 3, 4), Confusion(10, 20, 30, 40)
    assert a + b == Confusion(11, 22, 33, 44)
    assert Confusion(11, 22, 33, 44).as_dict()["tp"] == 11


def test_window_confusion_drops_ignore_and_uses_ge_threshold():
    y = np.array([1, 1, 1, 0, 0, 0, 0, -1, -1])
    s = np.array([0.9, 0.5, 0.1, 0.5, 0.4, 0.0, 0.6, 0.99, 0.99])
    c = window_confusion(y, s, 0.5)
    assert c == Confusion(tp=2, fp=2, fn=1, tn=2)                  # 0.5 counts as alarm; -1 ignored
    assert c.n == 7


def test_window_confusion_errors_and_empty():
    try:
        window_confusion([1, 0], [0.5], 0.5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert window_confusion([], [], 0.5) == Confusion(0, 0, 0, 0)
    assert window_confusion([-1, -1], [1.0, 1.0], 0.5).n == 0


def test_event_confusion_and_summary():
    ev = [E("a", 10000.0), E("b", 40000.0), E("c", 60000.0)]
    # a detected (+duplicate), b missed, c detected; one neutral in a's coda; two false alarms
    alarms = [2000.0, 10100.0, 10300.0, 11000.0, 60200.0, 80000.0]
    r = classify_alarms(alarms, ev, [], 86400.0)
    c = event_confusion(r)
    assert c == Confusion(tp=2, fp=2, fn=1, tn=None)
    s = event_summary(r)
    assert s["events"] == 3 and s["detected"] == 2 and s["missed"] == 1 and s["false_alarms"] == 2
    assert abs(s["event_recall"] - 2 / 3) < 1e-12 and abs(s["event_fnr"] - 1 / 3) < 1e-12
    assert abs(s["episode_precision"] - 0.5) < 1e-12              # detected / (detected + false alarms)
    assert s["duplicates"] == 1 and s["neutral"] == 1
    assert np.isnan(c.specificity) and np.isnan(c.fpr) and np.isnan(c.mcc)   # no TN at event level
    assert Confusion(1, 1, 1, None) + Confusion(1, 1, 1, 5) == Confusion(2, 2, 2, None)


def test_pool_event_summary():
    ev1, ev2 = [E("a", 10000.0)], [E("b", 20000.0), E("c", 50000.0)]
    r1 = classify_alarms([10100.0, 70000.0], ev1, [], 86400.0)       # detected, 1 false alarm
    r2 = classify_alarms([], ev2, [], 86400.0)                        # both missed
    p = pool_event_summary([r1, r2])
    assert p["events"] == 3 and p["detected"] == 1 and p["missed"] == 2 and p["false_alarms"] == 1
    assert abs(p["event_recall"] - 1 / 3) < 1e-12 and abs(p["episode_precision"] - 0.5) < 1e-12
    secs = r1.scored_seconds + r2.scored_seconds
    assert abs(p["fa_per_day"] - 1 / (secs / 86400.0)) < 1e-9
    assert pool_event_summary([])["events"] == 0


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