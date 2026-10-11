"""
Tests of the frozen STA/LTA (selection, JSON round trip, application) on a
synthetic world. No data needed:   python tests/test_frozen.py
"""

import os
import tempfile
from types import SimpleNamespace as NS

import numpy as np

from src.modules.detectors import (
    FrozenSpec,
    evaluate_frozen,
    freeze_sta_lta_sets,
    sta_lta_grid,
)
from src.modules.evaluation import EvalConfig, Evaluator

FS = 6.625


def fake_window(wid, x, end_sample, length=398):
    s = end_sample - length
    return NS(event=NS(evid=wid), data=x[s:end_sample], start_sample=s, end_sample=end_sample,
              start_time=s / FS, end_time=end_sample / FS, label=0)


def target(evid, arrival, coda=2500.0):
    return NS(evid=evid, event_type="impact_mq", arrival_s=float(arrival),
              detect=(arrival - 120.0, arrival + 600.0),
              ignore=(arrival - 60.0, arrival + coda),
              onset_s=arrival + 20.0, has_positive=True, peak_ratio=15.0)


def make_world(n_streams=6, hours=3, amp=15.0):

    rng = np.random.default_rng(3)
    n = int(hours * 3600 * FS)
    t = np.arange(n) / FS

    streams, windows = {}, []

    for i in range(n_streams):

        wid = f"w{i}"
        arrival = 4000.0 + 700.0 * i
        x = rng.normal(size=n)

        a = np.zeros(n)
        r = (t >= arrival) & (t < arrival + 200.0)
        a[r] = amp * (t[r] - arrival) / 200.0
        d = t >= arrival + 200.0
        a[d] = amp * np.exp(-(t[d] - arrival - 200.0) / 800.0)
        x = x + a * rng.normal(size=n)

        streams[wid] = NS(waveform_id=wid, data=x, sampling_rate=FS, start_time=None,
                          events=[target(wid, arrival)], gaps=[])

        windows.extend(fake_window(wid, x, e) for e in range(398, n, 66))

    return streams, windows


GRID = lambda: sta_lta_grid(sta=(30.0, 120.0), lta=(900.0, 3600.0), features=("energy",))
BUDGETS = {"strict": 0.1, "lenient": 4.0}


def test_freeze_selects_one_parameter_set_and_ordered_thresholds():
    streams, windows = make_world()
    ev = Evaluator(EvalConfig(n_thresholds=30))
    ids = sorted(streams)

    specs = freeze_sta_lta_sets(windows, streams, ev, BUDGETS, {"all": ids}, GRID(), verbose=False)
    spec = specs["all"]

    assert set(spec.points) == {"strict", "lenient"}
    assert spec.points["strict"].threshold >= spec.points["lenient"].threshold   # stricter -> higher threshold
    assert spec.info["selection_waveforms"] == 6 and spec.info["grid_size"] == 4
    assert spec.info["selection_recall"]["lenient"] >= spec.info["selection_recall"]["strict"]
    assert "STA" in spec.describe() and "strict" in spec.describe()


def test_in_sample_application_reproduces_the_selection_numbers():
    streams, windows = make_world()
    ev = Evaluator(EvalConfig(n_thresholds=30))
    ids = sorted(streams)

    spec = freeze_sta_lta_sets(windows, streams, ev, BUDGETS, {"all": ids}, GRID(), verbose=False)["all"]
    res = evaluate_frozen(spec, windows, [streams[i] for i in ids], ev)

    for name in BUDGETS:
        assert abs(res[name].recall("all") - spec.info["selection_recall"][name]) < 1e-12
        fa = res[name].fa_per_day
        assert abs(fa - spec.info["selection_fa_per_day"][name]) < 1e-9
    assert res["lenient"].recall("all") >= res["strict"].recall("all")
    assert res["lenient"].n_alarms >= res["strict"].n_alarms


def test_json_round_trip_and_reuse():
    streams, windows = make_world()
    ev = Evaluator(EvalConfig(n_thresholds=30))
    ids = sorted(streams)

    spec = freeze_sta_lta_sets(windows, streams, ev, BUDGETS, {"all": ids}, GRID(), verbose=False)["all"]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "frozen.json")
        spec.save(path)
        loaded = FrozenSpec.load(path)

    assert loaded.to_dict() == spec.to_dict()
    assert loaded.detector().name == spec.detector().name

    a = evaluate_frozen(spec, windows, [streams[i] for i in ids], ev)
    b = evaluate_frozen(loaded, windows, [streams[i] for i in ids], ev)
    for name in BUDGETS:
        assert a[name].recall("all") == b[name].recall("all") and a[name].n_alarms == b[name].n_alarms


def test_sets_are_selected_independently_from_their_own_waveforms():
    streams, windows = make_world()
    ev = Evaluator(EvalConfig(n_thresholds=30))
    ids = sorted(streams)
    sets = {"all": ids, "part": ids[:3]}

    specs = freeze_sta_lta_sets(windows, streams, ev, BUDGETS, sets, GRID(), verbose=False)

    assert specs["all"].info["selection_waveforms"] == 6
    assert specs["part"].info["selection_waveforms"] == 3
    # a set's numbers come from its own waveforms: re-evaluating the part-spec on those waveforms matches
    res = evaluate_frozen(specs["part"], windows, [streams[i] for i in sets["part"]], ev)
    for name in BUDGETS:
        assert abs(res[name].recall("all") - specs["part"].info["selection_recall"][name]) < 1e-12


def test_application_to_other_waveforms_does_not_retune():
    streams, windows = make_world()
    ev = Evaluator(EvalConfig(n_thresholds=30))
    ids = sorted(streams)

    spec = freeze_sta_lta_sets(windows, streams, ev, BUDGETS, {"fit": ids[:3]}, GRID(), verbose=False)["fit"]
    before = spec.to_dict()

    res = evaluate_frozen(spec, windows, [streams[i] for i in ids[3:]], ev)

    assert spec.to_dict() == before                          # nothing changed by applying it
    assert {o.waveform_id for o in res["strict"].outcomes} == set(ids[3:])


if __name__ == "__main__":

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    for t in tests:
        t()
        print("ok", t.__name__)

    print()
    print(f"ALL {len(tests)} TESTS PASSED")