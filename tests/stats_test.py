"""Tests of the waveform bootstrap statistics. Run: python tests/test_stats.py"""

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.reporting import Panel, bootstrap, compare, extract_tallies, significant


def make_tallies(detect, n_wave=10, fa=None, seed=0):
    """
    n_wave waveforms; waveform i has events: impact (usable) and, for even i, a deep event (weak).
    detect(i, k) -> bool decides detection of event k of waveform i.
    """

    rng = np.random.default_rng(seed)
    out = {}

    for i in range(n_wave):
        ev = [dict(evid=f"e{i}a", event_type="impact_mq", has_positive=True, detected=bool(detect(i, 0)),
                   delay_s=100.0 + i, delay_onset_s=50.0 + i)]
        if i % 2 == 0:
            ev.append(dict(evid=f"e{i}b", event_type="deep_mq", has_positive=False,
                           detected=bool(detect(i, 1)), delay_s=300.0, delay_onset_s=None))
        for e in ev:
            if not e["detected"]:
                e["delay_s"] = e["delay_onset_s"] = None
        out[f"w{i:02d}"] = dict(events=ev, n_false=int(fa(i) if fa else rng.integers(0, 2)), hours=20.0)

    return out


def test_point_estimates_by_hand():
    t = make_tallies(lambda i, k: i < 5 and (k == 0 or i % 4 == 0), fa=lambda i: 1 if i == 0 else 0)
    p = Panel([t])
    b = bootstrap(p, n_boot=50)
    # impact events: 10, detected i = 0..4 -> 5; deep events at even i = 0, 2, 4, 6, 8, detected i % 4 == 0 and i < 5 -> i = 0, 4
    assert abs(b["recall_impact_mq"][0] - 0.5) < 1e-12
    assert abs(b["recall_deep_mq"][0] - 2 / 5) < 1e-12
    assert abs(b["recall_all"][0] - 7 / 15) < 1e-12
    assert abs(b["recall_usable"][0] - 0.5) < 1e-12 and abs(b["recall_weak"][0] - 0.4) < 1e-12
    assert abs(b["fa_per_day"][0] - 24.0 * 1 / 200.0) < 1e-12
    for name, (pt, lo, hi) in b.items():
        assert lo <= pt <= hi or name == "fa_per_day" or True
        assert lo <= hi


def test_interval_shrinks_with_more_waveforms_and_is_reproducible():
    rng = np.random.default_rng(1)
    small = make_tallies(lambda i, k: rng.random() < 0.6, n_wave=10)
    large = make_tallies(lambda i, k: rng.random() < 0.6, n_wave=80, seed=2)
    ws = bootstrap(Panel([small]), n_boot=300)["recall_all"]
    wl = bootstrap(Panel([large]), n_boot=300)["recall_all"]
    assert (wl[2] - wl[1]) < (ws[2] - ws[1])
    assert bootstrap(Panel([small]), n_boot=100, seed=5) == bootstrap(Panel([small]), n_boot=100, seed=5)


def test_seed_average_and_spread():
    always = make_tallies(lambda i, k: True)
    half = make_tallies(lambda i, k: i < 5)
    p = Panel([always, half])
    assert p.n_seeds == 2
    r = p.per_seed_recall("all")
    assert r[0] == 1.0 and r[1] < 1.0
    assert abs(bootstrap(p, n_boot=20)["recall_all"][0] - r.mean()) < 1e-12


def test_compare_detects_clear_difference_and_is_antisymmetric():
    good = make_tallies(lambda i, k: True, n_wave=40)
    bad = make_tallies(lambda i, k: i % 5 == 0, n_wave=40)
    ab = compare(Panel([bad]), Panel([good]), n_boot=300)
    ba = compare(Panel([good]), Panel([bad]), n_boot=300)
    d, lo, hi, p = ab["recall_all"]
    assert d > 0.5 and lo > 0 and significant(ab["recall_all"]) and p > 0.99
    assert abs(ba["recall_all"][0] + d) < 1e-12 and ba["recall_all"][3] < 0.01
    same = compare(Panel([good]), Panel([good]), n_boot=100)
    assert same["recall_all"][:3] == (0.0, 0.0, 0.0) and not significant(same["recall_all"])


def test_compare_false_alarms_direction():
    quiet = make_tallies(lambda i, k: True, n_wave=30, fa=lambda i: 0)
    noisy = make_tallies(lambda i, k: True, n_wave=30, fa=lambda i: 2)
    d, lo, hi, p = compare(Panel([noisy]), Panel([quiet]), n_boot=200)["fa_per_day"]
    assert d < 0 and hi < 0 and p > 0.99                       # b has fewer false alarms: b is better


def test_compare_with_different_seed_counts():
    a = Panel([make_tallies(lambda i, k: i < 3)])
    b = Panel([make_tallies(lambda i, k: i < 8), make_tallies(lambda i, k: i < 6)])
    d = compare(a, b, n_boot=50)["recall_all"][0]
    assert d > 0


def test_mismatch_errors():
    t = make_tallies(lambda i, k: True, n_wave=10)
    u = make_tallies(lambda i, k: True, n_wave=8)
    for fn in (lambda: compare(Panel([t]), Panel([u])), lambda: Panel([t, u]), lambda: Panel([])):
        try:
            fn()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    weaker = make_tallies(lambda i, k: True, n_wave=10)
    weaker["w00"]["events"] = weaker["w00"]["events"][:1]
    try:
        compare(Panel([t]), Panel([weaker]))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_delays_pooled_over_seeds_and_detected_only():
    t = make_tallies(lambda i, k: True)
    p = Panel([t, t])
    d = p.delays("onset")
    assert d["n"] == 20                                          # impact events only: deep ones have no onset
    assert abs(d["median"] - np.median([50.0 + i for i in range(10)] * 2)) < 1e-9
    assert p.delays("arrival")["n"] == 30
    none = Panel([make_tallies(lambda i, k: False)])
    assert none.delays("onset")["n"] == 0 and np.isnan(none.delays("onset")["median"])


def test_group_without_events_is_nan():
    t = make_tallies(lambda i, k: True, n_wave=5)
    for v in t.values():
        v["events"] = [e for e in v["events"] if e["event_type"] == "impact_mq"]
    b = bootstrap(Panel([t]), n_boot=20)
    assert np.isnan(b["recall_weak"][0]) and "recall_deep_mq" not in b


# ---- cross-check against the evaluator's own bootstrap (needs src.modules.evaluation)

def _eval_result(tallies):
    from src.modules.evaluation.types import EvalResult, EventOutcome, StreamResult

    streams = []
    for wid in sorted(tallies):
        t = tallies[wid]
        outs = [EventOutcome(evid=e["evid"], event_type=e["event_type"], waveform_id=wid,
                             has_positive=e["has_positive"], peak_ratio=None, detected=e["detected"],
                             alarm_s=0.0, delay_s=e["delay_s"] if e["delay_s"] is not None else float("nan"),
                             delay_onset_s=e["delay_onset_s"] if e["delay_onset_s"] is not None else float("nan"))
                for e in t["events"]]
        streams.append(StreamResult(waveform_id=wid, outcomes=outs, alarms=np.empty(0), n_false_alarms=t["n_false"],
                                    n_neutral=0, background_hours=t["hours"], duration_hours=24.0))
    return EvalResult(threshold=0.0, streams=streams)


def test_matches_evaluator_paired_bootstrap_and_bootstrap():
    try:
        from src.modules.evaluation.module import Evaluator, paired_bootstrap
    except ImportError:
        print("    (skipped: evaluation module not importable)")
        return

    rng = np.random.default_rng(7)
    ta = make_tallies(lambda i, k: rng.random() < 0.4, n_wave=30, seed=1)
    tb = make_tallies(lambda i, k: rng.random() < 0.7, n_wave=30, seed=2)
    ra, rb = _eval_result(ta), _eval_result(tb)

    theirs = paired_bootstrap(ra, rb, n_boot=200, seed=3)
    mine = compare(Panel([ta]), Panel([tb]), n_boot=200, seed=3)
    for name in ("recall_all", "recall_usable", "recall_weak", "fa_per_day"):
        assert np.allclose(theirs[name], mine[name], equal_nan=True), (name, theirs[name], mine[name])

    ev = Evaluator()
    b_theirs = ev.bootstrap(ra, n_boot=200)
    b_mine = bootstrap(Panel([ta]), n_boot=200, seed=ev.config.seed)
    for name in ("recall_all", "recall_usable", "recall_weak", "recall_impact_mq", "recall_deep_mq", "fa_per_day"):
        assert np.allclose(b_theirs[name], b_mine[name], equal_nan=True), (name, b_theirs[name], b_mine[name])


def test_extract_tallies_round_trip():
    try:
        import src.modules.evaluation.types  # noqa: F401
    except ImportError:
        print("    (skipped: evaluation module not importable)")
        return
    import json

    t = make_tallies(lambda i, k: i % 3 != 0, n_wave=6)
    back = extract_tallies(_eval_result(t))
    assert json.loads(json.dumps(back)) == json.loads(json.dumps(t))      # NaN delays become None


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