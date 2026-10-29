"""Smoke test of tests/analyze_results.py on synthetic JSON files. Run: python tests/test_analyze.py"""

import contextlib
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats_test import make_tallies

import analyze_results as ar


def write_json(path, configs, seeds=2):
    rng = np.random.default_rng(0)
    recs = []
    for config, p in configs.items():
        for budget in (0.25, 2.0):
            for s in range(seeds if not config.startswith("amp") else 1):
                recs.append(dict(config=config, budget=budget, seed=s, recall=0.0, fa_per_day=0.0, test_ap=0.0,
                                 thresholds=[],
                                 outcomes=make_tallies(lambda i, k: rng.random() < p, n_wave=30, seed=1)))
    recs.append(dict(config="old-run", budget=0.25, seed=0, recall=0.5, fa_per_day=0.1, test_ap=0.0, thresholds=[]))
    Path(path).write_text(json.dumps(dict(records=recs)))


def test_end_to_end_on_synthetic_files():
    with tempfile.TemporaryDirectory() as d:
        write_json(f"{d}/a.json", {"amp-rms-N0": 0.5, "rfc-medium-N0": 0.9, "rfc-medium-N2": 0.4})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ar.main([f"{d}/*.json", "--n-boot", "100", "--out", f"{d}/res/analysis",
                     "--compare", "rfc-medium-N2", "amp-rms-N0"])
        text = buf.getvalue()
        assert "1 record(s) without 'outcomes' skipped" in text
        assert "recall all [95%]" in text and "rfc-medium-N0" in text
        assert "Every config against the reference amp-rms-N0" in text
        assert "N0 against N2" in text and "Requested pairs" in text
        assert "STA/LTA (nested CV" in text
        rows = (Path(d) / "res" / "analysis.csv").read_text().splitlines()
        assert rows[0].startswith("level,budget,config,other,metric") and len(rows) > 50
        assert (Path(d) / "res" / "analysis_recall.png").exists()
        # the clearly better config is flagged as a significant improvement over the reference
        line = [l for l in text.splitlines() if "amp-rms-N0" in l and "-> rfc-medium-N0" in l][0]
        assert "*" in line


def test_no_outcomes_exits():
    with tempfile.TemporaryDirectory() as d:
        Path(f"{d}/o.json").write_text(json.dumps(dict(records=[dict(config="x", budget=0.25, seed=0)])))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ar.main([f"{d}/o.json", "--no-plot"])
            raise AssertionError("expected SystemExit")
        except SystemExit:
            pass


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