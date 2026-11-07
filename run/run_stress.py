"""
Stress tests of FROZEN detectors (Apollo 12, nested CV): perturb the TEST streams only.

    python -m tests.run_stress --models cfcE:1k gruE:1k rfc:medium cnn:1k amp:rms --seeds 1 --perturb clean gain3 \
        --rec-epochs 150 --rec-min-epochs 60 --rec-patience 40 --rec-tag long --out results/stress_smoke.json
    python -m tests.run_stress                                  # default models, all perturbations, 5 seeds

Per fold the detector is fitted on the training windows and the thresholds are chosen on the CLEAN validation
streams (as in cross_validate_budgets); the same frozen detector + thresholds are then applied to the test streams
after each perturbation. Perturbations (names): clean | gain0.3 gain3 gain10 | noise0.5 noise1 noise2 (white noise,
highpassed like the data, x the stream's background std) | mnoise0.5 mnoise1 mnoise2 (same level, but with the
spectrum of the stream's own background) | gaps2 gaps10 (percent of the stream replaced by linearly
interpolated stretches of 20-120 s) | jitter0.25 (timestamp jitter in sample intervals) | lp1.3 lp0.8 (causal lowpass, Hz).
N0 only: N2 / N3 use scale tracks of the clean streams, which would not see the perturbation.

Output records use the same format as run_dnn_features.py with config = "<model config>@<perturbation>", so
    python -m tests.analyze_results results/stress.json --compare cfcE-1k-N0-long@gain3 rfc-medium-N0@gain3
works. --oracle adds records "<config>@<perturbation>~oracle" (threshold re-chosen on the perturbed test data:
an optimistic diagnostic, NOT a result).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.evaluation.module import build_series, pool_results
from src.modules.normalization import NormConfig, Normalizer
from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.reporting import extract_tallies
from src.modules.stress.module import DEFAULT_PERTURBATIONS, make_perturbation, stress_cross_validate
from run import run_dnn_features as R

RECURRENT = ("gru", "cfc", "gruE", "cfcE", "cfc0E")
DEFAULT_MODELS = ["cfcE:1k", "gruE:1k", "rfc:medium", "cnn:1k", "amp:rms"]


def config_name(kind, size, tag):
    return f"{kind}-{size}-N0" + (f"-{tag}" if kind in RECURRENT and tag else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="kind:size, e.g. cfcE:1k rfc:medium amp:rms")
    ap.add_argument("--perturb", nargs="+", default=list(DEFAULT_PERTURBATIONS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.25, 2.0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rec-epochs", type=int, default=None)
    ap.add_argument("--rec-patience", type=int, default=None)
    ap.add_argument("--rec-min-epochs", type=int, default=None)
    ap.add_argument("--rec-lr", type=float, default=1e-3)
    ap.add_argument("--rec-tag", default="")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--catalog", default=R.CATALOG_PATH)
    ap.add_argument("--data-dir", default=R.DATA_DIR)
    ap.add_argument("--derived-dir", default=R.DERIVED_DIR)
    ap.add_argument("--out", default="results/stress.json")
    a = ap.parse_args()

    R.REC.update(epochs=a.rec_epochs, patience=a.rec_patience, min_epochs=a.rec_min_epochs, lr=a.rec_lr)

    perts = [make_perturbation(n) for n in a.perturb]

    bundle = DataPipeline(PipelineConfig(a.catalog, a.data_dir, a.derived_dir)).build()
    fs = bundle.eval_stream(sorted(bundle._processed_by_id)[0]).sampling_rate
    evaluator = R.make_evaluator()

    records = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for spec in a.models:

        kind, size = spec.split(":")
        base = config_name(kind, size, a.rec_tag)
        n_seeds = 1 if kind == "amp" else a.seeds
        t0 = time.time()

        for seed in range(n_seeds):

            name = f"{base}-s{seed}"
            det = R.build_detector(kind, size, "N0", name, Normalizer(NormConfig("N0")), fs, seed, a.epochs)

            run = stress_cross_validate(bundle, det, evaluator, a.budgets, perts, build_series, pool_results,
                                        oracle=a.oracle, verbose=False)

            for tag, results in (("", run.pooled), ("~oracle", run.oracle)):
                for (pname, budget), pooled in results.items():
                    records.append(dict(
                        config=f"{base}@{pname}{tag}", budget=budget, seed=seed,
                        recall=float(pooled.recall("all")), fa_per_day=float(pooled.fa_per_day),
                        thresholds=[float(t[budget]) for t in run.thresholds] if not tag else None,
                        outcomes=extract_tallies(pooled),
                    ))

            clean = [r for r in records if r["config"] == f"{base}@clean" and r["seed"] == seed]
            if clean:
                print(f"  {name}: clean " + ", ".join(f"{r['budget']:g}/day recall {r['recall']:.2f} FA {r['fa_per_day']:.2f}"
                                                      for r in clean))

            out.write_text(json.dumps(dict(records=records), indent=1))      # written after every seed

        print(f"{base} done in {time.time() - t0:.0f} s")

    # summary: seed-mean recall / FA per day under every perturbation
    print()
    for budget in a.budgets:
        print(f"budget {budget:g} false alarms/day (frozen thresholds): recall / FA per day, mean over seeds")
        names = [p.name for p in perts]
        print(f"  {'config':22s}" + "".join(f"{n:>13s}" for n in names))
        for spec in a.models:
            kind, size = spec.split(":")
            base = config_name(kind, size, a.rec_tag)
            cells = []
            for n in names:
                rs = [r for r in records if r["config"] == f"{base}@{n}" and r["budget"] == budget]
                cells.append(f"{np.mean([r['recall'] for r in rs]):.2f}/{np.mean([r['fa_per_day'] for r in rs]):.2f}" if rs else "-")
            print(f"  {base:22s}" + "".join(f"{c:>13s}" for c in cells))
        print()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()