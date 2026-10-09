"""
Sanity check of the evaluator on the REAL streams with two detectors whose
result is known in advance:

  oracle : score 1 on windows labeled POSITIVE, 0 elsewhere
           -> every event with positives detected, none of the weak events,
              no false alarms
  random : uniform random scores -> chance level
"""

import numpy as np

from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.labeling import POSITIVE
from src.modules.evaluation import (
    EvalConfig,
    Evaluator,
    build_series,
    pool_results,
)


config = PipelineConfig(
    catalog_path=(
        "F:/Git/Machine-Learning-Seismic-Detection/"
        "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
    ),
    data_dir=(
        "F:/Git/Machine-Learning-Seismic-Detection/"
        "data/lunar/data/S12_GradeA/"
    ),
    derived_dir=(
        "F:/Git/Machine-Learning-Seismic-Detection/"
        "data/lunar/derived/"
    ),
)

bundle = DataPipeline(config).build()

evaluator = Evaluator(EvalConfig(refractory_s=300.0, max_fa_per_day=1.0))


def cross_validate(score_fn):
    """validation -> threshold -> test, for every fold, results pooled."""

    results = []
    thresholds = []

    for k in range(bundle.folds.n_folds):

        split = bundle.fold_split(k)

        val_streams = bundle.eval_streams(split.validation_events)
        test_streams = bundle.eval_streams(split.test_events)

        val_series = build_series(split.validation, score_fn(split.validation))
        test_series = build_series(split.test, score_fn(split.test))

        curve = evaluator.curve(val_streams, val_series)
        threshold = evaluator.select_threshold(curve)

        thresholds.append(threshold)
        results.append(evaluator.evaluate(test_streams, test_series, threshold))

    return pool_results(results), thresholds


# ------------------------------------------------------------
# Oracle
# ------------------------------------------------------------

def oracle(windows):
    return np.array([1.0 if w.label == POSITIVE else 0.0 for w in windows])


pooled, thresholds = cross_validate(oracle)

print()
print("#" * 60)
print("ORACLE (labels used as scores)")
print("#" * 60)
print(f"Selected thresholds per fold: {thresholds}")

Evaluator.print_result(pooled, evaluator.bootstrap(pooled))

n_usable = pooled.n_events("usable")
n_all = pooled.n_events("all")

print()
print("Expected: recall(usable) = 1.00, recall(weak) = 0.00, "
      f"recall(all) = {n_usable}/{n_all} = {n_usable / n_all:.2f}, "
      "false alarms = 0")
print(f"Checks: usable={pooled.recall('usable') == 1.0} | "
      f"weak={pooled.recall('weak') == 0.0} | "
      f"false alarms={pooled.n_false_alarms == 0}")


# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

rng = np.random.default_rng(0)


def random_scores(windows):
    return rng.random(len(windows))


pooled_r, thresholds_r = cross_validate(random_scores)

print()
print("#" * 60)
print("RANDOM SCORES (chance level)")
print("#" * 60)
print(f"Selected thresholds per fold: {[round(t, 4) for t in thresholds_r]}")

Evaluator.print_result(pooled_r, evaluator.bootstrap(pooled_r))

print()
print("Expected: false alarms near the 1 per day budget (or fewer), "
      "recall far below the oracle (a random alarm rarely falls into a "
      "720 s detect interval)")