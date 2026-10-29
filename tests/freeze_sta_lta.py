"""
Frozen STA/LTA with two operating points (strict / lenient).

1. Selection: one STA/LTA parameter set and one threshold per operating
   point are chosen from waveforms only (no test waveform is used):
     * "all"   : all 70 Apollo 12 waveforms  -> the detector that is frozen
                 and saved for later zero-shot tests
     * "foldK" : training + validation waveforms of fold K -> used only to
                 estimate honestly what the procedure achieves on waveforms
                 it has not seen (nested cross-validation)
2. Report: nested-CV result per operating point (pooled test folds) and the
   in-sample result of the final frozen detector.

Downlink view: every alarm would transmit a snippet of the recording. A
false alarm wastes bandwidth, a missed event loses science, and the second
is worse. Hence two points: LENIENT (science first) and STRICT
(bandwidth first).
"""

import time

import numpy as np

from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.detectors import evaluate_frozen, freeze_sta_lta_sets
from src.modules.evaluation import EvalConfig, Evaluator, pool_results


config = PipelineConfig(
    catalog_path=(
        "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
    ),
    data_dir=(
        "data/lunar/data/S12_GradeA/"
    ),
    derived_dir=(
        "data/lunar/derived/"
    ),
)

FROZEN_PATH = config.derived_dir + "baselines/sta_lta_frozen.json"

BUDGETS = {"strict": 0.25, "lenient": 2.0}      # false alarms per 24 scored hours
SNIPPETS = (300.0, 900.0)                       # seconds sent per alarm (illustrative)

bundle = DataPipeline(config).build()

evaluator = Evaluator(
    EvalConfig(refractory_s=300.0, n_thresholds=40, n_bootstrap=1000)
)

all_ids = sorted(bundle.grouping.event_groups)
streams_by_id = {s.waveform_id: s for s in bundle.eval_streams(all_ids)}

sets = {"all": all_ids}

for k in range(bundle.folds.n_folds):
    split = bundle.fold_split(k)
    sets[f"fold{k}"] = sorted(set(split.train_events) | set(split.validation_events))


# ------------------------------------------------------------
# Why these budgets: the catalogue event rate
# ------------------------------------------------------------

n_events = sum(len(v) for v in bundle.grouping.event_groups.values())
total_days = sum(len(s.data) / s.sampling_rate for s in streams_by_id.values()) / 86400

rate = n_events / total_days

print()
print("=" * 78)
print("EVENT RATE AND WHAT A BUDGET MEANS")
print("=" * 78)
print(f"Catalogue events: {n_events} in {total_days:.1f} days of data "
      f"-> {rate:.2f} events per day (Grade A only, so a LOWER bound)")
print(f"{'recall':>8s} " + " ".join(
    f"{'FA ' + str(b) + '/day':>16s}" for b in BUDGETS.values()))

for r in (0.6, 0.8, 0.9):
    row = []
    for b in BUDGETS.values():
        detected = r * rate
        row.append(f"{detected / (detected + b):16.2f}")
    print(f"{r:8.1f} " + " ".join(row) + "   <- share of alarms that are real events")


# ------------------------------------------------------------
# Selection
# ------------------------------------------------------------

start = time.time()

print()
print(f"Selecting on {len(sets)} sets (all + {bundle.folds.n_folds} folds), "
      f"84 candidates each ...")

specs = freeze_sta_lta_sets(
    bundle.windows, streams_by_id, evaluator, BUDGETS, sets
)

print(f"({time.time() - start:.0f} s)")

specs["all"].save(FROZEN_PATH)

print()
print("=" * 78)
print("CHOSEN DETECTORS")
print("=" * 78)

for name, spec in specs.items():
    print(f"{name:6s} | {spec.describe()}")

print()
print(f"Final frozen detector saved to: {FROZEN_PATH}")


# ------------------------------------------------------------
# Reporting helper
# ------------------------------------------------------------

def report(title, results):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    for point, res in results.items():

        iv = evaluator.bootstrap(res)
        r, r_lo, r_hi = iv["recall_all"]
        fa, fa_lo, fa_hi = iv["fa_per_day"]

        print()
        print(f"[{point}] budget {BUDGETS[point]:g} FA/day")
        print(f"  recall all    {r:.2f} [{r_lo:.2f}, {r_hi:.2f}]   "
              f"usable {res.recall('usable'):.2f}   weak {res.recall('weak'):.2f}   "
              f"missed events: {res.n_events() - sum(o.detected for o in res.outcomes)}"
              f" of {res.n_events()}")
        print(f"  false alarms  {fa:.2f}/day [{fa_lo:.2f}, {fa_hi:.2f}]   "
              f"all alarms {res.alarms_per_day:.2f}/day (repeats inside codas included)   "
              f"real share of transmissions: {res.episode_precision:.2f}")
        print("  downlink share: " + ", ".join(
            f"{100 * res.downlink_share(s):.2f}% at {s:.0f} s/alarm" for s in SNIPPETS))

        d = res.delays("all", "onset")
        if len(d):
            print(f"  delay vs visible onset: median {np.median(d):.0f} s "
                  f"(p25 {np.percentile(d, 25):.0f}, p75 {np.percentile(d, 75):.0f})")


# ------------------------------------------------------------
# Nested cross-validation (honest estimate for unseen waveforms)
# ------------------------------------------------------------

per_point = {p: [] for p in BUDGETS}

for k in range(bundle.folds.n_folds):

    split = bundle.fold_split(k)

    res = evaluate_frozen(
        specs[f"fold{k}"],
        split.test,
        bundle.eval_streams(split.test_events),
        evaluator,
    )

    for p in BUDGETS:
        per_point[p].append(res[p])

nested = {p: pool_results(v) for p, v in per_point.items()}

report("NESTED CROSS-VALIDATION: pooled test folds, selected on train+validation only",
       nested)


# ------------------------------------------------------------
# Final frozen detector on all waveforms (in-sample)
# ------------------------------------------------------------

final = evaluate_frozen(
    specs["all"], bundle.windows, list(streams_by_id.values()), evaluator
)

report("FINAL FROZEN DETECTOR on all 70 waveforms (IN-SAMPLE, optimistic)", final)


# ------------------------------------------------------------
# By event type, nested CV
# ------------------------------------------------------------

print()
print("=" * 78)
print("NESTED CV BY EVENT TYPE (events detected / events)")
print("=" * 78)

for p, res in nested.items():
    parts = []
    for t in sorted({o.event_type for o in res.outcomes}):
        d = sum(o.detected for o in res.outcomes if o.event_type == t)
        parts.append(f"{t} {d}/{res.n_events(t)}")
    print(f"{p:8s} " + " | ".join(parts))