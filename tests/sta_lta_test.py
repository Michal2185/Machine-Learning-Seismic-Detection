"""
STA/LTA baselines through the full evaluation chain on the real data.

Per fold the STA/LTA parameters and the threshold are chosen on the
VALIDATION waveforms only (budget: 1 false alarm per 24 scored hours) and
applied once to the TEST waveforms. Test results of the 5 folds are pooled.
Differences between runs are reported with a paired bootstrap over waveforms.
"""

import time

import numpy as np

from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.detectors import (
    StaLtaDetector,
    sta_lta_grid,
    window_sta_lta_grid,
)
from src.modules.evaluation import (
    EvalConfig,
    Evaluator,
    cross_validate,
    paired_bootstrap,
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

evaluator = Evaluator(
    EvalConfig(refractory_s=300.0, max_fa_per_day=1.0, n_thresholds=40)
)

runs = {}


def run(key, name, candidates):

    start = time.time()

    res = cross_validate(bundle, candidates, evaluator, name=name)

    res.print_summary(evaluator)

    print(f"\n({time.time() - start:.0f} s)")

    runs[key] = res

    return res


# ------------------------------------------------------------
# 1. Continuous STA/LTA: first grid (edge of the grid was hit) and wide grid
# ------------------------------------------------------------

run(
    "old",
    "STA/LTA continuous, FIRST grid (STA<=120 s, LTA<=7200 s)",
    sta_lta_grid(sta=(10.0, 30.0, 60.0, 120.0), lta=(900.0, 3600.0, 7200.0)),
)

run("wide", "STA/LTA continuous, WIDE grid (STA<=480 s, LTA<=28800 s)",
    sta_lta_grid())


# ------------------------------------------------------------
# 2. Delayed LTA (LTA ends before the STA window): helps in long codas
# ------------------------------------------------------------

run(
    "delayed",
    "STA/LTA continuous, delayed LTA",
    sta_lta_grid(
        sta=(60.0, 120.0, 240.0),
        lta=(7200.0, 14400.0, 28800.0),
        features=("energy",),
        delay_lta=True,
    ),
)


# ------------------------------------------------------------
# 3. Window-restricted STA/LTA
# ------------------------------------------------------------

run("window", "STA/LTA inside the 60 s window", window_sta_lta_grid())


# ------------------------------------------------------------
# 4. Interpolated gaps: masked versus not masked (same parameters)
# ------------------------------------------------------------

run("masked", "STA/LTA 120 s / 7200 s energy, gaps masked",
    [StaLtaDetector(120.0, 7200.0, "energy", mask_gaps=True)])

run("unmasked", "STA/LTA 120 s / 7200 s energy, gaps NOT masked",
    [StaLtaDetector(120.0, 7200.0, "energy", mask_gaps=False)])


# ------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------

print()
print("=" * 82)
print("COMPARISON (pooled test folds, threshold from validation, budget 1 FA/day)")
print("=" * 82)
print(f"{'run':56s} {'all':>5s} {'usable':>7s} {'weak':>5s} {'FA/day':>7s}")

for key, res in runs.items():

    p = res.pooled

    print(f"{res.name[:56]:56s} {p.recall('all'):5.2f} "
          f"{p.recall('usable'):7.2f} {p.recall('weak'):5.2f} "
          f"{p.fa_per_day:7.2f}")


# ------------------------------------------------------------
# Paired differences (b - a), 95 % interval over waveforms
# ------------------------------------------------------------

def compare(a_key, b_key):

    d = paired_bootstrap(runs[a_key].pooled, runs[b_key].pooled)

    print()
    print(f"{runs[b_key].name}\n  minus\n{runs[a_key].name}")

    for name in ("recall_all", "recall_usable", "recall_weak", "fa_per_day"):
        point, lo, hi, p_better = d[name]
        print(f"  {name:14s} {point:+7.2f}  [{lo:+.2f}, {hi:+.2f}]  "
              f"P(b better) = {p_better:.2f}")


print()
print("=" * 82)
print("PAIRED DIFFERENCES")
print("=" * 82)

compare("old", "wide")
compare("wide", "delayed")
compare("unmasked", "masked")
compare("window", "wide")