import os
from collections import Counter

import numpy as np

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.event_zones import ZoneEstimator, EventZoneTable
from src.modules.labeling import Labeling, LabelConfig, POSITIVE, IGNORE
from src.modules.folds import FoldBuilder, FoldSet


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)

ZONES_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/derived/zones/"
)

FOLDS_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/derived/folds/folds.csv"
)

N_FOLDS = 5
SEED = 42


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------

events = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
).run()

validation_report = Validator(DATA_DIR).run(events)

waveforms = WaveformLoader().run(validation_report.valid_events)

grouping = WaveformGrouper().run(waveforms)

if os.path.exists(os.path.join(ZONES_DIR, "event_zones.csv")):
    table = EventZoneTable.load(ZONES_DIR)
else:
    table = ZoneEstimator().run(grouping.waveforms, grouping.event_groups)
    table.save(ZONES_DIR)

processed = Preprocessor().run(grouping.waveforms)

windows = Windowing(
    WindowConfig(window_length=60, step_size=10)
).run(processed)

labeling = Labeling(
    LabelConfig(),
    event_groups=grouping.event_groups,
    zones=table,
)

labeled = labeling.run(windows)


# ------------------------------------------------------------
# Folds
# ------------------------------------------------------------

fold_set = FoldBuilder(n_folds=N_FOLDS, random_seed=SEED).run(
    grouping.event_groups,
    labeling,
)

fold_set.save(FOLDS_PATH)

reloaded = FoldSet.load(FOLDS_PATH)

print()
print("=" * 60)
print("FOLDS")
print("=" * 60)
print(f"Folds: {fold_set.n_folds} | seed: {fold_set.random_seed}")
print(f"Saved to: {FOLDS_PATH}")
print(f"Reload identical: {reloaded.assignment == fold_set.assignment}")

print()
print("Strata:", dict(Counter(fold_set.strata.values())))

print()
print("Waveforms per stratum and TEST fold")
print(f"{'stratum':12s} " + " ".join(f"{'f' + str(k):>4s}" for k in range(N_FOLDS)))

for name in sorted(set(fold_set.strata.values())):
    per = Counter(
        fold_set.assignment[w]
        for w, s in fold_set.strata.items()
        if s == name
    )
    print(f"{name:12s} " + " ".join(f"{per.get(k, 0):4d}" for k in range(N_FOLDS)))


# ------------------------------------------------------------
# Per fold, per split
# ------------------------------------------------------------

positives = Counter(
    w.event.evid for w in labeled if w.label == POSITIVE
)

with_positives = {
    e.evid
    for evs in grouping.event_groups.values()
    for e in evs
    if labeling.positive_rule(e) is not None
}

print()
print("=" * 60)
print("PER FOLD")
print("=" * 60)

for f in fold_set.folds:

    print()
    print(f"Fold {f.index}")

    for name, ids in (
        ("train", f.train),
        ("validation", f.validation),
        ("test", f.test),
    ):

        evs = grouping.events_of(ids)
        types = Counter(e.event_type for e in evs)
        usable = sum(e.evid in with_positives for e in evs)
        pos = sum(positives[w] for w in ids)

        print(
            f"  {name:10s} waveforms={len(ids):3d} events={len(evs):3d} "
            f"(with positives {usable:3d}) positives={pos:4d} "
            f"{dict(sorted(types.items()))}"
        )


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

test_counts = Counter()
leak = False

for f in fold_set.folds:

    tr, va, te = set(f.train), set(f.validation), set(f.test)

    leak |= bool((tr & va) or (tr & te) or (va & te))

    # no catalogue event may sit in two splits of the same fold
    sets = [set(grouping.evids_of(ids)) for ids in (f.train, f.validation, f.test)]
    leak |= bool((sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]))

    test_counts.update(grouping.evids_of(f.test))

all_evids = {e.evid for evs in grouping.event_groups.values() for e in evs}

print()
print("=" * 60)
print("CHECKS")
print("=" * 60)
print(f"Overlap between splits inside a fold: {leak}  (must be False)")
print(f"Every catalogue event in exactly one test fold: "
      f"{set(test_counts) == all_evids and set(test_counts.values()) == {1}}")

pos_in_test = [sum(positives[w] for w in f.test) for f in fold_set.folds]
print(f"Positive windows in the TEST folds: {pos_in_test} "
      f"(total {sum(pos_in_test)}, all positives {sum(positives.values())})")


# ------------------------------------------------------------
# Window-level split of one fold (DatasetSplit format)
# ------------------------------------------------------------

split = fold_set.split(labeled, 0)

print()
print("Fold 0 as DatasetSplit")

for name, ws in (("train", split.train), ("validation", split.validation),
                 ("test", split.test)):
    c = Counter(w.label for w in ws)
    print(f"  {name:10s} windows={len(ws):7d} pos={c[POSITIVE]:4d} "
          f"ign={c[IGNORE]:6d} neg={c[0]:7d}")