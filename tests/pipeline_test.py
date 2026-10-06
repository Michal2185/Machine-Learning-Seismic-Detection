import time
from collections import Counter, defaultdict

import numpy as np

from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.labeling import POSITIVE, NEGATIVE, IGNORE


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


# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

start = time.time()

bundle = DataPipeline(config).build()

print()
print(f"Build time: {time.time() - start:.0f} s")
print()

bundle.summary()


# ------------------------------------------------------------
# Folds as the models will use them
# ------------------------------------------------------------

print()
print("=" * 60)
print("FOLD SPLITS")
print("=" * 60)

for k in range(bundle.folds.n_folds):

    split = bundle.fold_split(k)

    row = []

    for name, ws in (("train", split.train),
                     ("val", split.validation),
                     ("test", split.test)):
        c = Counter(w.label for w in ws)
        row.append(f"{name}: {len(ws):7d} win, {c[POSITIVE]:3d} pos")

    print(f"Fold {k} | " + " | ".join(row))


# ------------------------------------------------------------
# Evaluation streams
# ------------------------------------------------------------

split0 = bundle.fold_split(0)

streams = bundle.eval_streams(split0.test_events)

print()
print("=" * 60)
print("EVALUATION STREAMS (fold 0, test waveforms)")
print("=" * 60)

for st in streams:

    hours = len(st.data) / st.sampling_rate / 3600

    print()
    print(f"{st.waveform_id} | {hours:.2f} h | fs {st.sampling_rate:.3f} Hz | "
          f"dtype {st.data.dtype} | interpolated gaps: {len(st.gaps)}")

    for t in st.events:

        onset = "-" if t.onset_s is None else f"{t.onset_s:.0f}"
        pr = "-" if t.peak_ratio is None else f"{t.peak_ratio:.1f}"

        print(f"   {t.evid} {t.event_type:10s} arrival {t.arrival_s:8.0f} | "
              f"detect [{t.detect[0]:8.0f}, {t.detect[1]:8.0f}] | "
              f"ignore [{t.ignore[0]:8.0f}, {t.ignore[1]:8.0f}] | "
              f"onset {onset:>7s} | positives {t.has_positive} | peak/bg {pr}")


# ------------------------------------------------------------
# Consistency: window labels vs evaluation intervals (all waveforms)
# ------------------------------------------------------------

by_waveform = defaultdict(list)

for w in bundle.windows:
    by_waveform[w.event.evid].append(w)

violations = Counter()
n_positive = 0

for wid, ws in by_waveform.items():

    st = bundle.eval_stream(wid)

    for w in ws:

        in_zone = any(
            w.end_time > t.ignore[0] and w.start_time < t.ignore[1]
            for t in st.events
        )

        gap_share = sum(
            max(0.0, min(w.end_time, g[1]) - max(w.start_time, g[0]))
            for g in st.gaps
        ) / (w.end_time - w.start_time)

        if w.label == NEGATIVE and in_zone:
            violations["negative inside an ignore interval"] += 1

        if w.label == IGNORE and not in_zone and gap_share < 0.5:
            violations["ignore window explained by neither zone nor gap"] += 1

        if w.label == POSITIVE:
            n_positive += 1
            if not any(t.detect[0] <= w.end_time <= t.detect[1]
                       for t in st.events):
                violations["positive window ends outside every detect interval"] += 1

print()
print("=" * 60)
print("CONSISTENCY CHECKS")
print("=" * 60)
print(f"Positive windows checked: {n_positive}")
print(f"Violations: {dict(violations) if violations else 'none'}  (must be none)")

events = [
    e for evs in bundle.grouping.event_groups.values() for e in evs
]

targets = [bundle.target(e) for e in events]

print()
print(f"Events: {len(targets)}")
print(f"  with usable onset (has positives): "
      f"{sum(t.has_positive for t in targets)}")
print(f"  weak (no positives):               "
      f"{sum(not t.has_positive for t in targets)}")
print("Weak events by type:",
      dict(Counter(t.event_type for t in targets if not t.has_positive)))