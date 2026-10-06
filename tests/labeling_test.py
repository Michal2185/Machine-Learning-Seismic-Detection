import os
from collections import Counter, defaultdict

import numpy as np

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.event_zones import ZoneEstimator, EventZoneTable
from src.modules.labeling import (
    Labeling,
    LabelConfig,
    POSITIVE,
    NEGATIVE,
    IGNORE,
)


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

config = LabelConfig()

variants = {
    "catalogue, per-type coda": Labeling(
        event_groups=grouping.event_groups
    ),
    "per-event zones, catalogue": Labeling(
        LabelConfig(anchor_to_onset=False),
        event_groups=grouping.event_groups,
        zones=table,
    ),
    "per-event zones, onset-anchored": Labeling(
        config,
        event_groups=grouping.event_groups,
        zones=table,
    ),
}

results = {name: lab.run(windows) for name, lab in variants.items()}

labeling = variants["per-event zones, onset-anchored"]
labeled = results["per-event zones, onset-anchored"]

total = len(windows)


# ------------------------------------------------------------
# Comparison of the three definitions
# ------------------------------------------------------------

print()
print("=" * 60)
print("LABELING VARIANTS")
print("=" * 60)
print(f"{'':34s} {'positive':>9s} {'ignore':>9s} {'negative':>9s} {'ign %':>6s}")

for name, lw in results.items():
    c = Counter(w.label for w in lw)
    print(f"{name:34s} {c[POSITIVE]:9d} {c[IGNORE]:9d} {c[NEGATIVE]:9d} "
          f"{100 * c[IGNORE] / total:6.2f}")

print()
print(f"Windows ignored because of gaps: {labeling.gap_ignored}")
print(f"Min signal in a positive window: {config.min_signal_s} s")
print(f"Usable visible-onset offsets:    {config.onset_bounds} s")
print(f"Detection tolerance (evaluation): "
      f"[-{config.detect_before_s:.0f}, +{config.detect_after_s:.0f}] s")


# ------------------------------------------------------------
# Summary of the anchored labels
# ------------------------------------------------------------

print()
s = Labeling.print_summary(labeled, grouping.event_groups, labeling)


# ------------------------------------------------------------
# Events without positives must be exactly the unusable-onset events
# ------------------------------------------------------------

unusable = {
    e.evid
    for evs in grouping.event_groups.values()
    for e in evs
    if labeling.positive_rule(e) is None
}

without = set(s.events_without_positive)

print()
print(f"Events with unusable visible onset: {len(unusable)}")
print(f"Events without a positive window:   {len(without)}")
print(f"Identical sets: {unusable == without}  (must be True)")

reasons = Counter()

for evid in unusable:
    z = table.zones[evid]
    if z.onset_status != "ok":
        reasons[z.onset_status] += 1
    else:
        reasons["offset outside bounds"] += 1

print(f"Reasons: {dict(reasons)}")

print()
print("Unusable events by type:",
      dict(Counter(table.zones[e].event_type for e in unusable)))


# ------------------------------------------------------------
# Positives per event
# ------------------------------------------------------------

per_event = Counter()

for w in labeled:
    if w.label != POSITIVE:
        continue
    for e in grouping.event_groups.get(w.event.evid, [w.event]):
        ref, min_sig, span = labeling.positive_rule(e) or (None, 0, 0)
        if ref is not None and w.start_time <= ref + span and w.end_time >= ref + min_sig:
            per_event[e.evid] += 1

vals = np.array(list(per_event.values()))

print()
print(f"Positive windows per event (events with positives: {len(vals)}): "
      f"min {vals.min()} / median {np.median(vals):.0f} / max {vals.max()}")


# ------------------------------------------------------------
# Timelines
# ------------------------------------------------------------

NAMES = {POSITIVE: "POSITIVE", NEGATIVE: "negative", IGNORE: "ignore"}

by_wave = defaultdict(list)

for w in labeled:
    by_wave[w.event.evid].append(w)


def timeline(evid):

    ws = by_wave[evid]

    for e in grouping.event_groups[evid]:
        z = table.zones[e.evid]
        print(f"{e.evid} | {e.event_type} | catalogue {e.time_rel:.0f} s | "
              f"onset {z.onset_status} "
              f"{'' if np.isnan(z.onset_offset_s) else f'{z.onset_offset_s:+.0f} s'} | "
              f"coda end {z.coda_end_s:.0f} s")

    segs = []

    for w in ws:
        if segs and segs[-1]["label"] == w.label:
            segs[-1]["end"] = w.end_time
            segs[-1]["n"] += 1
        else:
            segs.append({"label": w.label, "start": w.start_time,
                         "end": w.end_time, "n": 1})

    for sg in segs:
        if sg["label"] == NEGATIVE and sg["n"] > 200:
            print(f"  {NAMES[sg['label']]:9s} {sg['start']:9.0f} - "
                  f"{sg['end']:9.0f} s | {sg['n']} windows (background)")
        else:
            print(f"  {NAMES[sg['label']]:9s} {sg['start']:9.0f} - "
                  f"{sg['end']:9.0f} s | {sg['n']} windows")


print()
print("=" * 60)
print("LABEL TIMELINES (onset-anchored)")
print("=" * 60)

for evid in ["evid00002", "evid00009", "evid00003"]:
    if evid in by_wave:
        print()
        timeline(evid)


print()
print("Positive windows of evid00009 (signal inside = end - visible onset):")

z9 = table.zones.get("evid00009")

if z9 is not None and "evid00009" in by_wave:
    onset9 = z9.arrival_s + z9.onset_offset_s
    for w in by_wave["evid00009"]:
        if w.label == POSITIVE:
            print(f"  {w.start_time:9.2f} - {w.end_time:9.2f} s | "
                  f"signal inside: {w.end_time - onset9:5.1f} s")