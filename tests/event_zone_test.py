from collections import Counter, defaultdict

import numpy as np

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling, POSITIVE, NEGATIVE, IGNORE
from src.modules.event_zones import ZoneEstimator, EventZoneTable


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)

OUTPUT_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/derived/zones/"
)


# ------------------------------------------------------------
# Load + group
# ------------------------------------------------------------

events = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
).run()

validation_report = Validator(DATA_DIR).run(events)

waveforms = WaveformLoader().run(validation_report.valid_events)

grouping = WaveformGrouper().run(waveforms)


# ------------------------------------------------------------
# Zone table
# ------------------------------------------------------------

table = ZoneEstimator().run(
    grouping.waveforms,
    grouping.event_groups,
)

table.save(OUTPUT_DIR)

reloaded = EventZoneTable.load(OUTPUT_DIR)

zones = list(table.zones.values())

print()
print("=" * 60)
print("EVENT ZONE TABLE")
print("=" * 60)
print(f"Events:                  {len(zones)}")
print(f"Waveforms:               {len(table.gaps)}")
print(f"Events with NaN results: {table.n_nan()}  (must be 0)")
print(f"Saved to:                {OUTPUT_DIR}")
print(
    f"Reload identical:        "
    f"{all(abs(reloaded.zones[z.evid].coda_end_s - z.coda_end_s) < 1e-6 for z in zones)}"
)

print()
print("Onset status:", dict(Counter(z.onset_status for z in zones)))

censored = [z for z in zones if z.coda_censored]

print(
    f"Censored codas:          {len(censored)} "
    f"{dict(Counter(z.censor_reason for z in censored))}"
)

in_gap = [z for z in zones if z.gap_in_coda]

print(f"Gap inside the coda:     {len(in_gap)} {[z.evid for z in in_gap]}")


# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

def q(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return (
        f"p25 {np.percentile(a, 25):.0f} / "
        f"median {np.median(a):.0f} / "
        f"p75 {np.percentile(a, 75):.0f}  (n={len(a)})"
    )


print()
print("Visible onset - catalogue (s):", q([z.onset_offset_s for z in zones]))
print("Peak delay (s):               ", q([z.peak_delay_s for z in zones]))
print("Coda end (s), all:            ", q([z.coda_end_s for z in zones]))
print(
    "Coda end (s), not censored:   ",
    q([z.coda_end_s for z in zones if not z.coda_censored]),
)
print("Decay constant tau (s):       ", q([z.tau_s for z in zones]))
print("Peak / background:            ", q([z.peak_ratio for z in zones]))

by_type = defaultdict(list)

for z in zones:
    if not z.coda_censored:
        by_type[z.event_type].append(z.coda_end_s)

print()
print("Coda end (s) by type, not censored")

for t, v in sorted(by_type.items()):
    print(f"  {t:12s} {q(v)}")


# ------------------------------------------------------------
# Per-event table
# ------------------------------------------------------------

print()
print("-" * 60)
print("PER EVENT (first 25)")
print("-" * 60)
print(f"{'evid':10s} {'type':10s} {'onset':>7s} {'peak':>6s} {'coda':>7s} "
      f"{'cens':>5s} {'reason':>10s} {'gap':>4s} {'peak/bg':>8s}")

for z in zones[:25]:

    onset = "nan" if np.isnan(z.onset_offset_s) else f"{z.onset_offset_s:.0f}"

    print(
        f"{z.evid:10s} {z.event_type:10s} {onset:>7s} "
        f"{z.peak_delay_s:6.0f} {z.coda_end_s:7.0f} "
        f"{str(z.coda_censored):>5s} {z.censor_reason:>10s} "
        f"{str(z.gap_in_coda):>4s} {z.peak_ratio:8.1f}"
    )


# ------------------------------------------------------------
# Labeling: per-type placeholders vs zone table
# ------------------------------------------------------------

processed = Preprocessor().run(grouping.waveforms)

windows = Windowing(
    WindowConfig(window_length=60, step_size=10)
).run(processed)

old = Labeling(event_groups=grouping.event_groups)
new = Labeling(event_groups=grouping.event_groups, zones=table)

labeled_old = old.run(windows)
labeled_new = new.run(windows)

c_old = Counter(w.label for w in labeled_old)
c_new = Counter(w.label for w in labeled_new)

total = len(windows)

print()
print("=" * 60)
print("LABELING: per-type coda vs per-event zones")
print("=" * 60)
print(f"{'':22s} {'per-type':>10s} {'per-event':>10s}")
print(f"{'Positive (1)':22s} {c_old[POSITIVE]:10d} {c_new[POSITIVE]:10d}")
print(f"{'Ignore (-1)':22s} {c_old[IGNORE]:10d} {c_new[IGNORE]:10d}")
print(f"{'Negative (0)':22s} {c_old[NEGATIVE]:10d} {c_new[NEGATIVE]:10d}")
print(f"{'Ignore share':22s} {100 * c_old[IGNORE] / total:9.2f}% "
      f"{100 * c_new[IGNORE] / total:9.2f}%")
print(f"Windows ignored because of gaps: {new.gap_ignored}")

print()

Labeling.print_summary(labeled_new, grouping.event_groups)


# ------------------------------------------------------------
# Timelines
# ------------------------------------------------------------

NAMES = {POSITIVE: "POSITIVE", NEGATIVE: "negative", IGNORE: "ignore"}

by_wave = defaultdict(list)

for w in labeled_new:
    by_wave[w.event.evid].append(w)

show = ["evid00030", "evid00002"]

print()
print("=" * 60)
print("LABEL TIMELINE with zones")
print("=" * 60)

for evid in show:

    if evid not in by_wave:
        print(f"{evid}: not a primary waveform id")
        continue

    ws = by_wave[evid]

    print()
    print(f"{evid} | events: "
          f"{[(e.evid, e.event_type, e.time_rel) for e in grouping.event_groups[evid]]}")

    segs = []

    for w in ws:
        if segs and segs[-1]["label"] == w.label:
            segs[-1]["end"] = w.end_time
            segs[-1]["n"] += 1
        else:
            segs.append({"label": w.label, "start": w.start_time,
                         "end": w.end_time, "n": 1})

    for s in segs:
        print(f"  {NAMES[s['label']]:9s} {s['start']:10.0f} - "
              f"{s['end']:10.0f} s | {s['n']} windows")