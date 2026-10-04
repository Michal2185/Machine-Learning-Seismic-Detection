from collections import Counter

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling
from src.modules.dataset_builder import DatasetBuilder


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
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


print()
print("=" * 60)
print("WAVEFORM GROUPING")
print("=" * 60)
print(f"Catalogue records:    {len(waveforms)}")
print(f"Distinct waveforms:   {len(grouping.waveforms)}")

shared = {
    primary: evs
    for primary, evs in grouping.event_groups.items()
    if len(evs) > 1
}

print(f"Shared waveforms:     {len(shared)}")
print()

for primary, evs in shared.items():

    print(f"waveform of {primary}:")

    for e in evs:
        print(
            f"  {e.evid} | {e.event_type:10s} | "
            f"{e.time_abs} | arrival={e.time_rel:.0f} s"
        )

    gap = evs[-1].time_rel - evs[0].time_rel

    print(f"  gap between first and last onset: {gap:.0f} s")
    print()


# ------------------------------------------------------------
# Pipeline on distinct waveforms only
# ------------------------------------------------------------

processed = Preprocessor().run(grouping.waveforms)

windows = Windowing(
    WindowConfig(window_length=60, step_size=10)
).run(processed)

labeling = Labeling(event_groups=grouping.event_groups)

labeled = labeling.run(windows)

print("=" * 60)
print("WINDOWS AFTER GROUPING")
print("=" * 60)
print(f"Total windows: {len(labeled)}   (was 646549 with duplicates)")
print()

Labeling.print_summary(labeled, grouping.event_groups)


# ------------------------------------------------------------
# Split by waveform
# ------------------------------------------------------------

dataset = DatasetBuilder(
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
).run(labeled)

print()
print("=" * 60)
print("SPLIT (by distinct waveform)")
print("=" * 60)

for name, primaries, ws in [
    ("TRAIN", dataset.train_events, dataset.train),
    ("VALIDATION", dataset.validation_events, dataset.validation),
    ("TEST", dataset.test_events, dataset.test),
]:

    members = grouping.events_of(primaries)

    types = Counter(e.event_type for e in members)

    labels = Counter(w.label for w in ws)

    print()
    print(name)
    print(f"  waveforms:        {len(primaries)}")
    print(f"  catalogue events: {len(members)}  {dict(types)}")
    print(
        f"  windows: {len(ws)} | pos={labels[1]} "
        f"ign={labels[-1]} neg={labels[0]}"
    )


# ------------------------------------------------------------
# No catalogue event may appear in two splits
# ------------------------------------------------------------

split_of = {}

for name, primaries in [
    ("train", dataset.train_events),
    ("validation", dataset.validation_events),
    ("test", dataset.test_events),
]:
    for evid in grouping.evids_of(primaries):
        split_of.setdefault(evid, set()).add(name)

leaks = {e: s for e, s in split_of.items() if len(s) > 1}

print()
print(f"Events in more than one split: {len(leaks)}  (must be 0)")

print()
print("Shared-waveform events by split:")

for primary, evs in shared.items():
    print(
        f"  {[e.evid for e in evs]} -> "
        f"{sorted({next(iter(split_of[e.evid])) for e in evs})}"
    )