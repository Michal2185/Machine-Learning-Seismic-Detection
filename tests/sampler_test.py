from collections import Counter

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling, POSITIVE, NEGATIVE, IGNORE
from src.modules.dataset_builder import DatasetBuilder
from src.modules.sampler import Sampler


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


# ------------------------------------------------------------
# Pipeline up to the split
# ------------------------------------------------------------

events = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
).run()

validation_report = Validator(DATA_DIR).run(events)

waveforms = WaveformLoader().run(validation_report.valid_events)

grouping = WaveformGrouper().run(waveforms)

processed = Preprocessor().run(grouping.waveforms)

windows = Windowing(
    WindowConfig(window_length=60, step_size=10)
).run(processed)

labeled = Labeling(event_groups=grouping.event_groups).run(windows)

dataset = DatasetBuilder(
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
).run(labeled)


# ------------------------------------------------------------
# Sampler
# ------------------------------------------------------------

sampler = Sampler(
    negatives_per_positive=5,
    random_seed=42,
)

train_counts = Counter(w.label for w in dataset.train)

print()
print("=" * 60)
print("SAMPLER TEST")
print("=" * 60)
print("Training split before sampling:")
print(f"  Positive: {train_counts[POSITIVE]}")
print(f"  Ignore:   {train_counts[IGNORE]}")
print(f"  Negative: {train_counts[NEGATIVE]}")


def key(w):
    return (w.event.evid, round(w.start_time, 3))


epoch_negatives = []

for epoch in range(3):

    result = sampler.run(dataset.train, epoch=epoch)

    labels = Counter(w.label for w in result.windows)

    negs = {key(w) for w in result.windows if w.label == NEGATIVE}
    epoch_negatives.append(negs)

    waveforms_used = len({w.event.evid for w in result.windows if w.label == NEGATIVE})

    print()
    print(f"Epoch {epoch}")
    print(f"  Positive: {labels[POSITIVE]} (all {train_counts[POSITIVE]} kept: "
          f"{labels[POSITIVE] == train_counts[POSITIVE]})")
    print(f"  Negative: {labels[NEGATIVE]} "
          f"of {result.n_negative_available} available "
          f"({100 * labels[NEGATIVE] / result.n_negative_available:.2f}%)")
    print(f"  Ignore:   {labels[IGNORE]}  (must be 0)")
    print(f"  Negatives per positive: {result.negatives_per_positive:.2f}")
    print(f"  Waveforms contributing negatives: {waveforms_used}")
    print(f"  First 20 labels: {[w.label for w in result.windows[:20]]}")


print()
print("Overlap of sampled negatives between epochs")
print(f"  epoch 0 vs 1: {len(epoch_negatives[0] & epoch_negatives[1])}")
print(f"  epoch 0 vs 2: {len(epoch_negatives[0] & epoch_negatives[2])}")
print(f"  epoch 1 vs 2: {len(epoch_negatives[1] & epoch_negatives[2])}")

union = set().union(*epoch_negatives)
print(f"  distinct negatives seen over 3 epochs: {len(union)}")

again = sampler.run(dataset.train, epoch=0)
same = {key(w) for w in again.windows if w.label == NEGATIVE} == epoch_negatives[0]
print(f"Same seed + epoch gives the same subset: {same}")