from collections import Counter

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
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
# Load
# ------------------------------------------------------------

loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = loader.run()


# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

validator = Validator(DATA_DIR)

validation_report = validator.run(events)


# ------------------------------------------------------------
# Waveforms
# ------------------------------------------------------------

waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(
    validation_report.valid_events
)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

preprocessor = Preprocessor()

processed = preprocessor.run(
    waveforms
)


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------

windowing = Windowing(
    WindowConfig(
        window_length=60,
        step_size=10,
    )
)

windows = windowing.run(processed)


# ------------------------------------------------------------
# Labeling
# ------------------------------------------------------------

labeling = Labeling()

labeled_windows = labeling.run(
    windows
)


# ------------------------------------------------------------
# Dataset split
# ------------------------------------------------------------

dataset_builder = DatasetBuilder(
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)

dataset = dataset_builder.run(
    labeled_windows
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 60)
print("DATASET SPLIT TEST")
print("=" * 60)

print()

print(
    f"Train events:       {len(dataset.train_events)}"
)

print(
    f"Validation events:  {len(dataset.validation_events)}"
)

print(
    f"Test events:        {len(dataset.test_events)}"
)

print()

print(
    f"Train windows:      {len(dataset.train)}"
)

print(
    f"Validation windows: {len(dataset.validation)}"
)

print(
    f"Test windows:       {len(dataset.test)}"
)


# ------------------------------------------------------------
# Label distributions
# ------------------------------------------------------------

print()
print("Label distribution:")

for name, split in [
    ("TRAIN", dataset.train),
    ("VALIDATION", dataset.validation),
    ("TEST", dataset.test),
]:

    counts = Counter(
        window.label
        for window in split
    )

    print()
    print(name)
    print(
        f"  Negative: {counts[0]}"
    )
    print(
        f"  Positive: {counts[1]}"
    )


# ------------------------------------------------------------
# Leakage checks
# ------------------------------------------------------------

train_ids = set(dataset.train_events)
validation_ids = set(
    dataset.validation_events
)
test_ids = set(dataset.test_events)


print()
print("Leakage checks:")

print(
    "Train ∩ Validation:",
    train_ids & validation_ids
)

print(
    "Train ∩ Test:",
    train_ids & test_ids
)

print(
    "Validation ∩ Test:",
    validation_ids & test_ids
)