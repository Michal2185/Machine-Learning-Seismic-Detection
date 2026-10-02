from collections import Counter

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling
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
# Data loading
# ------------------------------------------------------------

loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = loader.run()


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

validator = Validator(DATA_DIR)

validation_report = validator.run(events)


# ------------------------------------------------------------
# Waveform loading
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
# Sampling
# ------------------------------------------------------------

sampler = Sampler(
    positive_to_negative_ratio=5,
    random_seed=42,
)

sampling_result = sampler.run(
    dataset.train
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 60)
print("SAMPLING TEST")
print("=" * 60)

print()

print("Original training set:")
print(
    f"  Positive: "
    f"{sampling_result.original_positive_count}"
)

print(
    f"  Negative: "
    f"{sampling_result.original_negative_count}"
)

print(
    f"  Total:    "
    f"{len(dataset.train)}"
)

print()

print("Sampled training set:")
print(
    f"  Positive: "
    f"{sampling_result.positive_count}"
)

print(
    f"  Negative: "
    f"{sampling_result.negative_count}"
)

print(
    f"  Total:    "
    f"{len(sampling_result.windows)}"
)

print()

ratio = (
    sampling_result.negative_count
    / sampling_result.positive_count
)

print(
    f"Negative/positive ratio: "
    f"{ratio:.2f}:1"
)

print()

counts = Counter(
    window.label
    for window in sampling_result.windows
)

print("Final label distribution:")
print(
    f"  0: {counts[0]}"
)

print(
    f"  1: {counts[1]}"
)