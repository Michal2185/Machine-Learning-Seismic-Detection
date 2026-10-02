from collections import Counter

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling


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

data_loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = data_loader.run()


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

processed_waveforms = preprocessor.run(
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

windows = windowing.run(
    processed_waveforms
)


# ------------------------------------------------------------
# Labeling
# ------------------------------------------------------------

labeling = Labeling()

labeled_windows = labeling.run(
    windows
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

labels = [
    window.label
    for window in labeled_windows
]

counts = Counter(labels)


print()
print("=" * 60)
print("LABELING TEST")
print("=" * 60)

print(
    f"Total windows: {len(labeled_windows)}"
)

print(
    f"Negative (0):  {counts[0]}"
)

print(
    f"Positive (1):  {counts[1]}"
)


# ------------------------------------------------------------
# Inspect positive windows
# ------------------------------------------------------------

positive_windows = [
    window
    for window in labeled_windows
    if window.label == 1
]


print()
print(
    f"Positive windows: {len(positive_windows)}"
)

print()

for window in positive_windows[:20]:

    print(
        f"{window.event.evid} | "
        f"{window.start_time:.2f} - "
        f"{window.end_time:.2f} s | "
        f"arrival={window.event.time_rel:.2f} | "
        f"label={window.label}"
    )