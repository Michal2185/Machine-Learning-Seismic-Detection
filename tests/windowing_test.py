import matplotlib.pyplot as plt
import numpy as np

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

data_loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = data_loader.run()


# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

validator = Validator(DATA_DIR)

validation_report = validator.run(events)


# ------------------------------------------------------------
# Load waveforms
# ------------------------------------------------------------

waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(
    validation_report.valid_events
)


# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

preprocessor = Preprocessor()

processed_waveforms = preprocessor.run(
    waveforms
)


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------

config = WindowConfig(
    window_length=60,
    step_size=10,
)

windowing = Windowing(config)

windows = windowing.run(
    processed_waveforms
)


# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print()
print("=" * 60)
print("WINDOWING TEST")
print("=" * 60)

print(f"Events:             {len(processed_waveforms)}")
print(f"Total windows:      {len(windows)}")
print(f"Window length:      {config.window_length} s")
print(f"Step size:          {config.step_size} s")


# ------------------------------------------------------------
# Inspect first event
# ------------------------------------------------------------

event_id = processed_waveforms[0].event.evid

event_windows = [
    window
    for window in windows
    if window.event.evid == event_id
]

print()
print(f"Event: {event_id}")
print(f"Windows: {len(event_windows)}")

for window in event_windows[:5]:

    print(
        f"Window: "
        f"{window.start_time:.2f} - "
        f"{window.end_time:.2f} s | "
        f"samples: {len(window.data)}"
    )


# ------------------------------------------------------------
# Plot windows around arrival
# ------------------------------------------------------------

event = processed_waveforms[0].event

arrival = event.time_rel


relevant_windows = [
    window
    for window in event_windows
    if (
        window.start_time <= arrival
        and window.end_time >= arrival
    )
]


plt.figure(figsize=(14, 5))


for window in relevant_windows:

    time = np.arange(
        len(window.data)
    ) / processed_waveforms[0].sampling_rate

    time += window.start_time

    plt.plot(
        time,
        window.data,
        linewidth=0.8,
    )


plt.axvline(
    arrival,
    color="red",
    linestyle="--",
    label="Catalogue arrival",
)


plt.title(
    f"Windows around arrival | {event.evid}"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()

plt.tight_layout()
plt.show()