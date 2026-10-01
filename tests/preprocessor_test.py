import matplotlib.pyplot as plt
import numpy as np

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import Preprocessor


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


# --------------------------------------------------
# 1. Load catalogue
# --------------------------------------------------

data_loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = data_loader.run()


# --------------------------------------------------
# 2. Validate catalogue
# --------------------------------------------------

validator = Validator(DATA_DIR)

validation_report = validator.run(events)

valid_events = validation_report.valid_events


# --------------------------------------------------
# 3. Load waveforms
# --------------------------------------------------

waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(valid_events)


# --------------------------------------------------
# 4. Preprocess
# --------------------------------------------------

preprocessor = Preprocessor()

processed_waveforms = preprocessor.run(waveforms)


# --------------------------------------------------
# 5. Select one event
# --------------------------------------------------

INDEX = 0

raw_record = waveforms[INDEX]
processed_record = processed_waveforms[INDEX]

event = processed_record.event


print(f"Event: {event.evid}")
print(f"Type:  {event.event_type}")
print(f"Arrival: {event.time_rel} seconds")


# --------------------------------------------------
# 6. Extract raw data
# --------------------------------------------------

trace = raw_record.stream[0]

raw_data = trace.data
processed_data = processed_record.data

sampling_rate = processed_record.sampling_rate


# --------------------------------------------------
# 7. Verify preprocessing
# --------------------------------------------------

print()
print("Raw data")
print("--------")
print(f"dtype: {raw_data.dtype}")
print(f"samples: {len(raw_data)}")


print()
print("Processed data")
print("--------------")
print(f"dtype: {processed_data.dtype}")
print(f"samples: {len(processed_data)}")


print()
print("Difference")
print("----------")

difference = raw_data.astype(np.float32) - processed_data

print(f"Maximum absolute difference: {np.max(np.abs(difference))}")


# --------------------------------------------------
# 8. Plot complete waveform
# --------------------------------------------------

raw_time = np.arange(len(raw_data)) / sampling_rate
processed_time = np.arange(len(processed_data)) / sampling_rate


fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True,
)


axes[0].plot(
    raw_time,
    raw_data,
    linewidth=0.5,
)

axes[0].axvline(
    event.time_rel,
    color="red",
    linestyle="--",
    label="Arrival",
)

axes[0].set_title(
    f"Raw waveform | {event.evid}"
)

axes[0].set_ylabel("Amplitude")
axes[0].legend()


axes[1].plot(
    processed_time,
    processed_data,
    linewidth=0.5,
)

axes[1].axvline(
    event.time_rel,
    color="red",
    linestyle="--",
    label="Arrival",
)

axes[1].set_title(
    "Preprocessed waveform"
)

axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()


plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. Zoom around arrival
# --------------------------------------------------

WINDOW_BEFORE = 300
WINDOW_AFTER = 300

start_time = max(
    0,
    event.time_rel - WINDOW_BEFORE,
)

end_time = min(
    raw_time[-1],
    event.time_rel + WINDOW_AFTER,
)


raw_mask = (
    (raw_time >= start_time)
    & (raw_time <= end_time)
)

processed_mask = (
    (processed_time >= start_time)
    & (processed_time <= end_time)
)


plt.figure(figsize=(14, 5))

plt.plot(
    raw_time[raw_mask],
    raw_data[raw_mask],
    linewidth=0.7,
    label="Raw",
)

plt.plot(
    processed_time[processed_mask],
    processed_data[processed_mask],
    linewidth=0.7,
    linestyle="--",
    label="Preprocessed",
)

plt.axvline(
    event.time_rel,
    color="red",
    linestyle="--",
    label="Arrival",
)

plt.title(
    f"{event.evid} | ±{WINDOW_BEFORE}s around arrival"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.tight_layout()
plt.show()