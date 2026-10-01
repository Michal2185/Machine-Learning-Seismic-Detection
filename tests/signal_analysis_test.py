import numpy as np
import matplotlib.pyplot as plt

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


# ============================================================
# Configuration
# ============================================================

# Events to inspect.
# Change these indices after looking at the catalogue.
EVENT_INDICES = [21, 4, 20]

# Time shown around the catalogue arrival.
WINDOW_BEFORE = 300
WINDOW_AFTER = 300


# ============================================================
# Load pipeline
# ============================================================

data_loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = data_loader.run()


validator = Validator(DATA_DIR)

validation_report = validator.run(events)


waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(
    validation_report.valid_events
)


preprocessor = Preprocessor()

processed_waveforms = preprocessor.run(waveforms)


# ============================================================
# Helper function
# ============================================================

def analyse_event(record):
    """
    Analyse one processed seismic event.
    """

    event = record.event

    data = record.data

    sampling_rate = record.sampling_rate

    arrival_time = event.time_rel

    # Time axis of the complete waveform
    time = np.arange(len(data)) / sampling_rate

    # --------------------------------------------------------
    # Select window around arrival
    # --------------------------------------------------------

    start = max(
        0,
        arrival_time - WINDOW_BEFORE,
    )

    end = min(
        time[-1],
        arrival_time + WINDOW_AFTER,
    )

    mask = (
        (time >= start)
        & (time <= end)
    )

    window_time = time[mask]
    window_data = data[mask]

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Event:          {event.evid}")
    print(f"Type:            {event.event_type}")
    print(f"Arrival:         {arrival_time:.3f} s")
    print(f"Sampling rate:   {sampling_rate} Hz")
    print(f"Samples in view: {len(window_data)}")

    print()
    print("Amplitude statistics")
    print("--------------------")
    print(f"Minimum:  {np.min(window_data):.6g}")
    print(f"Maximum:  {np.max(window_data):.6g}")
    print(f"Mean:     {np.mean(window_data):.6g}")
    print(f"Std:      {np.std(window_data):.6g}")
    print(f"RMS:      {np.sqrt(np.mean(window_data ** 2)):.6g}")

    # --------------------------------------------------------
    # Plot waveform
    # --------------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(
        window_time,
        window_data,
        linewidth=0.7,
    )

    plt.axvline(
        arrival_time,
        color="red",
        linestyle="--",
        label="Catalogue arrival",
    )

    plt.title(
        f"{event.evid} | {event.event_type}"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.legend()

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot amplitude distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        window_data,
        bins=100,
    )

    plt.title(
        f"Amplitude distribution | {event.evid}"
    )

    plt.xlabel("Amplitude")
    plt.ylabel("Number of samples")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Frequency analysis
    # --------------------------------------------------------

    # Remove mean before FFT
    signal = window_data - np.mean(window_data)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / sampling_rate,
    )

    spectrum = np.abs(
        np.fft.rfft(signal)
    )

    # --------------------------------------------------------
    # Plot spectrum
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        frequencies,
        spectrum,
        linewidth=0.8,
    )

    plt.xlim(
        0,
        sampling_rate / 2,
    )

    plt.title(
        f"Frequency spectrum | {event.evid}"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()


# ============================================================
# Analyse selected events
# ============================================================

for index in EVENT_INDICES:

    if index >= len(processed_waveforms):
        print(
            f"Skipping index {index}: "
            f"only {len(processed_waveforms)} events available."
        )
        continue

    analyse_event(
        processed_waveforms[index]
    )