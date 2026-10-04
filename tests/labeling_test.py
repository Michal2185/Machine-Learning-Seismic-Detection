from collections import Counter, defaultdict

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling, POSITIVE, NEGATIVE, IGNORE


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

total = len(labeled_windows)


print()
print("=" * 60)
print("LABELING TEST")
print("=" * 60)

print(
    f"Total windows: {total}"
)

print(
    f"Negative (0):  {counts[NEGATIVE]}"
)

print(
    f"Positive (1):  {counts[POSITIVE]}"
)

print(
    f"Ignore (-1):   {counts[IGNORE]} "
    f"({100 * counts[IGNORE] / total:.2f}% of windows)"
)

print(
    f"Check sum:     "
    f"{counts[NEGATIVE] + counts[POSITIVE] + counts[IGNORE] == total}"
)


# ------------------------------------------------------------
# Summary per event type + events without positives
# ------------------------------------------------------------

print()

Labeling.print_summary(labeled_windows)


# ------------------------------------------------------------
# Positives per event
# ------------------------------------------------------------

positives_per_event = Counter(
    window.event.evid
    for window in labeled_windows
    if window.label == POSITIVE
)

all_event_ids = {
    window.event.evid
    for window in labeled_windows
}

print()
print(
    f"Events:                    {len(all_event_ids)}"
)

print(
    f"Events with >=1 positive:  {len(positives_per_event)}"
)

print(
    f"Positives per event "
    f"(min/max): "
    f"{min(positives_per_event.values())}/"
    f"{max(positives_per_event.values())}"
)


# ------------------------------------------------------------
# Inspect positive windows
# ------------------------------------------------------------

positive_windows = [
    window
    for window in labeled_windows
    if window.label == POSITIVE
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


# ------------------------------------------------------------
# Inspect label timeline around events
# (consecutive windows with the same label merged into one segment)
# ------------------------------------------------------------

windows_by_event = defaultdict(list)

for window in labeled_windows:
    windows_by_event[window.event.evid].append(window)

NAMES = {POSITIVE: "POSITIVE", NEGATIVE: "negative", IGNORE: "ignore"}

print()
print("=" * 60)
print("LABEL TIMELINE (first 3 events, non-background segments)")
print("=" * 60)

for evid in list(windows_by_event)[:3]:

    event_windows = windows_by_event[evid]
    event = event_windows[0].event

    print()
    print(
        f"{evid} | type={event.event_type} | "
        f"arrival={event.time_rel:.2f} s"
    )

    segments = []

    for window in event_windows:

        if segments and segments[-1]["label"] == window.label:
            segments[-1]["end"] = window.end_time
            segments[-1]["n"] += 1
        else:
            segments.append(
                {
                    "label": window.label,
                    "start": window.start_time,
                    "end": window.end_time,
                    "n": 1,
                }
            )

    for seg in segments:

        # skip the long clean-background segments, keep the transitions
        if seg["label"] == NEGATIVE and seg["n"] > 50:
            print(
                f"  {NAMES[seg['label']]:9s} "
                f"{seg['start']:10.2f} - {seg['end']:10.2f} s "
                f"| {seg['n']} windows (background)"
            )
            continue

        print(
            f"  {NAMES[seg['label']]:9s} "
            f"{seg['start']:10.2f} - {seg['end']:10.2f} s "
            f"| {seg['n']} windows"
        )