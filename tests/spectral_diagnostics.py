"""
Spectral diagnostic for choosing the preprocessing band.

Compares the power spectrum of the first EVENT_SEC seconds after the
catalogue arrival with background segments of the same day, for TRAINING
events only (same seed-42 split as the pipeline), so the band choice does
not use validation/test events.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

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

EVENT_SEC = 600          # segment right after the arrival
BG_SEC = 600             # background segment length (same as EVENT_SEC)
BG_SEGMENTS = 10         # background segments per event
EXCLUDE_BEFORE = 300     # keep background away from the event ...
EXCLUDE_AFTER = 3600     # ... including a long coda
NPERSEG = 1024           # ~155 s -> 0.0065 Hz resolution at 6.625 Hz
SEED = 42

BANDS = [
    (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4),
    (0.4, 0.8), (0.8, 1.6), (1.6, 3.3),
]


# --------------------------------------------------
# 1. Pipeline up to the train/val/test split
# --------------------------------------------------

events = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
).run()

validation_report = Validator(DATA_DIR).run(events)

waveforms = WaveformLoader().run(validation_report.valid_events)

processed = Preprocessor().run(waveforms)       # unfiltered on purpose

windows = Windowing(
    WindowConfig(window_length=60, step_size=10)
).run(processed)

labeled = Labeling().run(windows)

dataset = DatasetBuilder(
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
).run(labeled)

train_ids = {w.event.evid for w in dataset.train}

train_records = [r for r in processed if r.event.evid in train_ids]

print()
print(f"Training events used: {len(train_records)}")


# --------------------------------------------------
# 2. Event vs background spectra
# --------------------------------------------------

rng = np.random.default_rng(SEED)

freqs = None
ratios_db = []
types = []
event_psd_all = []
bg_psd_all = []
raw_stds = []

for record in train_records:

    x = record.data.astype(np.float64)
    fs = record.sampling_rate
    arrival = record.event.time_rel

    n_seg = int(EVENT_SEC * fs)
    a = int(arrival * fs)

    if a + n_seg > len(x):
        continue

    f, p_event = welch(
        x[a:a + n_seg], fs=fs, nperseg=NPERSEG,
        noverlap=NPERSEG // 2, detrend="linear",
    )

    # background segments away from the event
    bg = []
    tries = 0

    while len(bg) < BG_SEGMENTS and tries < 500:
        tries += 1
        s = int(rng.uniform(0, len(x) - n_seg))
        t0, t1 = s / fs, (s + n_seg) / fs

        overlaps = (
            t1 > arrival - EXCLUDE_BEFORE
            and t0 < arrival + EXCLUDE_AFTER
        )

        if overlaps:
            continue

        _, p = welch(
            x[s:s + n_seg], fs=fs, nperseg=NPERSEG,
            noverlap=NPERSEG // 2, detrend="linear",
        )
        bg.append(p)

    if len(bg) < 3:
        continue

    p_bg = np.mean(bg, axis=0)

    freqs = f
    ratios_db.append(10 * np.log10(p_event[1:] / p_bg[1:]))
    event_psd_all.append(p_event[1:])
    bg_psd_all.append(p_bg[1:])
    types.append(record.event.event_type)
    raw_stds.append(float(1.4826 * np.median(np.abs(x - np.median(x)))))

ratios_db = np.array(ratios_db)
types = np.array(types)
f = freqs[1:]

print(f"Events analysed: {len(ratios_db)}")


# --------------------------------------------------
# 3. Table per band
# --------------------------------------------------

def band_stats(ratio_rows):

    rows = []

    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        per_event = ratio_rows[:, m].mean(axis=1)   # mean dB in the band
        rows.append(
            (
                lo, hi,
                np.median(per_event),
                np.percentile(per_event, 25),
                np.percentile(per_event, 75),
                np.mean(per_event > 3.0),
            )
        )

    return rows


def print_table(title, ratio_rows):

    print()
    print(title)
    print(f"{'band (Hz)':>14s} {'median dB':>10s} "
          f"{'IQR dB':>16s} {'frac > 3 dB':>12s}")

    for lo, hi, med, q1, q3, frac in band_stats(ratio_rows):
        print(
            f"{lo:6.2f} - {hi:5.2f} {med:10.1f} "
            f"{q1:7.1f} .. {q3:6.1f} {frac:12.2f}"
        )


print("=" * 60)
print("EVENT / BACKGROUND POWER RATIO PER BAND")
print("=" * 60)

print_table(f"All training events (n={len(ratios_db)})", ratios_db)

for t in sorted(set(types)):
    sel = types == t
    if sel.sum() >= 3:
        print_table(f"{t} (n={sel.sum()})", ratios_db[sel])
    else:
        print()
        print(f"{t}: n={sel.sum()} (too few for a table)")


# --------------------------------------------------
# 4. Raw amplitude spread (is one global scale reasonable?)
# --------------------------------------------------

raw_stds = np.array(raw_stds)

print()
print("Raw robust std per event")
print(f"  min / median / max: {raw_stds.min():.3e} / "
      f"{np.median(raw_stds):.3e} / {raw_stds.max():.3e}")
print(f"  max/min ratio: {raw_stds.max() / raw_stds.min():.1f}")


# --------------------------------------------------
# 5. Plots
# --------------------------------------------------

med = np.median(ratios_db, axis=0)
q1 = np.percentile(ratios_db, 25, axis=0)
q3 = np.percentile(ratios_db, 75, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

axes[0].semilogx(f, med, label="median")
axes[0].fill_between(f, q1, q3, alpha=0.3, label="IQR over events")
axes[0].axhline(0, color="gray", linewidth=0.8)
axes[0].axhline(3, color="red", linestyle=":", linewidth=0.8, label="3 dB")
axes[0].set_ylabel("Event / background (dB)")
axes[0].set_title("Spectral SNR of the first "
                  f"{EVENT_SEC}s after arrival (training events)")
axes[0].legend()

axes[1].loglog(f, np.median(event_psd_all, axis=0), label="event segment")
axes[1].loglog(f, np.median(bg_psd_all, axis=0), label="background")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Median PSD")
axes[1].legend()

plt.tight_layout()
plt.show()