import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import (
    Preprocessor,
    PreprocessConfig,
    apply_filter,
    robust_std,
)


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)

# Chosen from tests/diagnostic_spectra.py (52 training events):
# no event/background contrast below ~0.15 Hz, strongest at 0.4-0.8 Hz.
# Variant P0: highpass only. Variant P1: set HIGH_HZ = 1.0 (ablation).
LOW_HZ = 0.2
HIGH_HZ = None

BAND_LABEL = f"{LOW_HZ}-{HIGH_HZ if HIGH_HZ else 'Nyquist'} Hz"

INDEX = 0


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

events = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
).run()

validation_report = Validator(DATA_DIR).run(events)

waveforms = WaveformLoader().run(validation_report.valid_events)


# --------------------------------------------------
# 2. Backward compatibility: default config == old behaviour
# --------------------------------------------------

default_processed = Preprocessor().run(waveforms)

raw0 = waveforms[INDEX].stream[0].data
diff0 = np.max(
    np.abs(raw0.astype(np.float32) - default_processed[INDEX].data)
)

print()
print("=" * 60)
print("DEFAULT CONFIG (must be identical to the old preprocessor)")
print("=" * 60)
print(f"Maximum absolute difference: {diff0}")


# --------------------------------------------------
# 3. Filtered + fixed-scale preprocessing
# --------------------------------------------------

preprocessor = Preprocessor(
    PreprocessConfig(
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
        filter_order=4,
        normalization="fixed_scale",
    )
)

# NOTE: for the real pipeline fit on TRAINING records only.
# Here (visual check) all records are used.
scale = preprocessor.fit(waveforms)

processed = preprocessor.run(waveforms)

print()
print("=" * 60)
print("FILTERED + FIXED SCALE")
print("=" * 60)
print(f"Band:  {BAND_LABEL} (causal Butterworth, order 4)")
print(f"Fixed scale: {scale:.4e}")

record = processed[INDEX]
event = record.event
fs = record.sampling_rate

print(f"Event: {event.evid} | {event.event_type} | arrival {event.time_rel} s")
print(f"dtype: {record.data.dtype} | samples: {len(record.data)}")
print(f"min/max: {record.data.min():.3f} / {record.data.max():.3f}")
print(f"robust std: {robust_std(record.data):.3f}")


# --------------------------------------------------
# 4. Per-record amplitude statistics after scaling
#    (is ONE global scale reasonable across events?)
# --------------------------------------------------

stds = np.array([robust_std(r.data) for r in processed])
peaks = np.array([np.max(np.abs(r.data)) for r in processed])

print()
print("Robust std per record (after scaling)")
print(f"  min / median / max: {stds.min():.3f} / "
      f"{np.median(stds):.3f} / {stds.max():.3f}")
print("Peak |amplitude| per record (after scaling)")
print(f"  min / median / max: {peaks.min():.1f} / "
      f"{np.median(peaks):.1f} / {peaks.max():.1f}")


# --------------------------------------------------
# 4b. Low-amplitude / high-amplitude records (raw, before scaling)
#     A single global scale fails if a record is ~100x quieter.
# --------------------------------------------------

raw_stds = np.array(
    [robust_std(w.stream[0].data) for w in waveforms]
)

order = np.argsort(raw_stds)

print()
print("Lowest raw robust std (possible gaps / flat segments / bad gain)")

for i in order[:5]:

    d = np.asarray(waveforms[i].stream[0].data, dtype=np.float64)

    flat = np.mean(np.diff(d) == 0)

    print(
        f"  {waveforms[i].event.evid} | "
        f"{waveforms[i].event.event_type:10s} | "
        f"raw std={raw_stds[i]:.3e} | "
        f"filtered+scaled std={stds[i]:.3f} | "
        f"flat fraction={flat:.3f} | "
        f"zero fraction={np.mean(d == 0):.3f}"
    )

print("Highest raw robust std")

for i in order[-3:][::-1]:

    print(
        f"  {waveforms[i].event.evid} | "
        f"{waveforms[i].event.event_type:10s} | "
        f"raw std={raw_stds[i]:.3e} | "
        f"filtered+scaled std={stds[i]:.3f}"
    )


# --------------------------------------------------
# 5. Causality check on real data
# --------------------------------------------------

raw = np.asarray(waveforms[INDEX].stream[0].data, dtype=np.float64)

N = len(raw) // 2

full = apply_filter(raw, fs, LOW_HZ, HIGH_HZ, 4)
part = apply_filter(raw[:N], fs, LOW_HZ, HIGH_HZ, 4)

print()
print("Causality check (truncated vs full filtering)")
print(f"  max |difference| over first {N} samples: "
      f"{np.max(np.abs(full[:N] - part))}  (must be 0.0)")


# --------------------------------------------------
# 6. Start-up transient
# --------------------------------------------------

n60 = int(60 * fs)
n600 = int(600 * fs)

print()
print("Start-up transient")
print(f"  std first 60 s / std after 600 s: "
      f"{full[:n60].std() / full[n600:].std():.2f}  (should be ~1)")


# --------------------------------------------------
# 7. Plots: waveform, zoom, spectrum before/after
# --------------------------------------------------

time = np.arange(len(record.data)) / fs

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(time, raw, linewidth=0.5)
axes[0].axvline(event.time_rel, color="red", linestyle="--", label="Arrival")
axes[0].set_title(f"Raw | {event.evid}")
axes[0].set_ylabel("Amplitude")
axes[0].legend()

axes[1].plot(time, record.data, linewidth=0.5)
axes[1].axvline(event.time_rel, color="red", linestyle="--", label="Arrival")
axes[1].set_title(f"Filtered {BAND_LABEL} + fixed scale")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Scaled amplitude")
axes[1].legend()

plt.tight_layout()
plt.show()


BEFORE, AFTER = 300, 900

mask = (
    (time >= max(0, event.time_rel - BEFORE))
    & (time <= min(time[-1], event.time_rel + AFTER))
)

plt.figure(figsize=(14, 5))
plt.plot(time[mask], record.data[mask], linewidth=0.7)
plt.axvline(event.time_rel, color="red", linestyle="--", label="Arrival")
plt.title(f"{event.evid} | -{BEFORE}s / +{AFTER}s around arrival (filtered)")
plt.xlabel("Time (seconds)")
plt.ylabel("Scaled amplitude")
plt.legend()
plt.tight_layout()
plt.show()


f_raw, p_raw = welch(raw, fs=fs, nperseg=4096, detrend="linear")
f_fil, p_fil = welch(
    record.data.astype(np.float64) * record.scale,
    fs=fs, nperseg=4096, detrend="linear",
)

plt.figure(figsize=(10, 5))
plt.loglog(f_raw[1:], p_raw[1:], label="Raw")
plt.loglog(f_fil[1:], p_fil[1:], label="Filtered")
if LOW_HZ:
    plt.axvline(LOW_HZ, color="gray", linestyle=":")
if HIGH_HZ:
    plt.axvline(HIGH_HZ, color="gray", linestyle=":")
plt.title("Power spectral density, whole day")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()
plt.tight_layout()
plt.show()