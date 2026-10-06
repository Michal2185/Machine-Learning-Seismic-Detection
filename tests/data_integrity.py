"""
Data-integrity diagnostic.

1. Do several catalogue events share the same waveform (same day file)?
   -> affects event-level splitting (leakage) and labels (a second event
      in the same waveform is currently background for the first one).
2. How much of each record is near-silent (gaps / dropouts)?
3. Where are the largest peaks (event or glitch)?
"""

import hashlib
from collections import defaultdict

import numpy as np

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import (
    Preprocessor,
    PreprocessConfig,
)


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)

BLOCK_SEC = 60
SILENT_REL = 0.01     # block std < 1% of the record's median block std


events = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
).run()

validation_report = Validator(DATA_DIR).run(events)

waveforms = WaveformLoader().run(validation_report.valid_events)

# unfiltered, unscaled: exactly the stored samples (float32)
raw_records = Preprocessor().run(waveforms)

# highpass 0.2 Hz, fixed scale: same setting as the preprocessing test
pre = Preprocessor(
    PreprocessConfig(
        low_hz=0.2,
        high_hz=None,
        normalization="fixed_scale",
    )
)
pre.fit(waveforms)
records = pre.run(waveforms)


# --------------------------------------------------
# 1. Shared waveforms
# --------------------------------------------------

def fingerprint(x):
    return hashlib.md5(
        x[:5000].tobytes() + x[-5000:].tobytes() + str(len(x)).encode()
    ).hexdigest()


groups = defaultdict(list)

for r in raw_records:
    groups[fingerprint(r.data)].append(r)

print()
print("=" * 60)
print("1. SHARED WAVEFORMS")
print("=" * 60)
print(f"Records:                {len(raw_records)}")
print(f"Distinct waveforms:     {len(groups)}")

shared = [g for g in groups.values() if len(g) > 1]

print(f"Waveforms used by >1 event: {len(shared)}")

for g in shared:
    identical = all(np.array_equal(g[0].data, r.data) for r in g[1:])
    print()
    print(f"  identical samples: {identical}")
    for r in g:
        print(
            f"    {r.event.evid} | {r.event.event_type:10s} | "
            f"{r.event.time_abs} | arrival={r.event.time_rel:.0f} s"
        )

# same calendar day but different waveforms (overlapping files?)
by_day = defaultdict(list)

for r in raw_records:
    by_day[str(r.event.time_abs.date())].append(r.event.evid)

multi_day = {d: e for d, e in by_day.items() if len(e) > 1}

print()
print(f"Calendar days with >1 catalogue event: {len(multi_day)}")

for d, e in list(multi_day.items())[:15]:
    print(f"  {d}: {e}")


# --------------------------------------------------
# 2. Near-silent blocks (gaps / dropouts)
# --------------------------------------------------

print()
print("=" * 60)
print("2. NEAR-SILENT BLOCKS (filtered data)")
print("=" * 60)

rows = []

for r in records:

    fs = r.sampling_rate
    blk = int(BLOCK_SEC * fs)
    n = len(r.data) // blk

    blocks = r.data[: n * blk].astype(np.float64).reshape(n, blk)
    stds = blocks.std(axis=1)

    ref = np.median(stds)
    silent = np.mean(stds < SILENT_REL * ref)

    # longest run of exactly constant samples (seconds)
    d = (np.diff(r.data) == 0).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
    longest = int((edges[1::2] - edges[::2]).max()) if len(edges) else 0

    rows.append(
        (r.event.evid, r.event.event_type, silent, ref, longest / fs)
    )

rows.sort(key=lambda t: -t[2])

all_silent = np.array([t[2] for t in rows])

print(f"Fraction of near-silent {BLOCK_SEC}-s blocks per record")
print(f"  median / max: {np.median(all_silent):.3f} / {all_silent.max():.3f}")
print()
print(f"{'evid':10s} {'type':10s} {'silent':>7s} "
      f"{'median block std':>17s} {'longest const run (s)':>22s}")

for evid, t, silent, ref, longest in rows[:10]:
    print(f"{evid:10s} {t:10s} {silent:7.3f} {ref:17.3f} {longest:22.1f}")


# --------------------------------------------------
# 3. Largest peaks (event or glitch?)
# --------------------------------------------------

print()
print("=" * 60)
print("3. LARGEST PEAKS (scaled units)")
print("=" * 60)

peaks = []

for r in records:
    i = int(np.argmax(np.abs(r.data)))
    peaks.append(
        (
            float(np.abs(r.data[i])),
            r.event.evid,
            r.event.event_type,
            i / r.sampling_rate - r.event.time_rel,
        )
    )

peaks.sort(reverse=True)

print(f"{'|peak|':>8s} {'evid':10s} {'type':10s} "
      f"{'peak time - arrival (s)':>24s}")

for p, evid, t, dt in peaks[:8]:
    print(f"{p:8.1f} {evid:10s} {t:10s} {dt:24.0f}")