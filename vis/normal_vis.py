"""
Standalone figures for the normalisation variants.

    python -m tests.visualize_normalization                      # default waveforms
    python -m tests.visualize_normalization evid00003 evid00004  # chosen waveforms
    python -m tests.visualize_normalization --fold 1 --out figs

Per waveform two PNGs are written to OUT_DIR:

  <id>_overview.png  whole day, log scale
      1. robust std of every 10 s block (grey), N2 background estimate (65th pct, 6 h) (red),
         N0 fixed scale (blue dashed); orange = blocks more than 2x above the estimate, black ticks = skipped (gap) blocks, green lines = catalogue arrivals
      2. block level divided by the scale (N0 blue, N2 red): a good
         normalisation puts the background on 1 (grey line) and keeps it
         below the elevated ratio (dashed)
      3. per-block max |x| after N2 and after N3 (asinh): peak compression

  <id>_zoom.png      samples around the first event: raw, N0, N2, N3

Paths: edit the three constants once (same values as in normal_inspect).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.normalization import NormConfig, Normalizer, ScaleCache, robust_std

# ---- EDIT THESE THREE ONCE ---------------------------------------------------
CATALOG_PATH = "data/lunar/catalog.csv"
DATA_DIR = "data/lunar/raw"
DERIVED_DIR = "data/lunar/derived"
# -------------------------------------------------------------------------------

OUT_DIR = "figures/normalization"
DEFAULT_IDS = ["evid00003", "evid00004", "evid00002"]


def block_levels(data, block_len):
    n_blocks = int(np.ceil(len(data) / block_len))
    return np.array([
        robust_std(data[i * block_len:(i + 1) * block_len]) for i in range(n_blocks)
    ])


def block_max(y, block_len):
    n_blocks = int(np.ceil(len(y) / block_len))
    return np.array([
        np.abs(y[i * block_len:(i + 1) * block_len]).max() for i in range(n_blocks)
    ])


def plot_overview(stream, track, fixed_scale, cfg, out_path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = stream.sampling_rate
    L = track.block_len
    t_h = (np.arange(len(track.scale)) + 0.5) * L / fs / 3600

    level = track.level
    level_plot = np.where(level > 0, level, np.nan)

    n2 = Normalizer(NormConfig("N2"), _single_cache(stream, track)).stream(stream)
    n3 = Normalizer(NormConfig("N3"), _single_cache(stream, track)).stream(stream)

    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    a = ax[0]
    a.semilogy(t_h, level_plot, color="0.65", lw=0.7, label="block level (robust std)")
    a.semilogy(t_h, track.scale, color="C3", lw=1.6, label="N2 background estimate (65th pct, 6 h)")
    a.axhline(fixed_scale, color="C0", ls="--", lw=1.2, label="N0 fixed scale")
    a.semilogy(t_h[track.elevated], level_plot[track.elevated], ".", color="C1", ms=3,
               label=f"elevated ({100 * track.elevated.mean():.0f} %)")
    if track.skipped.any():
        a.plot(t_h[track.skipped], np.full(track.skipped.sum(), np.nanmin(level_plot)),
               "|", color="k", label="skipped")
    a.set_ylabel("amplitude")
    a.legend(loc="upper right", ncol=3, fontsize=8)
    a.set_title(f"{stream.waveform_id}: level vs background estimate")

    a = ax[1]
    a.semilogy(t_h, level_plot / fixed_scale, color="C0", lw=0.8, label="N0: level / fixed scale")
    a.semilogy(t_h, level_plot / track.scale, color="C3", lw=0.8, label="N2: level / scale")
    a.axhline(1.0, color="0.4", lw=1)
    a.axhline(cfg.elevated_ratio, color="0.4", ls="--", lw=1, label=f"elevated ratio {cfg.elevated_ratio:g}")
    a.set_ylabel("normalised level")
    a.legend(loc="upper right", ncol=3, fontsize=8)

    a = ax[2]
    a.semilogy(t_h, block_max(n2, L), color="C3", lw=0.8, label="N2: max |x| per block")
    a.semilogy(t_h, block_max(n3, L), color="C2", lw=0.8, label="N3 (asinh)")
    a.set_ylabel("block peak")
    a.set_xlabel("hours since file start")
    a.legend(loc="upper right", fontsize=8)

    for e in stream.events:
        for axis in ax:
            axis.axvline(e.arrival_s / 3600, color="g", alpha=0.6, lw=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_zoom(stream, track, fixed_scale, out_path, before_s=1800.0, after_s=5400.0):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not stream.events:
        return False

    fs = stream.sampling_rate
    ev = sorted(stream.events, key=lambda e: e.arrival_s)[0]
    i0 = max(0, int((ev.arrival_s - before_s) * fs))
    i1 = min(len(stream.data), int((ev.arrival_s + after_s) * fs))
    t = (np.arange(i0, i1) / fs - ev.arrival_s) / 60.0

    cache = _single_cache(stream, track)
    raw = stream.data[i0:i1]
    n0 = (stream.data / fixed_scale)[i0:i1]
    n2 = Normalizer(NormConfig("N2"), cache).stream(stream)[i0:i1]
    n3 = Normalizer(NormConfig("N3"), cache).stream(stream)[i0:i1]

    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for axis, y, name in zip(ax, (raw, n0, n2, n3), ("raw", "N0", "N2", "N3 (asinh)")):
        axis.plot(t, y, lw=0.4)
        axis.set_ylabel(name)
        axis.axvline(0, color="g", lw=1)
        if ev.onset_s is not None:
            axis.axvline((ev.onset_s - ev.arrival_s) / 60.0, color="r", ls="--", lw=1)
    ax[0].set_title(f"{stream.waveform_id} {ev.evid} ({ev.event_type}): green = catalogue time, "
                    f"red dashed = visible onset")
    ax[-1].set_xlabel("minutes from catalogue arrival")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return True


def _single_cache(stream, track):
    cache = ScaleCache(NormConfig("N2"))
    cache.tracks[stream.waveform_id] = track
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--catalog", default=CATALOG_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--derived-dir", default=DERIVED_DIR)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    from src.modules.pipeline import DataPipeline, PipelineConfig

    bundle = DataPipeline(PipelineConfig(a.catalog, a.data_dir, a.derived_dir)).build()
    split = bundle.fold_split(a.fold)

    cfg = NormConfig("N2")
    fixed = Normalizer(NormConfig("N0")).fit_streams(
        bundle.eval_streams(split.train_events)).fixed_scale
    print(f"N0 fixed scale (fold {a.fold}): {fixed:.4g}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for wid in a.ids:
        stream = bundle.eval_stream(wid)
        cache = ScaleCache.from_streams([stream], cfg)
        track = cache.get(wid, fixed)                      # N2 v3: reference = N0 constant of the fold
        plot_overview(stream, track, fixed, cfg, out / f"{wid}_overview.png")
        plot_zoom(stream, track, fixed, out / f"{wid}_zoom.png")
        print(f"{wid}: elevated {100 * track.elevated.mean():.1f} %, wrote {out / wid}_*.png")


if __name__ == "__main__":
    main()