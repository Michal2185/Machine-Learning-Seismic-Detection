"""
Manual check of the normalisation variants on the real Apollo 12 bundle.

    python -m tests.normal_inspect [--fig evid00002] [--fold 0]

Paths are set in the constants below (edit once); the optional flags
--catalog / --data-dir / --derived-dir override them.

Prints, per variant, how much of the day-to-day noise spread and of the
event-peak spread survives, plus scale-track diagnostics; optionally saves
a figure (raw, N2 scale track, N0 / N2 / N3 output) for one waveform.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.dnn_features import FEATURE_NAMES, extract_features
from src.modules.normalization import NormConfig, Normalizer, ScaleCache, robust_std
from src.modules.pipeline import DataPipeline, PipelineConfig


CATALOG_PATH = "C:/GIT/Machine-Learning-Seismic-Detection/data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
DATA_DIR = "C:/GIT/Machine-Learning-Seismic-Detection/data/lunar/data/S12_GradeA/"
DERIVED_DIR = "C:/GIT/Machine-Learning-Seismic-Detection/data/lunar/derived/"


def spread(values):
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0])
    return v.max() / v.min(), np.median(v), np.percentile(v, 90) / np.percentile(v, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=CATALOG_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--derived-dir", default=DERIVED_DIR)
    ap.add_argument("--fig", default=None, help="waveform id to plot")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    bundle = DataPipeline(PipelineConfig(a.catalog, a.data_dir, a.derived_dir)).build()
    ids = sorted(bundle._processed_by_id)
    streams = bundle.eval_streams(ids)
    fs = streams[0].sampling_rate

    print(f"{len(streams)} waveforms, fs {fs} Hz")

    cache = ScaleCache.from_streams(streams, NormConfig("N2"))
    split = bundle.fold_split(a.fold)

    by_id = {}
    for w in bundle.windows:
        by_id.setdefault(w.event.evid, []).append(w)

    print()
    print(f"{'variant':8s} {'bg std max/min':>15s} {'p90/p10':>9s} | {'event peak max/min':>19s} {'p90/p10':>9s}")

    for v in ("N0", "N1", "N2", "N3"):

        norm = Normalizer(NormConfig(v), cache if v in ("N2", "N3") else None).fit(split.train)

        bg, pk = [], []

        for wid, wins in by_id.items():
            neg = [w for w in wins if w.label == 0][::30]
            pos = [w for w in wins if w.label == 1]
            if neg:
                bg.append(robust_std(norm.windows(neg)))
            if pos:
                pk.append(np.abs(norm.windows(pos)).max())

        s1, _, s2 = spread(bg)
        p1, _, p2 = spread(pk)
        extra = f"  (fixed scale {norm.fixed_scale:.4g})" if norm.fixed_scale else ""
        print(f"{v:8s} {s1:15.1f} {s2:9.1f} | {p1:19.1f} {p2:9.1f}{extra}")

    print()
    print("scale tracks (N2), all waveforms; scale@1h = end of warm-up, level = median block level of the day")
    print(f"  {'waveform':10s} {'elev%':>7s} {'skip%':>6s} {'scale@1h':>10s} {'scale@end':>10s} "
          f"{'day level':>10s} {'level/end':>10s}")
    elev_share = []
    for s in streams:
        t = cache.get(s.waveform_id)
        i1h = min(len(t.scale) - 1, int(3600 * fs / t.block_len))
        lv = t.level
        day = np.nanmedian(lv) if np.isfinite(lv).any() else float("nan")
        elev_share.append(t.elevated.mean())
        print(f"  {s.waveform_id:10s} {100 * t.elevated.mean():7.1f} {100 * t.skipped.mean():6.1f} "
              f"{t.scale[i1h]:10.3g} {t.scale[-1]:10.3g} {day:10.3g} {day / t.scale[-1]:10.2f}")
    gs = np.array(elev_share) * 100
    print(f"  elevated share: median {np.median(gs):.1f} %, p90 {np.percentile(gs, 90):.1f} %, "
          f">30 % in {(gs > 30).sum()} of {len(gs)} waveforms")

    print()
    print("features on N2 windows (median of positives vs background, all waveforms)")
    norm = Normalizer(NormConfig("N2"), cache)
    pos = [w for w in bundle.windows if w.label == 1]
    neg = [w for w in bundle.windows if w.label == 0][::200]
    fp = np.median(extract_features(norm.windows(pos), fs), axis=0)
    fn = np.median(extract_features(norm.windows(neg), fs), axis=0)
    for name, p, n in zip(FEATURE_NAMES, fp, fn):
        print(f"  {name:9s} positive {p:10.4g}   background {n:10.4g}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = bundle.eval_stream(a.fig)
        t = cache.get(a.fig)
        hours = np.arange(len(s.data)) / fs / 3600

        n0 = Normalizer(NormConfig("N0")).fit_streams(
            bundle.eval_streams(split.train_events)).stream(s)
        n2 = Normalizer(NormConfig("N2"), cache).stream(s)
        n3 = Normalizer(NormConfig("N3"), cache).stream(s)

        fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        ax[0].plot(hours, s.data, lw=0.3)
        ax[0].plot(hours, 3 * t.per_sample(), "r", lw=1, label="3 x scale (N2)")
        ax[0].plot(hours, -3 * t.per_sample(), "r", lw=1)
        ax[0].legend()
        for axis, y, title in zip(ax[1:], (n0, n2, n3), ("N0", "N2", "N3")):
            axis.plot(hours, y, lw=0.3)
            axis.set_ylabel(title)
        for e in s.events:
            for axis in ax:
                axis.axvline(e.arrival_s / 3600, color="g", alpha=0.5)
        ax[-1].set_xlabel("hours")
        out = f"normalization_{a.fig}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()