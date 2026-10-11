"""
Visualise the frozen STA/LTA on one or more waveforms.

For every waveform two figures are drawn:
  1. the whole day: filtered waveform, STA/LTA ratio with the two
     thresholds, alarms of both operating points
  2. a zoom around each catalogue event

What is drawn
  red dashed line   catalogue arrival (the "real" arrival)
  green line        visible onset from the zone table (if the event has one)
  orange band       detect interval: an alarm inside it counts as a detection
  grey band         ignore zone (onset region + coda): alarms there are neutral
  yellow band       interpolated gap (missing data)
  alarm markers     green = detects an event, red = false alarm,
                    grey = neutral (repeat in a coda / inside a gap)
                    top row = lenient, second row = strict

Edit IDS below (primary waveform ids such as "evid00002"), or set
AUTO_PICK = True to get a detected event, a missed event and a waveform
with false alarms.
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.detectors import FrozenSpec
from src.modules.evaluation import (
    EvalConfig,
    Evaluator,
    build_series,
    contains,
    merge_intervals,
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

IDS = ["evid00002", "evid00009", "evid00003"]
AUTO_PICK = False

ZOOM_BEFORE_S = 1800.0           # 30 min before the catalogue arrival
ZOOM_AFTER_S = 3 * 3600.0        # 3 h after

SAVE_DIR = None                  # e.g. "F:/Git/Machine-Learning-Seismic-Detection/figures/"
SHOW = True

CONFIG = PipelineConfig(
    catalog_path=(
        "F:/Git/Machine-Learning-Seismic-Detection/"
        "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
    ),
    data_dir=(
        "F:/Git/Machine-Learning-Seismic-Detection/"
        "data/lunar/data/S12_GradeA/"
    ),
    derived_dir=(
        "F:/Git/Machine-Learning-Seismic-Detection/"
        "data/lunar/derived/"
    ),
)

FROZEN_PATH = CONFIG.derived_dir + "baselines/sta_lta_frozen.json"

COLORS = {"detect": "#2ca02c", "false": "#d62728", "neutral": "#7f7f7f"}
ROW = {"lenient": 0.93, "strict": 0.82}        # marker rows (axes fraction)


# ------------------------------------------------------------
# Computation
# ------------------------------------------------------------

def classify(alarms, stream):
    """detect / neutral / false for every alarm time."""

    alarms = np.asarray(alarms, dtype=float)

    detect = merge_intervals([t.detect for t in stream.events])
    neutral = merge_intervals(
        [t.ignore for t in stream.events] + list(stream.gaps)
    )

    d = contains(detect, alarms)
    n = contains(neutral, alarms) & ~d

    return np.where(d, "detect", np.where(n, "neutral", "false"))


def analyse(wid, bundle, spec, evaluator, windows_by_wid):
    """Scores, alarms and event outcomes of one waveform."""

    stream = bundle.eval_stream(wid)
    ws = windows_by_wid[wid]

    scores = spec.detector().score(ws, {wid: stream})
    series = build_series(ws, scores)[wid]

    points = {}

    for name, p in spec.points.items():

        alarms = evaluator.alarms(series, p.threshold)
        result = evaluator.score_stream(stream, series, p.threshold)

        points[name] = {
            "threshold": p.threshold,
            "alarms": alarms,
            "kinds": classify(alarms, stream),
            "outcomes": result.outcomes,
        }

    return stream, series, points


def status_text(points):
    """One line per event: what each operating point did."""

    lines = []

    names = list(points)

    for i, o in enumerate(points[names[0]]["outcomes"]):

        parts = []

        for n in names:
            oc = points[n]["outcomes"][i]
            parts.append(
                f"{n}: detected {oc.delay_s:+.0f} s" if oc.detected
                else f"{n}: MISSED"
            )

        lines.append(f"{o.evid} ({o.event_type}"
                     f"{'' if o.has_positive else ', weak'}): " + " | ".join(parts))

    return lines


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------

def draw(ax_w, ax_r, stream, series, points, t0, t1, unit_s, origin_s, xlabel):
    """Waveform on ax_w, STA/LTA ratio on ax_r, for the time range [t0, t1] s."""

    fs = stream.sampling_rate
    i0 = max(0, int(t0 * fs))
    i1 = min(len(stream.data), int(t1 * fs))

    x = np.asarray(stream.data[i0:i1], dtype=float)
    t = (np.arange(i0, i1) / fs - origin_s) / unit_s

    scale = np.median(np.abs(stream.data)) or 1.0

    step = max(1, len(x) // 200000)

    ax_w.plot(t[::step], x[::step] / scale, color="#1f4e79", linewidth=0.4)

    def X(sec):
        return (sec - origin_s) / unit_s

    for g0, g1 in stream.gaps:
        ax_w.axvspan(X(g0), X(g1), color="#ffe066", alpha=0.6, lw=0)
        ax_r.axvspan(X(g0), X(g1), color="#ffe066", alpha=0.6, lw=0)

    for tg in stream.events:
        for ax in (ax_w, ax_r):
            ax.axvspan(X(tg.ignore[0]), X(tg.ignore[1]), color="#bbbbbb", alpha=0.25, lw=0)
            ax.axvspan(X(tg.detect[0]), X(tg.detect[1]), color="#ff9f40", alpha=0.30, lw=0)
            ax.axvline(X(tg.arrival_s), color="red", linestyle="--", linewidth=1.3)
            if tg.onset_s is not None:
                ax.axvline(X(tg.onset_s), color="green", linewidth=1.1)

    # alarms
    trans = ax_w.get_xaxis_transform()

    for name, row in ROW.items():
        if name in points:
            ax_w.text(0.004, row, name, transform=ax_w.transAxes, fontsize=8,
                      va="center", color="#333333")

    for name, p in points.items():
        for kind in ("detect", "neutral", "false"):
            sel = p["kinds"] == kind
            if sel.any():
                ax_w.scatter(
                    X(p["alarms"][sel]), np.full(sel.sum(), ROW[name]),
                    transform=trans, marker="v", s=46, color=COLORS[kind],
                    edgecolor="black", linewidth=0.5, zorder=5, clip_on=True,
                )

    # ratio
    m = (series.times >= t0) & (series.times <= t1)

    ax_r.semilogy(
        X(series.times[m]), np.maximum(series.scores[m], 0.1),
        color="#444444", linewidth=0.8,
    )

    styles = {"strict": ("#8b0000", "--"), "lenient": ("#e08a00", ":")}

    for name, p in points.items():
        c, ls = styles.get(name, ("black", "-"))
        ax_r.axhline(p["threshold"], color=c, linestyle=ls, linewidth=1.3)

    top = max(
        [p["threshold"] for p in points.values()] + [series.scores[m].max() if m.any() else 1.0]
    )
    ax_r.set_ylim(0.3, top * 1.5)

    ax_w.set_xlim(X(t0), X(t1))
    ax_r.set_xlim(X(t0), X(t1))

    ax_w.set_ylabel("amplitude (x median |x|)")
    ax_r.set_ylabel("STA/LTA ratio")
    ax_r.set_xlabel(xlabel)

    ax_w.grid(alpha=0.2)
    ax_r.grid(alpha=0.2, which="both")

    # y room for the marker rows
    lo, hi = ax_w.get_ylim()
    ax_w.set_ylim(lo, hi + 0.45 * (hi - lo))


def legend_handles(points):

    h = [
        Line2D([0], [0], color="red", linestyle="--", label="catalogue arrival"),
        Line2D([0], [0], color="green", label="visible onset"),
        plt.Rectangle((0, 0), 1, 1, color="#ff9f40", alpha=0.4, label="detect interval"),
        plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.4, label="ignore zone"),
        plt.Rectangle((0, 0), 1, 1, color="#ffe066", alpha=0.7, label="interpolated gap"),
    ]

    for kind, label in (("detect", "alarm: detects event"),
                        ("false", "alarm: FALSE"),
                        ("neutral", "alarm: neutral")):
        h.append(Line2D([0], [0], marker="v", color="w", markerfacecolor=COLORS[kind],
                        markeredgecolor="black", markersize=9, label=label))

    for name, ls, c in (("strict", "--", "#8b0000"), ("lenient", ":", "#e08a00")):
        if name in points:
            h.append(Line2D([0], [0], color=c, linestyle=ls,
                            label=f"{name} threshold {points[name]['threshold']:.1f}"))

    return h


def render_waveform(bundle, spec, evaluator, wid, windows_by_wid, save_dir=None):
    """Draw the full-day figure and one zoom figure per event. Returns the figures."""

    stream, series, points = analyse(wid, bundle, spec, evaluator, windows_by_wid)

    duration = len(stream.data) / stream.sampling_rate
    title = "\n".join([f"{wid}  |  frozen STA/LTA {spec.sta_s:g} s / {spec.lta_s:g} s"]
                      + status_text(points))

    figures = []

    # 1. whole day
    fig, (ax_w, ax_r) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    draw(ax_w, ax_r, stream, series, points, 0.0, duration, 3600.0, 0.0,
         "hours since file start")
    ax_w.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles=legend_handles(points), loc="lower center", fontsize=8, ncol=6)
    figures.append((f"{wid}_day", fig))

    # 2. zoom per event
    for tg in stream.events:

        t0 = max(0.0, tg.arrival_s - ZOOM_BEFORE_S)
        t1 = min(duration, tg.arrival_s + ZOOM_AFTER_S)

        fig, (ax_w, ax_r) = plt.subplots(
            2, 1, figsize=(15, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        draw(ax_w, ax_r, stream, series, points, t0, t1, 60.0, tg.arrival_s,
             "minutes after the catalogue arrival")

        pr = "" if tg.peak_ratio is None else f" | peak/background {tg.peak_ratio:.1f}"
        onset = ("no usable visible onset" if tg.onset_s is None
                 else f"visible onset {tg.onset_s - tg.arrival_s:+.0f} s vs catalogue")

        ax_w.set_title(f"{tg.evid} ({tg.event_type}) zoom | {onset}{pr}",
                       fontsize=10, loc="left")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.legend(handles=legend_handles(points), loc="lower center", fontsize=8, ncol=6)
        figures.append((f"{tg.evid}_zoom", fig))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figures:
            fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=130)

    return figures


def auto_pick(bundle, spec, evaluator, windows_by_wid):
    """A detected event, a missed event (highest peak ratio) and a waveform with false alarms."""

    detected, missed, false_alarms = None, None, (None, 0)

    for wid in sorted(bundle.grouping.event_groups):

        _, _, points = analyse(wid, bundle, spec, evaluator, windows_by_wid)
        p = points["lenient"] if "lenient" in points else next(iter(points.values()))

        n_false = int((p["kinds"] == "false").sum())

        if n_false > false_alarms[1]:
            false_alarms = (wid, n_false)

        for o in p["outcomes"]:
            if o.detected and o.has_positive and detected is None:
                detected = wid
            if not o.detected and (missed is None or (o.peak_ratio or 0) > missed[1]):
                missed = (wid, o.peak_ratio or 0)

    picks = [detected, missed[0] if missed else None, false_alarms[0]]

    return [w for i, w in enumerate(picks) if w and w not in picks[:i]]


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

def main():

    spec = FrozenSpec.load(FROZEN_PATH)

    print("Frozen detector:", spec.describe())

    bundle = DataPipeline(CONFIG).build()
    evaluator = Evaluator(EvalConfig(refractory_s=300.0))

    windows_by_wid = defaultdict(list)

    for w in bundle.windows:
        windows_by_wid[w.event.evid].append(w)

    ids = auto_pick(bundle, spec, evaluator, windows_by_wid) if AUTO_PICK else IDS

    print("Waveforms:", ids)

    for wid in ids:

        if wid not in windows_by_wid:
            print(f"{wid}: not a primary waveform id (events sharing a day are "
                  f"drawn under the id of the first event of that day)")
            continue

        render_waveform(bundle, spec, evaluator, wid, windows_by_wid, SAVE_DIR)

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()