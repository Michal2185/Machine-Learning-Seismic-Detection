"""
Marked time vs real arrival time on full waveforms.

    python -m tests.visualize_alarms                                 # 3 random waveforms, 3 detectors
    python -m tests.visualize_alarms --n 5 --seed 7                  # 5 other random waveforms
    python -m tests.visualize_alarms --ids evid00003 evid00004 evid00002
    python -m tests.visualize_alarms --detectors rfc-medium-N0 dnn-1k-N0 dnn-10k-N2 --budget 0.25

For every chosen waveform (a full 24 h file) the detectors are fitted on the TRAINING part of the fold in which
that waveform is a TEST waveform; the threshold is selected on the validation streams of that fold at the false
alarm budget (exactly as in cross_validate_budgets); then the waveform is scored. Seed 0 of each detector.

Written to OUT_DIR:
  <waveform>_overview.png   whole day: 60 s RMS envelope with catalogue arrival (green), visible onset (red dashed),
                            detect interval (green band), ignore interval (grey band), gaps; one row per detector
                            with its score, threshold and alarms (green = detection, light green = duplicate,
                            grey = neutral, red = false alarm)
  <waveform>_zoom_<evid>.png  -20 min .. +40 min around each catalogue arrival, with the alarm delay in the title
  delay_summary.png         all events of the chosen waveforms: alarm time minus catalogue arrival per detector
                            (missed events at the bottom)
  confusion.png             per detector, pooled over the chosen waveforms: window-level confusion matrix
                            (event windows vs background windows, score >= threshold) and event-level one
                            (detected / missed / false alarms; no TN exists for a continuous stream)
  confusion_metrics.csv     the same numbers per waveform and pooled: TP FP FN TN, recall, false negative rate,
                            specificity, FPR, precision, F1, balanced accuracy, MCC; event recall, episode
                            precision, false alarms per day

Window level: labels 1 / 0 only (label -1 windows dropped); every detector is scored on the SAME windows.
The threshold was chosen for the event-level false-alarm budget, so window-level recall is expected to be lower.

Detector names: dnn-<1k|10k|100k>-<N0..N3>, cnn-<1k|10k|100k>-<N0..N3>, rfc-<small|medium>-<N0..N3>,
amp-<rms|ptp>-<N0|N2> (the no-learning control).
The alarm rule is re-implemented in src/modules/reporting/alarms.py for plotting; the printed 'evaluator' line
is the authoritative number for the same waveform.
Paths: edit the constants once (same values as in the other scripts).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from src.modules.reporting import (
    classify_alarms,
    event_summary,
    extract_alarms,
    pool_event_summary,
    window_confusion,
)

# ---- EDIT THESE ONCE ----------------------------------------------------------
CATALOG_PATH = "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
DATA_DIR = "data/lunar/data/S12_GradeA/"
DERIVED_DIR = "data/lunar/derived/"
OUT_DIR = "figures/alarms"
# -------------------------------------------------------------------------------

DEFAULT_DETECTORS = ["rfc-medium-N0", "dnn-1k-N0", "dnn-10k-N2"]
LABEL_STYLE = {
    "detection": dict(color="tab:green", marker="v", label="detection (first alarm in detect interval)"),
    "duplicate": dict(color="lightgreen", marker="v", label="duplicate in detect interval"),
    "neutral": dict(color="0.55", marker="v", label="neutral (ignore interval / gap)"),
    "false": dict(color="tab:red", marker="v", label="false alarm"),
}


# ------------------------------------------------------------------ plotting (plain data in, PNG out)

def rms_envelope(data, fs, win_s):
    n = max(1, int(round(win_s * fs)))
    m = len(data) // n
    x = np.asarray(data[: m * n], dtype=np.float64).reshape(m, n)
    return (np.arange(m) + 0.5) * win_s, np.sqrt((x ** 2).mean(axis=1))


def _shade(ax, events, gaps, scale):

    for e in events:
        ax.axvspan(e.ignore[0] / scale, e.ignore[1] / scale, color="0.5", alpha=0.12, lw=0)
        ax.axvspan(e.detect[0] / scale, e.detect[1] / scale, color="tab:green", alpha=0.25, lw=0)
        ax.axvline(e.arrival_s / scale, color="tab:green", lw=1.2)
        if e.onset_s is not None:
            ax.axvline(e.onset_s / scale, color="tab:red", ls="--", lw=1.0)

    for g0, g1 in gaps:
        ax.axvspan(g0 / scale, g1 / scale, color="k", alpha=0.15, lw=0)


def _alarm_markers(ax, rep, thr, scale, size=7):

    for lab, st in LABEL_STYLE.items():
        t = [a for a, l in zip(rep.alarm_s, rep.labels) if l == lab]
        if t:
            ax.plot(np.array(t) / scale, np.full(len(t), thr), linestyle="none", marker=st["marker"],
                    color=st["color"], ms=size, mec="k", mew=0.4, label=st["label"], zorder=5)


def _summary_text(rep):
    return f"{rep.n_detected}/{len(rep.outcomes)} detected, {rep.n_false} false alarms ({rep.fa_per_day:.2f}/day)"


def plot_overview(wd, out_path):
    """wd: dict(waveform_id, data, fs, events, gaps, det: {name: dict(t_end, scores, thr, report)})."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(wd["det"])
    fig, ax = plt.subplots(1 + len(names), 1, figsize=(14, 2.2 + 2.0 * len(names) + 1.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.4] + [1.0] * len(names)))
    ax = np.atleast_1d(ax)

    t, env = rms_envelope(wd["data"], wd["fs"], 60.0)
    ax[0].semilogy(t / 3600, np.where(env > 0, env, np.nan), color="0.3", lw=0.6)
    _shade(ax[0], wd["events"], wd["gaps"], 3600.0)
    ax[0].set_ylabel("60 s RMS")
    ax[0].set_title(f"{wd['waveform_id']}: green line = catalogue arrival, red dashed = visible onset, "
                    f"green band = detect interval, grey band = ignore interval")

    for a, name in zip(ax[1:], names):
        d = wd["det"][name]
        a.plot(d["t_end"] / 3600, d["scores"], color="tab:blue", lw=0.6)
        a.axhline(d["thr"], color="k", ls=":", lw=1)
        _shade(a, wd["events"], wd["gaps"], 3600.0)
        _alarm_markers(a, d["report"], d["thr"], 3600.0)
        a.set_ylabel(name, fontsize=9)
        a.text(0.005, 0.92, _summary_text(d["report"]), transform=a.transAxes, va="top", fontsize=9,
               bbox=dict(fc="w", ec="0.7", alpha=0.9))

    h, l = ax[1].get_legend_handles_labels()
    for a in ax[2:]:
        h2, l2 = a.get_legend_handles_labels()
        for x, y in zip(h2, l2):
            if y not in l:
                h.append(x), l.append(y)
    ax[1].legend(h, l, loc="upper right", fontsize=7, ncol=2)
    ax[-1].set_xlabel("hours since file start")
    ax[-1].set_xlim(0, len(wd["data"]) / wd["fs"] / 3600)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_zoom(wd, event, out_path, before_s=1200.0, after_s=2400.0):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(wd["det"])
    lo, hi = event.arrival_s - before_s, event.arrival_s + after_s
    ref = event.arrival_s

    fig, ax = plt.subplots(1 + len(names), 1, figsize=(13, 2.0 + 2.0 * len(names) + 1.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3] + [1.0] * len(names)))
    ax = np.atleast_1d(ax)

    t, env = rms_envelope(wd["data"], wd["fs"], 20.0)
    m = (t >= lo) & (t <= hi)
    ax[0].semilogy((t[m] - ref) / 60, np.where(env[m] > 0, env[m], np.nan), color="0.3", lw=0.8)
    ax[0].set_ylabel("20 s RMS")
    ax[0].set_title(f"{wd['waveform_id']} / {event.evid}: delay of the first alarm in the detect interval "
                    f"(catalogue arrival = 0)")

    for a in ax:
        a.axvspan((event.detect[0] - ref) / 60, (event.detect[1] - ref) / 60, color="tab:green", alpha=0.2, lw=0)
        a.axvline(0, color="tab:green", lw=1.3)
        if event.onset_s is not None:
            a.axvline((event.onset_s - ref) / 60, color="tab:red", ls="--", lw=1.0)

    for a, name in zip(ax[1:], names):
        d = wd["det"][name]
        mm = (d["t_end"] >= lo) & (d["t_end"] <= hi)
        a.plot((d["t_end"][mm] - ref) / 60, d["scores"][mm], color="tab:blue", lw=0.9)
        a.axhline(d["thr"], color="k", ls=":", lw=1)

        rep = d["report"]
        sel = [(x, l) for x, l in zip(rep.alarm_s, rep.labels) if lo <= x <= hi]
        for lab, st in LABEL_STYLE.items():
            tt = [(x - ref) / 60 for x, l in sel if l == lab]
            if tt:
                a.plot(tt, np.full(len(tt), d["thr"]), linestyle="none", marker=st["marker"], color=st["color"],
                       ms=8, mec="k", mew=0.4, zorder=5)

        o = [o for o in rep.outcomes if o.evid == event.evid][0]
        if o.detected:
            txt = f"alarm {o.delay_arrival_s / 60:+.1f} min vs catalogue"
            if o.delay_onset_s is not None:
                txt += f", {o.delay_onset_s / 60:+.1f} min vs visible onset"
        else:
            txt = "MISSED (no alarm in detect interval)"
        a.set_ylabel(name, fontsize=9)
        a.text(0.005, 0.92, txt, transform=a.transAxes, va="top", fontsize=9,
               bbox=dict(fc="w", ec="0.7", alpha=0.9))

    ax[-1].set_xlabel("minutes from catalogue arrival")
    ax[-1].set_xlim(-before_s / 60, after_s / 60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_delay_summary(waveforms, names, out_path):
    """Alarm time minus catalogue arrival for every event of the chosen waveforms, per detector."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = [(wd["waveform_id"], e) for wd in waveforms for e in wd["events"]]
    x = np.arange(len(events))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    w = 0.8 / max(1, len(names))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(events) + 3), 5))
    ax.axhspan(-2, 10, color="tab:green", alpha=0.12, lw=0, label="detect interval (-2 .. +10 min)")
    ax.axhline(0, color="tab:green", lw=1.2)
    miss_y = -3.2

    for j, name in enumerate(names):
        for i, (wid, e) in enumerate(events):
            wd = next(w_ for w_ in waveforms if w_["waveform_id"] == wid)
            o = [o for o in wd["det"][name]["report"].outcomes if o.evid == e.evid][0]
            xx = x[i] - 0.4 + w * (j + 0.5)
            if o.detected:
                ax.plot(xx, o.delay_arrival_s / 60, "o", color=colors[j], ms=9, mec="k",
                        label=name if i == 0 else None)
                if o.onset_s is not None:
                    ax.plot(xx, o.delay_onset_s / 60, "_", color=colors[j], ms=14, mew=2)
            else:
                ax.plot(xx, miss_y, "x", color=colors[j], ms=10, mew=2, label=name if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e.evid}\n{wid}" if wid != e.evid else e.evid for wid, e in events], fontsize=8)
    ax.set_yticks([miss_y, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticklabels(["missed", "-2", "0", "2", "4", "6", "8", "10"])
    ax.set_ylim(-4, 11)
    ax.set_ylabel("first alarm minus catalogue arrival [min]\n(dot) and minus visible onset (dash)")
    ax.set_title("Marked time (alarm) vs real arrival time (catalogue)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def plot_confusion(names, window_c, event_c, out_path):
    """window_c: name -> Confusion; event_c: name -> pooled event summary dict."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(names), 2, figsize=(10.5, 3.6 * len(names)), squeeze=False)

    for i, name in enumerate(names):

        c = window_c[name]
        m = np.array([[c.tp, c.fn], [c.fp, c.tn]], dtype=float)
        rate = m / np.maximum(m.sum(axis=1, keepdims=True), 1)

        a = ax[i, 0]
        a.imshow(rate, vmin=0, vmax=1, cmap="Blues")

        for r in range(2):
            for k in range(2):
                a.text(k, r, f"{int(m[r, k]):,}\n({100 * rate[r, k]:.2f}% of row)", ha="center", va="center",
                       fontsize=9, color="w" if rate[r, k] > 0.6 else "k")

        a.set_xticks([0, 1], ["alarm (score >= thr)", "no alarm"], fontsize=8)
        a.set_yticks([0, 1], ["event window", "background window"], fontsize=8)
        a.set_title(f"{name} - window level\nrecall {_fmt(c.recall)}  FNR {_fmt(c.fnr)}  spec. {_fmt(c.specificity, 4)}  "
                    f"prec. {_fmt(c.precision)}  F1 {_fmt(c.f1)}  MCC {_fmt(c.mcc)}", fontsize=8)

        e = event_c[name]
        a = ax[i, 1]
        ev_row = np.array([e["detected"], e["missed"]], dtype=float)
        a.imshow(np.array([ev_row / max(ev_row.sum(), 1), [0, 0]]), vmin=0, vmax=1, cmap="Greens")
        a.add_patch(plt.Rectangle((-0.5, 0.5), 2, 1, color="0.9"))
        frac = ev_row / max(ev_row.sum(), 1)
        a.text(0, 0, f"{e['detected']}\nTP detected", ha="center", va="center", fontsize=10,
               color="w" if frac[0] > 0.6 else "k")
        a.text(1, 0, f"{e['missed']}\nFN missed", ha="center", va="center", fontsize=10,
               color="w" if frac[1] > 0.6 else "k")
        a.text(0, 1, f"{e['false_alarms']}\nFP false alarms", ha="center", va="center", fontsize=10)
        a.text(1, 1, "TN\nn/a (stream)", ha="center", va="center", fontsize=9, color="0.4")
        a.set_xticks([0, 1], ["alarm", "no alarm"], fontsize=8)
        a.set_yticks([0, 1], ["event", "no event"], fontsize=8)
        a.set_title(f"{name} - event level\nrecall {_fmt(e['event_recall'])}  FNR {_fmt(e['event_fnr'])}  "
                    f"episode precision {_fmt(e['episode_precision'])}  FA/day {_fmt(e['fa_per_day'])}", fontsize=8)

    fig.suptitle("Confusion matrices at the validation-selected threshold (pooled over the chosen waveforms)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def confusion_tables(chosen, names, scored):
    """Returns (rows for the CSV, pooled window confusion per detector, pooled event summary per detector)."""

    rows, window_pooled, event_pooled = [], {}, {}

    for n in names:
        wcs, reps = [], []

        for wid in chosen:
            d = scored[wid][n]
            wc = window_confusion(d["labels"], d["scores"], d["thr"])
            wcs.append(wc)
            reps.append(d["report"])
            rows.append(dict(level="window", scope=wid, detector=n, **wc.as_dict()))
            rows.append(dict(level="event", scope=wid, detector=n, **event_summary(d["report"])))

        window_pooled[n] = sum(wcs[1:], wcs[0])
        event_pooled[n] = pool_event_summary(reps)
        rows.append(dict(level="window", scope="pooled", detector=n, **window_pooled[n].as_dict()))
        rows.append(dict(level="event", scope="pooled", detector=n, **event_pooled[n]))

    return rows, window_pooled, event_pooled


def print_confusion_tables(names, window_pooled, event_pooled):

    print("\nWindow-level confusion, pooled over the chosen waveforms (same windows for every detector)")
    print(f"  {'detector':16s} {'windows':>8s} {'TP':>5s} {'FP':>6s} {'FN':>5s} {'TN':>9s} {'recall':>7s} {'FNR':>6s} "
          f"{'spec.':>7s} {'prec.':>6s} {'F1':>6s} {'bAcc':>6s} {'MCC':>6s}")
    for n in names:
        c = window_pooled[n]
        print(f"  {n:16s} {c.n:8d} {c.tp:5d} {c.fp:6d} {c.fn:5d} {c.tn:9d} {_fmt(c.recall):>7s} {_fmt(c.fnr):>6s} "
              f"{_fmt(c.specificity, 4):>7s} {_fmt(c.precision):>6s} {_fmt(c.f1):>6s} "
              f"{_fmt(c.balanced_accuracy):>6s} {_fmt(c.mcc):>6s}")

    print("\nEvent-level, pooled (TP = detected, FN = missed, FP = false alarm episodes; no TN for a stream)")
    print(f"  {'detector':16s} {'events':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'FA/day':>7s} {'recall':>7s} "
          f"{'FNR':>6s} {'ep.prec':>8s}")
    for n in names:
        e = event_pooled[n]
        print(f"  {n:16s} {e['events']:6d} {e['detected']:4d} {e['missed']:4d} {e['false_alarms']:4d} "
              f"{_fmt(e['fa_per_day']):>7s} {_fmt(e['event_recall']):>7s} {_fmt(e['event_fnr']):>6s} "
              f"{_fmt(e['episode_precision']):>8s}")


def write_csv(rows, path):

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ orchestration

def make_detector(spec, cache, fs, seed):

    from src.modules.models import (
        AmplitudeDetector,
        build_cnn_detector,
        build_dnn_detector,
        build_rfc_detector,
    )
    from src.modules.normalization import NormConfig, Normalizer
    from src.modules.training import TrainConfig

    kind, size, norm = spec.split("-")
    normalizer = Normalizer(NormConfig(norm), cache if norm in ("N2", "N3") else None)

    if kind == "dnn":
        return build_dnn_detector(spec, normalizer, fs, size, TrainConfig(seed=seed))

    if kind == "cnn":
        return build_cnn_detector(spec, normalizer, size, TrainConfig(seed=seed))

    if kind == "rfc":
        return build_rfc_detector(spec, normalizer, fs, size, seed=seed)

    if kind == "amp":
        return AmplitudeDetector(spec, size, normalizer, fs)

    raise ValueError(f"unknown detector {spec!r}")


def make_evaluator():
    """Same construction as tests/run_dnn_features.py (worked there)."""

    from src.modules.evaluation.module import Evaluator
    from src.modules.evaluation.types import EvalConfig

    return Evaluator(EvalConfig())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None, help="waveform ids (default: random)")
    ap.add_argument("--n", type=int, default=3, help="number of random waveforms")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random choice")
    ap.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    ap.add_argument("--budget", type=float, default=0.25, help="false alarms per day for the threshold")
    ap.add_argument("--detector-seed", type=int, default=0)
    ap.add_argument("--catalog", default=CATALOG_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--derived-dir", default=DERIVED_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    from src.modules.evaluation.module import build_series
    from src.modules.normalization import NormConfig, ScaleCache, end_sample_of
    from src.modules.pipeline import DataPipeline, PipelineConfig

    bundle = DataPipeline(PipelineConfig(a.catalog, a.data_dir, a.derived_dir)).build()
    all_ids = sorted(bundle._processed_by_id)
    streams = {w: bundle.eval_stream(w) for w in all_ids}
    fs = next(iter(streams.values())).sampling_rate

    if a.ids:
        chosen = list(a.ids)
    else:
        chosen = sorted(np.random.default_rng(a.seed).choice(all_ids, size=a.n, replace=False).tolist())

    print(f"waveforms: {chosen}")

    cache = ScaleCache.from_streams(list(streams.values()), NormConfig("N2"))
    evaluator = make_evaluator()
    refractory = float(getattr(evaluator.config, "refractory_s", 300.0))

    fold_of = {}
    for k in range(bundle.folds.n_folds):
        for w in bundle.fold_split(k).test_events:
            fold_of[w] = k

    scored = {wid: {} for wid in chosen}

    for spec in a.detectors:
        for k in sorted({fold_of[w] for w in chosen}):

            ids_k = [w for w in chosen if fold_of[w] == k]
            split = bundle.fold_split(k)
            det = make_detector(spec, cache, fs, a.detector_seed)
            print(f"[{spec}] fold {k}: fitting, waveforms {ids_k}")
            det.fit(split.train)

            val_streams = bundle.eval_streams(split.validation_events)
            val_by_id = {s.waveform_id: s for s in val_streams}
            val_series = build_series(split.validation, det.score(split.validation, val_by_id))
            curve = evaluator.curve(val_streams, val_series)
            thr = evaluator.select_threshold(curve, max_fa_per_day=a.budget, group="all")
            print(f"[{spec}] fold {k}: threshold {thr:.4g} at {a.budget:g} FA/day (validation)")

            for wid in ids_k:
                wins = sorted((w for w in split.test if w.event.evid == wid), key=end_sample_of)
                s = streams[wid]
                scores = np.asarray(det.score(wins, {wid: s}), dtype=float)
                t_end = np.array([end_sample_of(w) / fs for w in wins])
                duration = len(s.data) / fs

                rep = classify_alarms(extract_alarms(t_end, scores, thr, refractory), s.events, s.gaps, duration)

                line = ""
                try:
                    res = evaluator.evaluate([s], build_series(wins, scores), thr)
                    line = f"evaluator: recall {res.recall('all'):.2f}, FA/day {res.fa_per_day:.2f}"
                except Exception as e:                      # cross-check only
                    line = f"evaluator cross-check skipped ({type(e).__name__}: {e})"

                print(f"  {wid}: plotted rule: {rep.n_detected}/{len(rep.outcomes)} detected, "
                      f"{rep.n_false} false alarms ({rep.fa_per_day:.2f}/day) | {line}")

                scored[wid][spec] = dict(t_end=t_end, scores=scores, thr=float(thr), report=rep,
                                         labels=np.array([w.label for w in wins]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    waveforms = []

    for wid in chosen:
        s = streams[wid]
        wd = dict(waveform_id=wid, data=s.data, fs=fs, events=s.events, gaps=s.gaps,
                  det={n: scored[wid][n] for n in a.detectors})
        waveforms.append(wd)

        plot_overview(wd, out / f"{wid}_overview.png")

        for e in s.events:
            plot_zoom(wd, e, out / f"{wid}_zoom_{e.evid}.png")

        print(f"\n{wid}")
        for e in s.events:
            print(f"  {e.evid} ({e.event_type}) catalogue {e.arrival_s / 3600:.3f} h"
                  + (f", visible onset {(e.onset_s - e.arrival_s) / 60:+.1f} min" if e.onset_s is not None
                     else ", no usable visible onset"))
            for n in a.detectors:
                o = [o for o in wd["det"][n]["report"].outcomes if o.evid == e.evid][0]
                msg = ("missed" if not o.detected else
                       f"alarm {o.delay_arrival_s / 60:+.1f} min vs catalogue"
                       + (f", {o.delay_onset_s / 60:+.1f} min vs onset" if o.delay_onset_s is not None else ""))
                print(f"    {n:16s} {msg}")

    plot_delay_summary(waveforms, a.detectors, out / "delay_summary.png")

    rows, window_pooled, event_pooled = confusion_tables(chosen, a.detectors, scored)
    print_confusion_tables(a.detectors, window_pooled, event_pooled)
    plot_confusion(a.detectors, window_pooled, event_pooled, out / "confusion.png")
    write_csv(rows, out / "confusion_metrics.csv")

    print(f"\nfigures and confusion_metrics.csv written to {out}")


if __name__ == "__main__":
    main()