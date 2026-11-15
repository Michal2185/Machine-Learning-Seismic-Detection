"""
Full comparison of every model on everything done so far: Apollo in-domain, resources, stress tests, Mars zero-shot.

    python -m tests.compare_all --mars results/mars.json --out results/comparison

Sources (be explicit about them in any paper table):
  * Apollo in-domain and stress rows are TRANSCRIBED from docs/HANDOFF_MODELS.md (section 3) and docs/HANDOFF_STRESS.md
    (sections 5-6); this script does not recompute them. 95 % intervals exist only where the handoffs give them.
    CNN / DNN / RFC use the phase-1 training schedule, GRU / CfC the longer one (150 epochs); the CNN was not rerun on it.
  * Mars rows are computed from the JSON written by run/run_mars.py (audit = every fold model x every seed).
    Two catalogued events, about 1.15 h of scored background: counts, not rates; no significance statements.
Writes: fig1..fig6 (png), apollo_in_domain.csv, resources.csv, stress.csv, mars_summary.csv, scorecard.csv, COMPARISON.md
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- transcribed Apollo numbers
# name: (recall_strict, ci_lo, ci_hi, fa_strict, recall_lenient, fa_lenient, recall_weak, delay_onset_s, schedule, source)
NAN = float("nan")
APOLLO = {
    "CfC-E 1k":        (0.92, 0.86, 0.96, 0.15, 0.98, 0.39, 0.83, 38, "long", "STRESS s5"),
    "CfC-E dt=1 1k":   (0.86, 0.81, 0.92, 0.13, 0.98, 0.48, 0.75, 35, "long", "STRESS s5"),
    "GRU-E 1k":        (0.85, 0.78, 0.91, 0.20, 0.98, 0.40, 0.72, 46, "long", "STRESS s5"),
    "RFC":             (0.91, 0.86, 0.96, 0.16, 0.97, 0.26, 0.87, 47, "phase1", "STRESS s5"),
    "CNN 1k":          (0.85, 0.78, 0.91, 0.14, 0.99, 0.53, 0.74, 52, "phase1", "STRESS s5"),
    "GRU-W 1k":        (0.79, 0.70, 0.86, 0.23, 0.99, 0.60, 0.67, 52, "long", "STRESS s5"),
    "CfC-W 1k":        (0.74, 0.67, 0.82, 0.26, 0.94, 0.72, 0.58, 107, "long", "STRESS s5"),
    "Amp (RMS)":       (0.68, 0.57, 0.79, 0.31, 0.97, 0.87, 0.54, 51, "none", "STRESS s5"),
    "STA/LTA frozen":  (0.67, 0.56, 0.76, 0.30, 0.81, 0.43, 0.58, 112, "none", "STRESS s5 / BASELINE"),
    "CNN 10k":         (0.84, 0.78, 0.89, 0.18, 0.99, 0.43, NAN, NAN, "phase1", "MODELS s3"),
    "DNN 10k":         (0.82, 0.75, 0.89, 0.20, 0.99, 0.72, NAN, NAN, "phase1", "MODELS s3"),
    "DNN 1k":          (0.76, 0.68, 0.84, 0.12, 0.98, 0.57, NAN, NAN, "phase1", "MODELS s3"),
    "CfC-E 10k":       (0.89, 0.85, 0.93, NAN, NAN, NAN, NAN, NAN, "long", "STRESS s5"),
    "GRU-E 10k":       (0.78, NAN, NAN, NAN, NAN, NAN, NAN, NAN, "long", "STRESS s5"),
    "CfC-E dt=1 10k":  (0.81, NAN, NAN, NAN, NAN, NAN, NAN, NAN, "long", "STRESS s5"),
    "CNN 10k (N2 v3)": (0.50, 0.39, 0.60, 0.17, 0.73, 1.11, NAN, NAN, "phase1", "MODELS s3"),
    "DNN 10k (N2 v3)": (0.43, NAN, NAN, 0.19, 0.59, 0.81, NAN, NAN, "phase1", "MODELS s3"),
    "RFC (N2 v3)":     (0.30, NAN, NAN, 0.23, 0.73, 1.76, NAN, NAN, "phase1", "MODELS s3"),
}
# name: (params, model MACs or ops per window, input ops per window, int8 bytes approx, note)
RESOURCES = {
    "CfC-E 1k":      (950, 10621, 3242, 950 + 4 * 9, "4 Linear layers, hook-verified MACs"),
    "GRU-E 1k":      (961, 10275, 3242, 961 + 4 * 6, "params from the closed form 3(Fh+h^2+2h)+h+1"),
    "RFC":           (4280, 733, 12573, 29 * 1024, "4 280 nodes (about 29 KiB), 733 comparisons"),
    "CNN 1k":        (1025, 77384, 398, 1025 + 4 * 8, ""),
    "CNN 10k":       (9729, 624000, 398, 9729 + 4 * 8, ""),
    "DNN 1k":        (1037, 980, 12600, 1037 + 4 * 6, "front end about 12 600 ops"),
    "DNN 10k":       (10081, 9900, 12600, 10081 + 4 * 6, ""),
    "GRU-W 1k":      (1038, 365381, 398, 1038 + 4 * 6, ""),
    "CfC-W 1k":      (1036, 382095, 398, 1036 + 4 * 9, "params from the closed form 4h(F+h+1)+h+1"),
    "CfC-E 10k":     (None, 119856, 3242, None, ""),
    "GRU-E 10k":     (None, 116875, 3242, None, ""),
    "Amp (RMS)":     (1, 0, 1194, 8, "1 threshold"),
}
PERT = ["clean", "gain0.3", "gain3", "white 1", "matched 1", "matched 2", "gaps 10%", "jitter 0.25", "lowpass 1.3", "lowpass 0.8"]
STRESS_MODELS = ["CfC-E 1k", "GRU-E 1k", "RFC", "CNN 1k", "Amp (RMS)"]
# recall / FA-per-day, strict budget, frozen thresholds, perturbed test streams (HANDOFF_STRESS s6)
STRESS = {
    "clean":       [(0.92, 0.15), (0.85, 0.20), (0.91, 0.16), (0.85, 0.14), (0.68, 0.31)],
    "gain0.3":     [(0.31, 0.01), (0.30, 0.04), (0.35, 0.00), (0.27, 0.01), (0.24, 0.09)],
    "gain3":       [(0.98, 90.6), (1.00, 51.6), (0.98, 76.6), (1.00, 19.9), (0.88, 66.6)],
    "white 1":     [(0.92, 8.7), (0.87, 121.7), (0.92, 2.9), (0.85, 0.14), (0.68, 0.31)],
    "matched 1":   [(0.92, 0.19), (0.85, 0.26), (0.92, 0.38), (0.85, 0.14), (0.68, 0.31)],
    "matched 2":   [(0.93, 12.8), (0.90, 5.2), (0.93, 19.4), (0.85, 1.7), (0.69, 6.1)],
    "gaps 10%":    [(0.89, 0.14), (0.83, 0.19), (0.91, 0.15), (0.84, 0.13), (0.68, 0.30)],
    "jitter 0.25": [(0.90, 0.12), (0.83, 0.17), (0.90, 0.14), (0.82, 0.12), (0.67, 0.31)],
    "lowpass 1.3": [(0.92, 0.14), (0.83, 0.16), (0.92, 0.15), (0.84, 0.13), (0.68, 0.31)],
    "lowpass 0.8": [(0.66, 0.03), (0.48, 0.11), (0.60, 0.01), (0.51, 0.04), (0.56, 0.31)],
}

COLOR = {"CfC-E": "#1f77b4", "CfC-E dt=1": "#7fb2e5", "GRU-E": "#ff7f0e", "RFC": "#2ca02c", "CNN": "#d62728", "DNN": "#9467bd",
         "GRU-W": "#ffbb78", "CfC-W": "#aec7e8", "Amp": "#7f7f7f", "STA/LTA": "#000000"}


def color_of(name):
    for k in sorted(COLOR, key=len, reverse=True):
        if name.startswith(k):
            return COLOR[k]
    return "0.5"


MARS_MODELS = {"cfcE-1k-N0-long": "CfC-E 1k", "gruE-1k-N0-long": "GRU-E 1k", "cfc0E-1k-N0-long": "CfC-E dt=1 1k",
               "rfc-medium-N0": "RFC", "cnn-1k-N0": "CNN 1k", "amp-rms-N0": "Amp (RMS)", "amp-ptp-N0": "Amp (ptp)"}
SCALES = [0.5, 1.0, 2.0]
EVENTS = ["evid0005", "evid0006"]
BUDGETS = {0.25: "strict (0.25/day thresholds)", 2.0: "lenient (2/day thresholds)"}


# --------------------------------------------------------------------------- Mars summary
def mars_summary(path):

    d = json.load(open(path))
    rows = defaultdict(list)

    for a in d["audit"]:
        base, tag = a["config"].split("@")
        if base not in MARS_MODELS:
            continue
        arm, mult = tag.replace("mars-", "").rsplit("-x", 1)
        rows[(MARS_MODELS[base], arm, float(mult), a["budget"])].append(a)

    hours = float(np.mean([r["scored_hours"] for r in d["records"]]))
    out = []

    for (model, arm, mult, budget), rs in sorted(rows.items()):
        rec = dict(model=model, arm=arm, scale=mult, budget=budget, n_fold_models=len(rs), scored_hours=hours)
        for e in EVENTS:
            det = [r["events"][e][0] for r in rs]
            dl = [r["events"][e][1] for r in rs if r["events"][e][0] and r["events"][e][1] is not None]
            rec[f"det_{e}"] = float(np.mean(det))
            rec[f"delay_{e}"] = float(np.median(dl)) if dl else NAN
        rec["det_mean"] = float(np.mean([rec[f"det_{e}"] for e in EVENTS]))
        rec["fa_per_run"] = float(np.mean([r["n_false_alarms"] for r in rs]))
        out.append(rec)

    return out


def mget(S, model, arm, scale, budget, key):
    for r in S:
        if (r["model"], r["arm"], r["scale"], r["budget"]) == (model, arm, scale, budget):
            return r[key]
    return NAN


# --------------------------------------------------------------------------- figures
def fig1(out):

    import matplotlib.pyplot as plt

    names = [n for n in APOLLO if "N2" not in n]
    names = sorted(names, key=lambda n: APOLLO[n][0])
    n2 = [n for n in APOLLO if "N2" in n]
    names = n2 + names
    y = np.arange(len(names))

    fig, ax = plt.subplots(1, 4, figsize=(17, 6.5), sharey=True)

    def bar(a, idx, title, xlabel, xlim=None, ci=None, fmt="{:.2f}"):
        for i, n in enumerate(names):
            v = APOLLO[n][idx]
            if np.isnan(v):
                a.text(0.01 if xlim is None else xlim[0] + 0.01, i, "n/a", va="center", fontsize=7, color="0.5")
                continue
            a.barh(i, v, color=color_of(n), alpha=0.45 if "N2" in n else 0.9, hatch="//" if n.endswith(("W 1k")) else None)
            if ci is not None and not np.isnan(APOLLO[n][ci[0]]):
                a.errorbar(v, i, xerr=[[v - APOLLO[n][ci[0]]], [APOLLO[n][ci[1]] - v]], color="k", capsize=2, lw=1)
            a.text(v + (0.01 if xlim is None else 0.01 * (xlim[1] - xlim[0])), i - 0.28, fmt.format(v), va="center", fontsize=7)
        a.set_title(title, fontsize=10); a.set_xlabel(xlabel); a.grid(axis="x", alpha=0.3)
        if xlim: a.set_xlim(*xlim)

    bar(ax[0], 0, "Recall, strict budget (0.25 FA/day)", "recall, 95 % interval over waveforms", (0, 1.08), ci=(1, 2))
    ax[0].axvspan(0.56, 0.76, color="0.85", zorder=0); ax[0].axvline(0.67, color="k", ls="--", lw=0.8)
    bar(ax[1], 4, "Recall, lenient budget (2 FA/day)", "recall", (0, 1.12))
    ax[1].axvline(0.81, color="k", ls="--", lw=0.8)
    bar(ax[2], 6, "Recall on weak events, strict\n(no usable visible onset)", "recall", (0, 1.05))
    bar(ax[3], 7, "Median delay vs visible onset, strict", "seconds", (0, 135), fmt="{:.0f}")
    ax[0].set_yticks(y, names, fontsize=8)
    fig.suptitle("Fig 1. Apollo 12 in-domain, nested CV, N0 unless marked (dashed = STA/LTA; hatched = window variant W; "
                 "faded = N2 v3)\nrows without a value were not reported in the handoffs", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig1_apollo_in_domain.png", dpi=110)
    plt.close(fig)


def total_ops(n):
    p, m, i, b, _ = RESOURCES[n]
    return m + i


def fig2(out):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    for n in RESOURCES:
        if n not in APOLLO or np.isnan(APOLLO[n][0]):
            continue
        r, lo, hi = APOLLO[n][0], APOLLO[n][1], APOLLO[n][2]
        p, m, i, b, _ = RESOURCES[n]
        jit = {"CfC-E 1k": 0.90, "GRU-E 1k": 1.10, "CNN 1k": 0.90, "GRU-W 1k": 1.10, "DNN 1k": 0.90}.get(n, 1.0)
        off = {"CfC-E 1k": (-52, 6), "GRU-E 1k": (6, -12), "RFC": (6, 4), "CNN 1k": (-44, 6), "GRU-W 1k": (6, 4),
               "CfC-W 1k": (6, -12), "DNN 1k": (-44, -12), "DNN 10k": (6, -12)}.get(n, (6, 4))
        for a, x in ((ax[0], m + i), (ax[1], b)):
            if x is None or x == 0:
                continue
            xx = x * (jit if (a is ax[1] or n in ("CfC-E 1k", "GRU-E 1k")) else 1.0)
            a.errorbar(xx, r, yerr=None if np.isnan(lo) else [[r - lo], [hi - r]], fmt="o", color=color_of(n), ms=8,
                       capsize=2, mfc=color_of(n) if "W" not in n else "white", mew=1.5, alpha=0.9, elinewidth=0.8, ecolor=color_of(n))
            a.annotate(n, (xx, r), textcoords="offset points", xytext=off, fontsize=7.5)

    for a, t, xl in ((ax[0], "Accuracy vs compute", "operations per 60 s window: model MACs + input ops (log)"),
                     (ax[1], "Accuracy vs memory", "int8 model size in bytes (log; RFC about 29 KiB)")):
        a.set_xscale("log"); a.set_ylim(0.6, 1.0); a.axhline(0.67, color="k", ls="--", lw=0.8)
        a.axhspan(0.56, 0.76, color="0.88", zorder=0); a.text(a.get_xlim()[0] * 1.1, 0.675, "STA/LTA (a few running sums)", fontsize=7)
        a.set_title(t, fontsize=10); a.set_xlabel(xl); a.set_ylabel("recall, strict budget"); a.grid(alpha=0.3)

    fig.suptitle("Fig 2. Accuracy-resource trade-off (Apollo 12, strict 0.25 FA/day). Open markers = window variant W; E-variant, DNN and RFC "
                 "costs\ninclude the feature front end; points are jittered by +-10 % in x where they would overlap. The amplitude control (about 1.2k ops, 8 bytes) is the cheap reference.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "fig2_accuracy_resources.png", dpi=110)
    plt.close(fig)


def heat(a, M, rows, cols, title, fmt, cmap, norm=None, vmin=None, vmax=None, textcolor=None, T=None):
    T = M if T is None else T
    im = a.imshow(M, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    a.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8)
    a.set_yticks(range(len(rows)), rows, fontsize=8)
    a.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = T[i, j]
            a.text(j, i, "n/a" if np.isnan(v) else fmt(v), ha="center", va="center", fontsize=7.5,
                   color="k" if textcolor is None else textcolor(v))
    return im


def fig3(out):

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    R = np.array([[STRESS[p][m][0] for p in PERT] for m in range(5)])
    F = np.array([[STRESS[p][m][1] for p in PERT] for m in range(5)])
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    im = heat(ax[0], R, STRESS_MODELS, PERT, "Recall at the strict budget (frozen thresholds; meaningless where FA is huge)", lambda v: f"{v:.2f}",
              "RdYlGn", vmin=0.2, vmax=1.0)
    fig.colorbar(im, ax=ax[0], fraction=0.025)
    im = heat(ax[1], np.maximum(F, 1e-2), STRESS_MODELS, PERT, "False alarms per day (log colour; budget 0.25 = dashed line on the colour bar)",
              lambda v: f"{v:.2f}" if v < 10 else f"{v:.0f}", "RdYlGn_r", norm=LogNorm(vmin=0.05, vmax=150), T=F)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.025); cb.ax.axhline(0.25, color="k", ls="--")
    fig.suptitle("Fig 3. Stress tests, Apollo 12 test streams perturbed, thresholds frozen from the clean validation (5 seeds; amplitude control 1)\n"
                 "gain = calibration failure for every model; white noise breaks GRU-E (spectral shape), matched noise 2 = true SNR loss", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig3_stress.png", dpi=110)
    plt.close(fig)


def fig4(out, S):

    import matplotlib.pyplot as plt

    models = ["CfC-E 1k", "CfC-E dt=1 1k", "GRU-E 1k", "RFC", "CNN 1k", "Amp (RMS)", "Amp (ptp)"]
    cols = [f"x{s:g}\n{e[-4:]}" for s in SCALES for e in EVENTS]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    for r, arm in enumerate(("lp1.3", "nolp")):
        for c, b in enumerate(BUDGETS):
            M = np.array([[mget(S, m, arm, s, b, f"det_{e}") for s in SCALES for e in EVENTS] for m in models])
            im = heat(ax[r, c], M, models, cols, f"arm {arm}, {BUDGETS[b]}", lambda v: f"{v:.2f}", "RdYlGn", vmin=0, vmax=1)
            for k in (2, 4):
                ax[r, c].axvline(k - 0.5, color="k", lw=1.2)
    fig.subplots_adjust(hspace=0.5, wspace=0.18)
    fig.colorbar(im, ax=ax, fraction=0.02, label="share of fold models x seeds that detect the event")
    fig.suptitle("Fig 4. Mars zero-shot, detection share per event (evid0005 = low-frequency, evid0006 = 2.4 Hz-type), N0 recalibrated x0.5 / x1 / x2\n"
                 "25 fold-model runs per cell (amplitude controls: 5). Two events: descriptive only. STA/LTA: 0 of 2 everywhere (warm-up artefact, not shown).", fontsize=10)
    fig.savefig(out / "fig4_mars_detection.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig5(out, S):

    import matplotlib.pyplot as plt

    models = ["CfC-E 1k", "CfC-E dt=1 1k", "GRU-E 1k", "RFC", "CNN 1k", "Amp (RMS)"]
    fig, ax = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    row = 0

    for b in BUDGETS:
        for kind in ("det_mean", "fa_per_run"):
            for c, arm in enumerate(("lp1.3", "nolp")):
                a = ax[row, c]
                for m in models:
                    y = [mget(S, m, arm, s, b, kind) for s in SCALES]
                    a.plot(SCALES, y, "o-", color=color_of(m), label=m, lw=1.6)
                a.set_xscale("log", base=2); a.set_xticks(SCALES, [f"x{s:g}" for s in SCALES]); a.grid(alpha=0.3)
                if kind == "det_mean":
                    a.set_ylim(-0.03, 1.05); a.set_ylabel("mean detection share\n(2 events)")
                else:
                    a.set_yscale("symlog", linthresh=0.1); a.set_ylim(0, None); a.set_ylabel("false alarms per run\n(1.15 h scored, counts)")
                a.set_title(f"{arm}, {BUDGETS[b].split(' ')[0]}: " + ("detection" if kind == "det_mean" else "false alarms"), fontsize=9)
            row += 1

    ax[0, 0].legend(fontsize=7, loc="lower left")
    for a in ax[-1]:
        a.set_xlabel("N0 recalibration multiplier (x1 = pre-registered; x0.5 / x2 = sensitivity band)")
    fig.suptitle("Fig 5. Mars: the N0 scale constant dominates. A factor of 2 in the scale moves every model's detection and false-alarm counts a lot,\n"
                 "and changes the ranking; x1 is the pre-registered choice, x0.5 / x2 the sensitivity band (2 events, 1.15 h scored background)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig5_mars_scale_sensitivity.png", dpi=110)
    plt.close(fig)


def scorecard(S):

    names = ["CfC-E 1k", "GRU-E 1k", "RFC", "CNN 1k", "Amp (RMS)"]
    idx = {n: i for i, n in enumerate(STRESS_MODELS)}
    benign = ["gaps 10%", "jitter 0.25", "lowpass 1.3", "matched 1"]
    cols = [
        ("Apollo recall\nstrict", lambda n: APOLLO[n][0], 1, "{:.2f}"),
        ("Apollo recall\nweak events", lambda n: APOLLO[n][6], 1, "{:.2f}"),
        ("Apollo FA/day\nlenient", lambda n: APOLLO[n][5], -1, "{:.2f}"),
        ("ops / window\n(log)", lambda n: np.log10(total_ops(n)), -1, None),
        ("Stress: mean recall\ngaps/jitter/lp1.3/matched1", lambda n: np.mean([STRESS[p][idx[n]][0] for p in benign]), 1, "{:.2f}"),
        ("Stress: recall\nlowpass 0.8", lambda n: STRESS["lowpass 0.8"][idx[n]][0], 1, "{:.2f}"),
        ("Stress: FA/day\nmatched noise 2", lambda n: STRESS["matched 2"][idx[n]][1], -1, "{:.1f}"),
        ("Stress: FA/day\nwhite noise 1", lambda n: STRESS["white 1"][idx[n]][1], -1, "{:.1f}"),
        ("Stress: FA/day\ngain x3", lambda n: STRESS["gain3"][idx[n]][1], -1, "{:.0f}"),
        ("Mars (n=2) detection\nx1 lp1.3 strict", lambda n: mget(S, n, "lp1.3", 1.0, 0.25, "det_mean"), 1, "{:.2f}"),
        ("Mars: worst detection\nover x0.5/1/2", lambda n: min(mget(S, n, "lp1.3", s, 0.25, "det_mean") for s in SCALES), 1, "{:.2f}"),
        ("Mars FA/run\nnolp x1 strict", lambda n: mget(S, n, "nolp", 1.0, 0.25, "fa_per_run"), -1, "{:.2f}"),
        ("Mars: worst FA/run\nlp1.3 x0.5/1/2", lambda n: max(mget(S, n, "lp1.3", s, 0.25, "fa_per_run") for s in SCALES), -1, "{:.2f}"),
    ]
    V = np.array([[f(n) for _, f, _, _ in cols] for n in names], dtype=float)
    return names, cols, V


def fig6(out, S):

    import matplotlib.pyplot as plt

    names, cols, V = scorecard(S)
    G = np.zeros_like(V)

    for j, (_, _, sign, _) in enumerate(cols):
        v = V[:, j] * sign
        r = v.max() - v.min()
        G[:, j] = 0.5 if r == 0 else (v - v.min()) / r

    fig, ax = plt.subplots(figsize=(17, 4.6))
    ax.imshow(G, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for i in range(len(names)):
        for j, (_, _, _, f) in enumerate(cols):
            txt = f"{int(round(10 ** V[i, j])):,}" if f is None else f.format(V[i, j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if (G[i, j] < 0.1 or G[i, j] > 0.9) else "k")

    ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=7.5)
    ax.set_yticks(range(len(names)), names)
    ax.xaxis.tick_top()
    for x in (2.5, 3.5, 8.5):
        ax.axvline(x, color="k", lw=1.5)
    fig.suptitle("Fig 6. Scorecard across all phases (colour = relative rank within the column, green best; no overall score is computed)\n"
                 "Apollo | cost | stress | Mars. Mars columns rest on 2 events and 1.15 h scored background.", fontsize=10, y=0.01, va="bottom")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "fig6_scorecard.png", dpi=110)
    plt.close(fig)

    return names, cols, V


# --------------------------------------------------------------------------- tables
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def md_table(header, rows, fmt=None):
    def c(v):
        return "n/a" if isinstance(v, float) and np.isnan(v) else (f"{v:.3g}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(c(v) for v in r) + " |" for r in rows])


def main(argv=None):

    ap = argparse.ArgumentParser()
    ap.add_argument("--mars", default="results/mars.json")
    ap.add_argument("--out", default="results/comparison")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = mars_summary(a.mars)

    fig1(out); fig2(out); fig3(out); fig4(out, S); fig5(out, S)
    names, cols, V = fig6(out, S)

    write_csv(out / "apollo_in_domain.csv",
              ["model", "recall_strict", "ci_lo", "ci_hi", "fa_day_strict", "recall_lenient", "fa_day_lenient", "recall_weak_strict",
               "delay_vs_onset_s", "schedule", "source"], [[n, *v] for n, v in APOLLO.items()])
    write_csv(out / "resources.csv", ["model", "params", "model_macs_or_ops", "input_ops", "int8_bytes_approx", "note"],
              [[n, *v] for n, v in RESOURCES.items()])
    write_csv(out / "stress.csv", ["perturbation", "model", "recall_strict", "fa_per_day_strict"],
              [[p, STRESS_MODELS[m], *STRESS[p][m]] for p in PERT for m in range(5)])
    write_csv(out / "mars_summary.csv", list(S[0].keys()), [list(r.values()) for r in S])
    write_csv(out / "scorecard.csv", ["model"] + [c[0].replace("\n", " ") for c in cols],
              [[n, *[float(x) if cols[j][3] is not None else float(10 ** x) for j, x in enumerate(V[i])]] for i, n in enumerate(names)])

    main_rows = [n for n in APOLLO if "N2" not in n]
    md = ["# Comparison of all models on all results", "",
          "Generated by `tests/compare_all.py`. Apollo and stress rows are transcribed from the handoff notes; Mars rows come from the run JSON. "
          "Intervals exist only where the handoffs give them. Mars: 2 catalogued events, about 1.15 h scored background, descriptive only.", "",
          "## Apollo 12 in-domain (nested CV, N0)", "",
          md_table(["model", "recall strict [95 %]", "FA/day", "recall lenient", "FA/day", "weak", "delay (s)"],
                   [[n, (f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if not np.isnan(v[1]) else f"{v[0]:.2f}"), v[3], v[4], v[5], v[6], v[7]]
                    for n, v in APOLLO.items()]), "",
          "## Resources", "", md_table(["model", "params", "model MACs/ops", "input ops", "int8 bytes (approx)"],
                                       [[n, *(v[:4])] for n, v in RESOURCES.items()]), "",
          "## Stress tests (strict budget, frozen thresholds): recall / FA per day", "",
          md_table(["perturbation"] + STRESS_MODELS, [[p] + [f"{r:.2f} / {f:g}" for r, f in STRESS[p]] for p in PERT]), "",
          "## Mars zero-shot, strict thresholds, detection share (evid0005 / evid0006) and false alarms per run", ""]
    for arm in ("lp1.3", "nolp"):
        md += [f"### arm {arm}", "", md_table(["model"] + [f"x{s:g}" for s in SCALES],
              [[m] + [f"{mget(S, m, arm, s, 0.25, 'det_evid0005'):.2f} / {mget(S, m, arm, s, 0.25, 'det_evid0006'):.2f}, FA {mget(S, m, arm, s, 0.25, 'fa_per_run'):.2f}"
                      for s in SCALES] for m in ["CfC-E 1k", "CfC-E dt=1 1k", "GRU-E 1k", "RFC", "CNN 1k", "Amp (RMS)", "Amp (ptp)"]]), ""]
    md += ["All lenient-budget Mars numbers are in `mars_summary.csv`. Figures: fig1 to fig6.", ""]
    (out / "COMPARISON.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()