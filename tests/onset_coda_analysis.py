"""
Envelope diagnostic: how far is the VISIBLE onset from the catalogue
time, and how long is the coda?

Used to replace the placeholder LabelConfig values
(pre_onset_margin_s = 60, coda = 1200 s).

TRAINING waveforms only (same seed-42 split as the pipeline).
Data: causal highpass 0.2 Hz, no scaling (only ratios matter).

Definitions (all relative to the catalogue arrival)
---------------------------------------------------
envelope   : 20 s RMS of the filtered signal
b_loc      : median envelope in [arrival-2400 s, arrival-600 s]
             (local background: what a detector sees before the event)
b_ref      : median envelope of the whole day outside all event zones
             (typical background level of that day)
onset      : first time in [arrival-900, arrival+300] the envelope stays
             above k_on * b_loc for >= 30 s
             (negative offset = visible BEFORE the catalogue time)
peak       : maximum envelope within 3 h after the arrival
coda end   : first time after the peak the envelope stays below
             k_end * b_ref for >= 600 s
censored   : the envelope never fell below the threshold before the end
             of the file / the next event -> duration is a LOWER BOUND
dropout    : envelope below DROPOUT_REL * b_ref (zero-filled / missing
             data). Dropouts carry no information: they are excluded from
             the background levels and do NOT count as "coda has ended".
tau        : decay time constant of the coda, assuming an exponential
             decay from the peak:
             coda_end - peak_delay = tau * ln(peak_ratio / k_end)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


ENV_SEC = 20

K_ON = (2.0, 3.0, 5.0)
K_END = (1.5, 2.0, 3.0, 5.0)
K_ON_DEFAULT = 3.0
K_END_DEFAULT = 2.0

ONSET_SEARCH = (-900, 300)
ONSET_MIN_DUR = 30
LOCAL_REF = (-2400, -600)
PEAK_SEARCH = 3 * 3600
CODA_MIN_BELOW = 600
EVENT_ZONE = (-600, 6 * 3600)

DROPOUT_REL = 0.05
SIGNAL_SEC = 20          # seconds of visible signal a window needs inside

N_PLOT = 6


# ---- ANALYSIS FUNCTIONS (no pipeline imports) ----------------------

def first_sustained(mask, n):
    """Index of the first run of >= n consecutive True values, else None."""
    mask = np.asarray(mask, dtype=bool)
    if n <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else None
    if len(mask) < n:
        return None
    cs = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    idx = np.flatnonzero(cs[n:] - cs[:-n] == n)
    return int(idx[0]) if len(idx) else None


def envelope(x, fs):
    x = np.asarray(x, dtype=np.float64)
    m = uniform_filter1d(x * x, size=max(1, int(ENV_SEC * fs)), mode="nearest")
    # running sums can leave tiny NEGATIVE residuals in zero-filled
    # stretches -> sqrt would give NaN
    return np.sqrt(np.maximum(m, 0.0))


def background_level(values):
    """Median of the non-dropout values (iterated, so a large share of
    dropouts does not drag the level down)."""
    level = float(np.median(values))
    for _ in range(3):
        good = values[values > DROPOUT_REL * level]
        if len(good) == 0:
            break
        level = float(np.median(good))
    return level


def day_reference(env, fs, arrivals):
    """Background level outside all event zones, ignoring dropouts."""
    mask = np.ones(len(env), dtype=bool)
    for arr in arrivals:
        a = max(0, int(round((arr + EVENT_ZONE[0]) * fs)))
        b = max(0, int(round((arr + EVENT_ZONE[1]) * fs)))
        mask[a:b] = False
    if mask.sum() / fs >= 3600:
        return background_level(env[mask]), False
    return background_level(env), True


def trailing_dropout_seconds(env, fs, b_ref):
    flags = env <= DROPOUT_REL * b_ref
    if not flags[-1]:
        return 0.0
    ok = np.flatnonzero(~flags)
    return float((len(env) - 1 - ok[-1]) / fs) if len(ok) else len(env) / fs


def analyse_event(env, fs, arrival, next_arrival, b_ref):

    n = len(env)

    def I(s):
        return int(round(s * fs))

    limit = n if next_arrival is None else min(n, I(next_arrival - 60))
    a = min(max(I(arrival), 0), n - 1)

    # local background
    l0 = max(0, I(arrival + LOCAL_REF[0]))
    l1 = max(0, I(arrival + LOCAL_REF[1]))
    loc = env[l0:l1]
    loc = loc[loc > DROPOUT_REL * b_ref]
    b_loc = float(np.median(loc)) if len(loc) >= I(600) else b_ref

    # onset
    s0 = max(0, I(arrival + ONSET_SEARCH[0]))
    s1 = min(n, I(arrival + ONSET_SEARCH[1]))
    seg = env[s0:s1]

    onset, onset_status = {}, {}

    for k in K_ON:
        m = seg > k * b_loc
        if len(m) and m[0]:
            onset[k], onset_status[k] = np.nan, "pre_active"
            continue
        idx = first_sustained(m, I(ONSET_MIN_DUR))
        if idx is None:
            onset[k], onset_status[k] = np.nan, "none"
        else:
            onset[k], onset_status[k] = (s0 + idx) / fs - arrival, "ok"

    # peak
    p1 = max(a + 1, min(limit, a + I(PEAK_SEARCH)))
    peak_idx = a + int(np.argmax(env[a:p1]))

    # coda end
    coda, censored = {}, {}

    sub = env[peak_idx:limit]
    valid = np.flatnonzero(sub > DROPOUT_REL * b_ref)   # drop dropouts

    for k in K_END:
        idx = None
        if len(valid):
            m = sub[valid] < k * b_ref
            j = first_sustained(m, I(CODA_MIN_BELOW))
            idx = None if j is None else int(valid[j])
        if idx is None:
            coda[k], censored[k] = limit / fs - arrival, True
        else:
            coda[k], censored[k] = (peak_idx + idx) / fs - arrival, False

    span = env[peak_idx:min(limit, peak_idx + I(6 * 3600))]
    dropout_frac = float(np.mean(span <= DROPOUT_REL * b_ref)) if len(span) else 0.0

    # how many of the 6 windows that contain the catalogue time have
    # >= SIGNAL_SEC of visible signal inside? (window ends ~ +5..+55 s)
    o = onset[K_ON_DEFAULT]
    if np.isnan(o):
        sig_frac = np.nan
    else:
        ends = np.array([5, 15, 25, 35, 45, 55], dtype=float)
        sig_frac = float(np.mean((ends - o) >= SIGNAL_SEC))

    return {
        "b_loc": b_loc,
        "onset": onset,
        "onset_status": onset_status,
        "peak_delay": peak_idx / fs - arrival,
        "peak_ratio": float(env[peak_idx] / b_ref),
        "dropout_frac": dropout_frac,
        "sig_frac": sig_frac,
        "coda": coda,
        "censored": censored,
        "limit_s": limit / fs,
    }


# ---- PIPELINE ------------------------------------------------------

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.preprocessor import Preprocessor, PreprocessConfig
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


def main():

    events = DataLoader(
        catalog_path=CATALOG_PATH,
        data_dir=DATA_DIR,
    ).run()

    validation_report = Validator(DATA_DIR).run(events)

    waveforms = WaveformLoader().run(validation_report.valid_events)

    grouping = WaveformGrouper().run(waveforms)

    # split exactly as in the pipeline
    plain = Preprocessor().run(grouping.waveforms)

    windows = Windowing(
        WindowConfig(window_length=60, step_size=10)
    ).run(plain)

    labeled = Labeling(event_groups=grouping.event_groups).run(windows)

    dataset = DatasetBuilder(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    ).run(labeled)

    del windows, labeled

    filtered = Preprocessor(
        PreprocessConfig(low_hz=0.2, high_hz=None)
    ).run(grouping.waveforms)

    by_primary = {r.event.evid: r for r in filtered}

    # ------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------

    rows = []
    plot_items = []
    wave_stats = []
    fallbacks = 0

    for primary in dataset.train_events:

        rec = by_primary[primary]
        fs = rec.sampling_rate
        env = envelope(rec.data, fs)

        evs = grouping.event_groups[primary]      # sorted by arrival

        b_ref, fb = day_reference(env, fs, [e.time_rel for e in evs])
        fallbacks += int(fb)

        wave_stats.append(
            (
                primary,
                float(np.mean(env <= DROPOUT_REL * b_ref)),
                trailing_dropout_seconds(env, fs, b_ref),
            )
        )

        for k, e in enumerate(evs):

            nxt = evs[k + 1].time_rel if k + 1 < len(evs) else None

            r = analyse_event(env, fs, e.time_rel, nxt, b_ref)
            r.update(
                evid=e.evid,
                type=e.event_type,
                arrival=e.time_rel,
                b_ref=b_ref,
            )
            rows.append(r)

            if len(plot_items) < N_PLOT:
                lo = max(0, int((e.time_rel - 3600) * fs))
                hi = min(len(env), int((e.time_rel + 6 * 3600) * fs))
                plot_items.append(
                    (r, (np.arange(lo, hi) / fs - e.time_rel) / 3600.0,
                     env[lo:hi] / b_ref)
                )

    types = np.array([r["type"] for r in rows])

    print()
    print("=" * 60)
    print("ONSET / CODA DIAGNOSTIC (training events)")
    print("=" * 60)
    print(f"Events analysed: {len(rows)}")
    print(f"Waveforms using fallback reference: {fallbacks}")

    n_nan = sum(
        int(np.isnan(r["b_loc"]) or np.isnan(r["peak_ratio"])) for r in rows
    )
    print(f"Events with NaN results (must be 0): {n_nan}")

    bl = np.array([r["b_loc"] / r["b_ref"] for r in rows])
    print(f"Local / day background ratio: median {np.median(bl):.2f}, "
          f"min {bl.min():.2f}, max {bl.max():.2f}")

    # ------------------------------------------------------------
    # 0. Dropouts
    # ------------------------------------------------------------

    fr = np.array([w[1] for w in wave_stats])

    print()
    print("-" * 60)
    print("0. DROPOUTS (envelope < "
          f"{DROPOUT_REL} x background; zero-filled / missing data)")
    print("-" * 60)
    print(f"Share of the day flagged: median {np.median(fr):.3f} / "
          f"p90 {np.percentile(fr, 90):.3f} / max {fr.max():.3f}")
    print(f"Waveforms with >1% dropout: {(fr > 0.01).sum()} of {len(fr)}")
    print(f"{'waveform':10s} {'dropout share':>14s} "
          f"{'dropout at end of file (s)':>28s}")

    for wid, f_, tail in sorted(wave_stats, key=lambda t: -t[1])[:8]:
        print(f"{wid:10s} {f_:14.3f} {tail:28.0f}")

    tails = np.array([w[2] for w in wave_stats])
    print(f"Waveforms ending in a dropout > 60 s: {(tails > 60).sum()}")

    # ------------------------------------------------------------
    # 1. Onset offset
    # ------------------------------------------------------------

    print()
    print("-" * 60)
    print("1. VISIBLE ONSET - CATALOGUE TIME (s; negative = earlier)")
    print("-" * 60)
    print(f"{'k_on':>5s} {'ok':>4s} {'pre-active':>11s} {'none':>5s} "
          f"{'p10':>7s} {'p25':>7s} {'median':>7s} {'p75':>7s} "
          f"{'frac < -60 s':>13s}")

    for k in K_ON:

        v = np.array([r["onset"][k] for r in rows])
        st = [r["onset_status"][k] for r in rows]
        ok = ~np.isnan(v)

        if ok.sum() == 0:
            continue

        p10, p25, p50, p75 = np.percentile(v[ok], [10, 25, 50, 75])

        print(f"{k:5.1f} {ok.sum():4d} {st.count('pre_active'):11d} "
              f"{st.count('none'):5d} {p10:7.0f} {p25:7.0f} {p50:7.0f} "
              f"{p75:7.0f} {np.mean(v[ok] < -60):13.2f}")

    sf = np.array([r["sig_frac"] for r in rows])
    sf = sf[~np.isnan(sf)]

    print()
    print(f"Windows that contain the catalogue time AND >= {SIGNAL_SEC} s of "
          f"visible signal (k_on={K_ON_DEFAULT}):")
    print(f"  events with a visible onset: {len(sf)}")
    print(f"  median share of the 6 positive windows with signal: "
          f"{np.median(sf):.2f}")
    print(f"  events where NONE of them contains signal:  "
          f"{np.mean(sf == 0):.2f}")
    print(f"  events where ALL of them contain signal:    "
          f"{np.mean(sf == 1):.2f}")

    # ------------------------------------------------------------
    # 2. Peak
    # ------------------------------------------------------------

    pk = np.array([r["peak_delay"] for r in rows])
    pr = np.array([r["peak_ratio"] for r in rows])

    print()
    print("-" * 60)
    print("2. PEAK ENVELOPE")
    print("-" * 60)
    print(f"Peak time after catalogue arrival (s): "
          f"p25 {np.percentile(pk, 25):.0f} / median {np.median(pk):.0f} / "
          f"p75 {np.percentile(pk, 75):.0f}")
    print(f"Peak / day background: "
          f"p25 {np.percentile(pr, 25):.1f} / median {np.median(pr):.1f} / "
          f"p75 {np.percentile(pr, 75):.1f}")

    # ------------------------------------------------------------
    # 3. Coda duration
    # ------------------------------------------------------------

    print()
    print("-" * 60)
    print("3. CODA END - CATALOGUE TIME (s), censored = lower bound")
    print("-" * 60)
    print(f"{'k_end':>6s} {'censored':>9s} {'p25':>7s} {'median':>7s} "
          f"{'p75':>7s} {'p90':>7s}")

    for k in K_END:

        d = np.array([r["coda"][k] for r in rows])
        c = np.array([r["censored"][k] for r in rows])

        print(f"{k:6.1f} {c.mean():9.2f} "
              f"{np.percentile(d, 25):7.0f} {np.median(d):7.0f} "
              f"{np.percentile(d, 75):7.0f} {np.percentile(d, 90):7.0f}")

    print()
    print(f"Per type at k_end = {K_END_DEFAULT}")
    print(f"{'type':12s} {'n':>3s} {'censored':>9s} {'p25':>7s} "
          f"{'median':>7s} {'p75':>7s}")

    for t in sorted(set(types)):
        sel = types == t
        d = np.array([r["coda"][K_END_DEFAULT] for r in rows])[sel]
        c = np.array([r["censored"][K_END_DEFAULT] for r in rows])[sel]
        print(f"{t:12s} {sel.sum():3d} {c.mean():9.2f} "
              f"{np.percentile(d, 25):7.0f} {np.median(d):7.0f} "
              f"{np.percentile(d, 75):7.0f}")

    # decay constant
    tau_all = []
    tau_type = []

    for r in rows:
        c = r["coda"][K_END_DEFAULT]
        if r["censored"][K_END_DEFAULT] or r["peak_ratio"] <= K_END_DEFAULT:
            continue
        tau = (c - r["peak_delay"]) / np.log(r["peak_ratio"] / K_END_DEFAULT)
        tau_all.append(tau)
        tau_type.append(r["type"])

    tau_all = np.array(tau_all)
    tau_type = np.array(tau_type)

    print()
    print(f"Decay time constant tau (s), non-censored events "
          f"(n={len(tau_all)}):")
    print(f"  p25 {np.percentile(tau_all, 25):.0f} / "
          f"median {np.median(tau_all):.0f} / "
          f"p75 {np.percentile(tau_all, 75):.0f}")

    for t in sorted(set(tau_type)):
        sel = tau_type == t
        if sel.sum() >= 3:
            print(f"  {t}: n={sel.sum()} median {np.median(tau_all[sel]):.0f}")

    pr_ok = np.array([r["peak_ratio"] for r in rows
                      if not r["censored"][K_END_DEFAULT]])
    cd_ok = np.array([r["coda"][K_END_DEFAULT] for r in rows
                      if not r["censored"][K_END_DEFAULT]])
    if len(pr_ok) > 3:
        corr = np.corrcoef(np.log10(pr_ok), cd_ok)[0, 1]
        print(f"  correlation of coda length with log10(peak/background): "
              f"{corr:.2f}")

    # ------------------------------------------------------------
    # 4. Per-event table
    # ------------------------------------------------------------

    print()
    print("-" * 60)
    print(f"4. PER EVENT (k_on={K_ON_DEFAULT}, k_end={K_END_DEFAULT}), "
          f"first 20")
    print("-" * 60)
    print(f"{'evid':10s} {'type':10s} {'onset':>7s} {'peak':>7s} "
          f"{'coda':>7s} {'cens':>5s} {'limit':>7s} {'peak/bg':>8s} "
          f"{'dropout':>8s}")

    for r in rows[:20]:
        o = r["onset"][K_ON_DEFAULT]
        print(f"{r['evid']:10s} {r['type']:10s} "
              f"{'nan' if np.isnan(o) else f'{o:7.0f}':>7s} "
              f"{r['peak_delay']:7.0f} {r['coda'][K_END_DEFAULT]:7.0f} "
              f"{str(r['censored'][K_END_DEFAULT]):>5s} "
              f"{r['limit_s'] - r['arrival']:7.0f} "
              f"{r['peak_ratio']:8.1f} {r['dropout_frac']:8.2f}")

    # ------------------------------------------------------------
    # 5. Suggested LabelConfig (to be reviewed, not applied)
    # ------------------------------------------------------------

    v = np.array([r["onset"][K_ON_DEFAULT] for r in rows])
    v = v[~np.isnan(v)]

    margin = 60.0
    if len(v):
        margin = max(60.0, float(np.ceil(-np.percentile(v, 10) / 30) * 30))

    def p75_up(a):
        return float(np.ceil(np.percentile(a, 75) / 300) * 300)

    all_d = np.array([r["coda"][K_END_DEFAULT] for r in rows])
    suggestion = {}

    for t in sorted(set(types)):
        sel = types == t
        d = all_d[sel]
        suggestion[str(t)] = p75_up(d if sel.sum() >= 3 else all_d)

    print()
    print("-" * 60)
    print("5. SUGGESTED LabelConfig (review before using)")
    print("-" * 60)
    print(f"pre_onset_margin_s = {margin}")
    print(f"coda_duration_s    = {suggestion}")
    print(f"default_coda_s     = {p75_up(all_d)}")
    print("Censored events give lower bounds, so these are, if anything, "
          "too short.")
    print("NOTE: the coda depends on how strong the event is (see tau and "
          "the correlation above), so a per-event value is preferable.")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    for ax, (r, t_h, e_rel) in zip(axes.ravel(), plot_items):

        ax.semilogy(t_h, e_rel, linewidth=0.6)
        ax.axhline(1.0, color="gray", linewidth=0.8, label="day background")
        ax.axhline(K_END_DEFAULT, color="purple", linestyle=":",
                   linewidth=0.8)
        ax.axvline(0, color="red", linestyle="--", label="catalogue")

        o = r["onset"][K_ON_DEFAULT]
        if not np.isnan(o):
            ax.axvline(o / 3600.0, color="green", label="visible onset")

        ax.axvline(r["peak_delay"] / 3600.0, color="black", linewidth=0.8,
                   label="peak")
        ax.axvline(r["coda"][K_END_DEFAULT] / 3600.0, color="purple",
                   label="coda end")

        ax.set_title(f"{r['evid']} | {r['type']}"
                     f"{' | censored' if r['censored'][K_END_DEFAULT] else ''}")
        ax.set_ylabel("envelope / day background")

    axes[0, 0].legend(fontsize=7)
    axes[2, 0].set_xlabel("hours after catalogue arrival")
    axes[2, 1].set_xlabel("hours after catalogue arrival")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()