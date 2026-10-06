"""
Dropout check on the RAW samples (no filtering).

The earlier integrity check looked for constant runs in the FILTERED
data, where a highpass filter turns exact zeros into a smoothly decaying
signal, so it could not see zero-filled stretches. Here we look at the
stored samples directly:

  * exact zeros
  * linear interpolation (zero second difference)

and report how much of each waveform is affected, how long the runs are,
and whether files start / end with a gap.
"""

import numpy as np


MIN_RUN_SEC = 60
FOCUS = ("evid00055", "evid00120", "evid00107", "evid00108", "evid00114")


def run_lengths(flags):
    """Lengths (in samples) of consecutive True runs, plus start indices."""
    f = np.asarray(flags, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], f, [0]))))
    starts, ends = edges[::2], edges[1::2]
    return ends - starts, starts


def analyse_raw(x, fs):

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    min_run = int(MIN_RUN_SEC * fs)

    ref = float(np.median(np.abs(x[x != 0]))) if np.any(x != 0) else 1.0

    zero = x == 0
    d2 = np.abs(np.diff(x, 2))            # d2[i] belongs to sample i + 1
    interp = np.zeros(n, dtype=bool)
    interp[1:-1] = d2 <= 1e-6 * ref
    interp[0] = interp[1]                 # ends take their neighbour's value,
    interp[-1] = interp[-2]               # so a gap at the start/end is seen

    out = {}

    for name, flags in (("zero", zero), ("interp", interp)):

        lens, starts = run_lengths(flags)
        long = lens >= min_run

        mid = lens >= int(10 * fs)

        out[name] = {
            "runs": [
                (float(a / fs), float((a + l) / fs))
                for a, l in zip(starts[long], lens[long])
            ],
            "n_runs_10s": int(mid.sum()),
            "share_samples": float(flags.mean()),
            "share_in_long_runs": float(lens[long].sum() / n) if long.any() else 0.0,
            "n_long_runs": int(long.sum()),
            "longest_s": float(lens.max() / fs) if len(lens) else 0.0,
            "longest_start_s": float(starts[np.argmax(lens)] / fs) if len(lens) else 0.0,
            "starts_with": bool(flags[0]),
            "ends_with": bool(flags[-1]),
            "tail_s": 0.0,
        }

        if flags[-1]:
            ok = np.flatnonzero(~flags)
            out[name]["tail_s"] = float((n - 1 - ok[-1]) / fs) if len(ok) else n / fs

    return out


# ---- PIPELINE ------------------------------------------------------

from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.waveform_grouping import WaveformGrouper


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

    group_of = grouping.group_of

    rows = []

    for rec in grouping.waveforms:
        tr = rec.stream[0]
        res = analyse_raw(tr.data, float(tr.stats.sampling_rate))
        rows.append((rec.event.evid, res))

    print()
    print("=" * 60)
    print("RAW DROPOUT CHECK (distinct waveforms)")
    print("=" * 60)
    print(f"Waveforms: {len(rows)}")

    for kind, label in (("zero", "exact zeros"),
                        ("interp", "linear interpolation")):

        sh = np.array([r[1][kind]["share_in_long_runs"] for r in rows])
        nr = np.array([r[1][kind]["n_long_runs"] for r in rows])
        longest = np.array([r[1][kind]["longest_s"] for r in rows])
        tails = np.array([r[1][kind]["tail_s"] for r in rows])
        heads = sum(r[1][kind]["starts_with"] for r in rows)

        print()
        print(f"-- {label} (runs >= {MIN_RUN_SEC} s) --")
        print(f"Waveforms with at least one long run: {(nr > 0).sum()}")
        print(f"Share of samples inside long runs: median {np.median(sh):.3f} / "
              f"p90 {np.percentile(sh, 90):.3f} / max {sh.max():.3f}")
        print(f"Longest run (s): median {np.median(longest):.0f} / "
              f"max {longest.max():.0f}")
        print(f"Waveforms ending in such a run: {(tails > 0).sum()} "
              f"(tail > 60 s: {(tails > 60).sum()}, max {tails.max():.0f} s)")
        print(f"Waveforms starting in such a run: {heads}")

    print()
    print("-" * 60)
    print("Waveforms with the most INTERPOLATED data (runs >= "
          f"{MIN_RUN_SEC} s)")
    print("-" * 60)
    print(f"{'waveform':10s} {'share':>7s} {'runs':>5s} {'longest s':>10s} "
          f"{'starts at s':>12s} {'head?':>6s} {'tail s':>8s}")

    ranked = sorted(
        rows, key=lambda t: -t[1]["interp"]["share_in_long_runs"]
    )

    for wid, r in ranked[:10]:
        i = r["interp"]
        if i["n_long_runs"] == 0:
            continue
        print(f"{wid:10s} {i['share_in_long_runs']:7.3f} {i['n_long_runs']:5d} "
              f"{i['longest_s']:10.0f} {i['longest_start_s']:12.0f} "
              f"{str(i['starts_with']):>6s} {i['tail_s']:8.0f}")

    print()
    print("Distinct waveforms by share of interpolated data (long runs):")
    shares = np.array([r[1]["interp"]["share_in_long_runs"] for r in rows])
    for thr in (0.0, 0.001, 0.01, 0.05, 0.10):
        print(f"  > {thr:5.3f}: {(shares > thr).sum()}")

    print()
    print("-" * 60)
    print("Top 10 by share of samples in long INTERPOLATED runs")
    print("-" * 60)
    print(f"{'waveform':10s} {'share':>7s} {'runs>=60s':>10s} "
          f"{'runs>=10s':>10s} {'longest s':>10s}")

    for wid, r in sorted(
        rows, key=lambda t: -t[1]["interp"]["share_in_long_runs"]
    )[:10]:
        i = r["interp"]
        print(f"{wid:10s} {i['share_in_long_runs']:7.3f} "
              f"{i['n_long_runs']:10d} {i['n_runs_10s']:10d} "
              f"{i['longest_s']:10.0f}")

    n10 = np.array([r[1]["interp"]["n_runs_10s"] for r in rows])
    print()
    print(f"Interpolated runs >= 10 s per waveform: median {np.median(n10):.0f} "
          f"/ max {n10.max()}  (waveforms with at least one: {(n10 > 0).sum()})")

    print()
    print("-" * 60)
    print("Catalogue events inside / within 600 s of a long interpolated run")
    print("-" * 60)

    found = False

    for wid, r in rows:
        runs = r["interp"]["runs"]
        if not runs:
            continue
        for e in grouping.event_groups[wid]:
            d = min(
                0.0 if a <= e.time_rel <= b
                else min(abs(e.time_rel - a), abs(e.time_rel - b))
                for a, b in runs
            )
            if d <= 600:
                found = True
                print(f"{e.evid} | {e.event_type:10s} | "
                      f"arrival={e.time_rel:.0f} s | distance to gap {d:.0f} s")

    if not found:
        print("none")

    print()
    print("-" * 60)
    print("The records with the smallest robust std earlier")
    print("-" * 60)

    by_wid = dict(rows)

    for evid in FOCUS:
        wid = group_of.get(evid)
        if wid is None or wid not in by_wid:
            print(f"{evid}: not found")
            continue
        z = by_wid[wid]["zero"]
        i = by_wid[wid]["interp"]
        print(f"{evid} (waveform {wid}): zero share {z['share_in_long_runs']:.3f}, "
              f"longest {z['longest_s']:.0f} s, tail {z['tail_s']:.0f} s | "
              f"interp share {i['share_in_long_runs']:.3f}")


if __name__ == "__main__":
    main()