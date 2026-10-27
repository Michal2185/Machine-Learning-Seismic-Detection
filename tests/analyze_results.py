"""
Detail output and bootstrap comparisons from the JSON files written by tests/run_dnn_features.py.

    python -m tests.analyze_results results/final.json                       # everything in the file
    python -m tests.analyze_results results/a.json results/b.json --reference amp-rms-N0
    python -m tests.analyze_results results/*.json --compare cnn-10k-N2 rfc-medium-N2 --n-boot 2000

Needs runs made with the current runner (records carry 'outcomes'); older files are skipped with a note.
Waveforms are resampled (events and false alarms of one day are not independent); recall / false alarms are
SEED-AVERAGED, so the intervals describe uncertainty over waveforms, and 'seed sd' shows the spread over seeds.
Paired comparisons use the same resamples for both detectors; '*' = the 95 % interval excludes 0.

Writes <out>.csv (every number) and <out>_recall.png (recall with intervals per budget, STA/LTA as a band).
STA/LTA is only drawn as a reference band here: its per-event outcomes are needed for a paired comparison.
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.models.report import STA_LTA_REFERENCE
from src.modules.reporting import Panel, bootstrap, compare, significant

OUT_PREFIX = "results/analysis"


def load_panels(paths) -> tuple:
    """Returns ({(config, budget): Panel}, skipped record count)."""

    by = {}
    skipped = 0

    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path) as f:
                data = json.load(f)
            for r in data["records"]:
                if "outcomes" not in r:
                    skipped += 1
                    continue
                by.setdefault((r["config"], r["budget"]), {})[r["seed"]] = r["outcomes"]

    return {k: Panel([v[s] for s in sorted(v)]) for k, v in by.items()}, skipped


def _f(x, nd=2):
    return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def _iv(t, nd=2):
    return f"{_f(t[0], nd)} [{_f(t[1], nd)}, {_f(t[2], nd)}]"


def detail_rows(panels, n_boot, seed):

    rows, stats = [], {}

    for (config, budget), p in sorted(panels.items(), key=lambda kv: (kv[0][1], kv[0][0])):

        b = bootstrap(p, n_boot, seed)
        stats[(config, budget)] = b

        for name, (pt, lo, hi) in b.items():
            rows.append(dict(level="detail", budget=budget, config=config, other="", metric=name,
                             point=pt, lo=lo, hi=hi, p=""))

        for ref in ("onset", "arrival"):
            d = p.delays(ref)
            for k in ("n", "median", "p25", "p75"):
                rows.append(dict(level="delay", budget=budget, config=config, other="", metric=f"delay_{ref}_{k}",
                                 point=d[k], lo="", hi="", p=""))

    return rows, stats


def print_detail(panels, stats):

    for budget in sorted({k[1] for k in panels}):

        keys = sorted((c for c, b in panels if b == budget))
        ref = panels[(keys[0], budget)]
        counts = ", ".join(f"{g} {int(ref.ev[:, j].sum())}" for j, g in enumerate(ref.groups))

        print(f"\n{'=' * 100}\nbudget {budget:g} false alarms/day   events: {counts}")
        print(f"  {'config':18s} {'seeds':>5s} {'recall all [95%]':>20s} {'seed sd':>8s} {'usable':>6s} {'weak':>6s} "
              f"{'FA/day [95%]':>20s} {'delay vs onset (s)':>26s}")

        for c in keys:
            p, b = panels[(c, budget)], stats[(c, budget)]
            sd = float(np.std(p.per_seed_recall("all"), ddof=1)) if p.n_seeds > 1 else 0.0
            d = p.delays("onset")
            delay = f"{_f(d['median'], 0)} ({_f(d['p25'], 0)}-{_f(d['p75'], 0)}, n={d['n']})"
            print(f"  {c:18s} {p.n_seeds:5d} {_iv(b['recall_all']):>20s} {sd:8.3f} {_f(b['recall_usable'][0]):>6s} "
                  f"{_f(b['recall_weak'][0]):>6s} {_iv(b['fa_per_day']):>20s} {delay:>26s}")

        types = [g for g in ref.groups if g not in ("all", "usable", "weak")]
        if types:
            print("  recall by event type: " + " | ".join(
                f"{g} " + ", ".join(f"{c}={_f(stats[(c, budget)][f'recall_{g}'][0])}" for c in keys)
                for g in types))

        r = STA_LTA_REFERENCE.get(budget)
        if r:
            print(f"  STA/LTA (nested CV, handoff): recall {r['recall']:.2f} {list(r['recall_ci'])}, "
                  f"FA/day {r['fa_per_day']:.2f} {list(r['fa_ci'])}")


def compare_rows(panels, pairs, n_boot, seed, level):

    rows = []

    for budget, a_name, b_name in pairs:
        a, b = panels.get((a_name, budget)), panels.get((b_name, budget))
        if a is None or b is None:
            continue
        for name, row in compare(a, b, n_boot, seed).items():
            rows.append(dict(level=level, budget=budget, config=a_name, other=b_name, metric=name,
                             point=row[0], lo=row[1], hi=row[2], p=row[3]))

    return rows


def print_pairs(rows, title):

    if not rows:
        return

    print(f"\n{title}: difference = second minus first, 95 % interval, p = share of resamples where the second is better")

    keyf = lambda r: (r["budget"], r["config"], r["other"])
    seen = []

    for r in rows:
        if keyf(r) not in seen:
            seen.append(keyf(r))

    for k in seen:
        sel = {r["metric"]: r for r in rows if keyf(r) == k}
        parts = []
        for m in ("recall_all", "recall_usable", "recall_weak", "fa_per_day"):
            if m in sel:
                r = sel[m]
                star = "*" if significant((r["point"], r["lo"], r["hi"])) else " "
                parts.append(f"{m.replace('recall_', 'rec ')} {r['point']:+.2f} [{_f(r['lo'])}, {_f(r['hi'])}]{star}")
        print(f"  {k[0]:g}/day  {k[1]:16s} -> {k[2]:16s} " + "  ".join(parts) + f"   p(all) {_f(sel['recall_all']['p'])}")


def auto_pairs(panels, reference, extra):

    budgets = sorted({b for _, b in panels})
    names = sorted({c for c, _ in panels})
    vs_ref = [(b, reference, c) for b in budgets for c in names if c != reference and (reference, b) in panels]

    norm = []
    for c in names:
        if c.endswith("-N2") and c[:-3] + "-N0" in names:
            norm += [(b, c, c[:-3] + "-N0") for b in budgets]          # N0 minus N2

    return vs_ref, norm, [(b, x, y) for b in budgets for x, y in extra]


def plot_recall(panels, stats, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = sorted({b for _, b in panels})
    fig, ax = plt.subplots(1, len(budgets), figsize=(6.5 * len(budgets), 0.38 * len({c for c, _ in panels}) + 2.5),
                           squeeze=False)

    for a, budget in zip(ax[0], budgets):
        names = sorted((c for c, b in panels if b == budget), key=lambda c: stats[(c, budget)]["recall_all"][0])
        for i, c in enumerate(names):
            pt, lo, hi = stats[(c, budget)]["recall_all"]
            a.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", color="C0", capsize=3)
        a.set_yticks(range(len(names)), names, fontsize=8)
        r = STA_LTA_REFERENCE.get(budget)
        if r:
            a.axvspan(*r["recall_ci"], color="0.85", label="STA/LTA nested CV 95 % interval")
            a.axvline(r["recall"], color="0.4", ls="--")
            a.legend(fontsize=7, loc="lower right")
        a.set_xlim(0, 1.02)
        a.set_xlabel("recall (all events), seed-averaged, 95 % interval over waveforms")
        a.set_title(f"budget {budget:g} false alarms/day")
        a.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--reference", default="amp-rms-N0", help="config every other config is compared with")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_PREFIX)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    panels, skipped = load_panels(a.files)

    if skipped:
        print(f"note: {skipped} record(s) without 'outcomes' skipped (rerun them with the current runner)")

    if not panels:
        sys.exit("no records with outcomes found")

    rows, stats = detail_rows(panels, a.n_boot, a.seed)
    print_detail(panels, stats)

    vs_ref, norm, extra = auto_pairs(panels, a.reference, [tuple(p) for p in a.compare])
    r1 = compare_rows(panels, vs_ref, a.n_boot, a.seed, "vs_reference")
    r2 = compare_rows(panels, norm, a.n_boot, a.seed, "n2_to_n0")
    r3 = compare_rows(panels, extra, a.n_boot, a.seed, "pair")

    print_pairs(r1, f"Every config against the reference {a.reference}")
    print_pairs(r2, "N0 against N2 (same model)")
    print_pairs(r3, "Requested pairs")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(str(out) + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "budget", "config", "other", "metric", "point", "lo", "hi", "p"])
        w.writeheader()
        w.writerows(rows + r1 + r2 + r3)

    if not a.no_plot:
        plot_recall(panels, stats, str(out) + "_recall.png")

    print(f"\nwrote {out}.csv" + ("" if a.no_plot else f" and {out}_recall.png"))


if __name__ == "__main__":
    main()