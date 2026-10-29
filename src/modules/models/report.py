import numpy as np

# frozen STA/LTA, nested cross-validation, all events (docs/EVALUATION_BRANCH_HANDOFF.md)
STA_LTA_REFERENCE = {
    0.25: dict(recall=0.67, fa_per_day=0.30, recall_ci=(0.56, 0.76), fa_ci=(0.14, 0.50)),
    2.0: dict(recall=0.81, fa_per_day=0.43, recall_ci=(0.72, 0.89), fa_ci=(0.24, 0.64)),
}


def summarise(records: list) -> list:
    """
    records: dicts with config, budget, seed, recall, fa_per_day, test_ap
    (one per config x budget x seed). Returns one row per config x budget
    with mean and sample standard deviation over seeds.
    """

    groups = {}

    for r in records:
        groups.setdefault((r["config"], r["budget"]), []).append(r)

    rows = []

    for (config, budget), rs in groups.items():

        rec = np.array([r["recall"] for r in rs], dtype=float)
        fa = np.array([r["fa_per_day"] for r in rs], dtype=float)
        ap = np.array([r["test_ap"] for r in rs], dtype=float)

        rows.append(dict(
            config=config,
            budget=budget,
            n_seeds=len(rs),
            recall_mean=float(np.mean(rec)),
            recall_sd=float(np.std(rec, ddof=1)) if len(rs) > 1 else 0.0,
            fa_mean=float(np.mean(fa)),
            fa_sd=float(np.std(fa, ddof=1)) if len(rs) > 1 else 0.0,
            ap_mean=float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
        ))

    return sorted(rows, key=lambda r: (r["budget"], r["config"]))


def format_table(rows: list, resources: dict) -> str:
    """resources: config -> ResourceReport (or None)."""

    lines = []

    for budget in sorted({r["budget"] for r in rows}):

        ref = STA_LTA_REFERENCE.get(budget)
        head = f"budget {budget:g} false alarms/day"
        head += f"   [STA/LTA: recall {ref['recall']:.2f} at {ref['fa_per_day']:.2f}/day]" if ref else ""
        lines += ["", head,
                  f"  {'config':26s} {'seeds':>5s} {'recall':>14s} {'FA/day':>12s} {'test AP':>8s} "
                  f"{'params':>9s} {'int8 KiB':>9s} {'MACs':>9s}"]

        for r in (x for x in rows if x["budget"] == budget):

            res = resources.get(r["config"])
            tail = (f"{res.params:9,d} {res.bytes_int8 / 1024:9.1f} {res.macs_per_window:9,d}"
                    if res else f"{'-':>9s} {'-':>9s} {'-':>9s}")

            lines.append(
                f"  {r['config']:26s} {r['n_seeds']:5d} "
                f"{r['recall_mean']:6.2f} ± {r['recall_sd']:4.2f} "
                f"{r['fa_mean']:6.2f} ± {r['fa_sd']:4.2f} {r['ap_mean']:8.3f} {tail}"
            )

    lines += ["", "Recall differences below about 0.15 are inconclusive (see EVALUATION handoff).",
              "FA/day is measured on test folds and can differ from the budget used for the threshold."]

    return "\n".join(lines)