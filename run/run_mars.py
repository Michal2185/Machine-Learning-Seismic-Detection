"""
Zero-shot test of the FROZEN detectors on InSight snippets (feat/mars-insight).

    python -m run.run_mars --check-windowing                      # gate: my window grid == the Apollo windows (do this first)
    python -m run.run_mars --seeds 1 --arms lp1.3 --scale-mults 1 # smoke run
    python -m run.run_mars                                        # all models, both arms, N0 x 0.5 / 1 / 2, 5 seeds

What happens
  * every catalogued InSight file is resampled to 6.625 Hz, highpassed (0.2 Hz) and, in arm 'lp1.3', lowpassed (1.3 Hz);
    arm 'nolp' is the spectral-mismatch ablation. Targets (detect / ignore intervals) are shared by the arms.
  * the N0 constant is re-estimated on the Mars background of the same arm (calibrate_n0) and set on every loaded
    detector (set_fixed_scale); thresholds and feature / envelope standardisation stay at the Apollo values.
    --scale-mults runs c x 0.5, 1, 2 as a sensitivity band. There is NO uncalibrated arm: Mars counts and Apollo SI
    units differ by ~1e11, so that run would only test float overflow.
  * every (model, seed) is scored by exactly one fold model per waveform: waveform i -> fold (i + seed) mod 5, with that
    fold's clean-validation thresholds (policy D). A separate audit scores every waveform with EVERY fold model.
  * frozen STA/LTA (data/lunar/derived/baselines/sta_lta_frozen.json) is evaluated on the same streams.
  * records use the format of run_stress.py (config = "<model>@mars-<arm>-x<mult>"), so tests.analyze_results works.

Read the false-alarm numbers with care: the files are 1 h snippets that contain the event, so the scored background is
about 1.1 h in total for the two catalogued files (detect and ignore intervals and masked edges removed). One hour at the
strict budget (0.25/day) expects 0.01 false alarms, so false alarms per day are NOT an estimate from this data; the output
prints counts and scored hours instead.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.checkpoints import load_checkpoint
from src.modules.evaluation.module import build_series, pool_results
from src.modules.mars import chain
from src.modules.mars.scale import calibrate_n0, find_normalizers, set_fixed_scale
from src.modules.mars.streams import build_streams, fold_for, make_windows, read_catalogue, window_grid
from src.modules.normalization import NormConfig, Normalizer
from src.modules.reporting import extract_tallies
from run import run_dnn_features as R
from run.run_stress import RECURRENT, config_name

CATALOG = "data/mars/catalogs/Mars_InSight_training_catalog_final.csv"
DATA_DIR = "data/mars/data/"
STA_LTA_JSON = "data/lunar/derived/baselines/sta_lta_frozen.json"
DEFAULT_MODELS = ["cfcE:1k", "gruE:1k", "cfc0E:1k", "rfc:medium", "cnn:1k", "amp:rms", "amp:ptp"]
POINTS = {"strict": 0.25, "lenient": 2.0}              # frozen STA/LTA operating points -> budgets


def check_windowing(a):
    """My window grid must equal the Apollo windows of one real waveform (end_sample), and report the end_time rule."""

    from src.modules.normalization.module import end_sample_of
    from src.modules.pipeline import DataPipeline, PipelineConfig

    bundle = DataPipeline(PipelineConfig(R.CATALOG_PATH, R.DATA_DIR, R.DERIVED_DIR)).build()
    wid = sorted(bundle._processed_by_id)[0]
    s = bundle.eval_stream(wid)
    theirs = sorted((w for w in bundle.windows if w.event.evid == wid), key=end_sample_of)
    mine = make_windows(s)
    fs = float(s.sampling_rate)

    print(f"{wid}: {len(s.data)} samples, {len(mine)} windows mine vs {len(theirs)} Apollo")
    print("first Apollo windows (end_sample, start_time, end_time):",
          [(end_sample_of(w), getattr(w, "start_time", None), getattr(w, "end_time", None)) for w in theirs[:3]])
    print("last  Apollo window  (end_sample, start_time, end_time):",
          (end_sample_of(theirs[-1]), getattr(theirs[-1], "start_time", None), getattr(theirs[-1], "end_time", None)))

    m = min(len(mine), len(theirs))
    de = max(abs(mine[i].end_sample - end_sample_of(theirs[i])) for i in range(m))
    ok = len(mine) == len(theirs) and de == 0
    print(f"max |end_sample difference| over {m} windows: {de}")

    if ok and hasattr(theirs[0], "end_time"):
        by_samples = max(abs(float(t.end_time) - end_sample_of(t) / fs) for t in theirs)
        by_nominal = max(abs(float(t.end_time) - (60.0 + 10.0 * k)) for k, t in enumerate(theirs))
        print(f"Apollo end_time vs end_sample / fs: max difference {by_samples:.3g} s; vs 60 + 10 k: {by_nominal:.3g} s")
        print("end_time rule: " + ("'samples' (default, matches)" if by_samples < 1e-6 else
                                    "'nominal': call make_windows(..., time_rule='nominal') in run_mars" if by_nominal < 1e-6
                                    else "NEITHER: paste this output"))
        ok = ok and (by_samples < 1e-6)

    print("WINDOWING MATCHES" if ok else "WINDOWING NOT CONFIRMED: see the lines above before trusting any Mars number")
    sys.exit(0 if ok else 1)


def summarise_outcomes(pooled):
    return {o.evid: (bool(o.detected), None if not np.isfinite(o.delay_s) else float(o.delay_s)) for o in pooled.outcomes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--arms", nargs="+", default=["lp1.3", "nolp"])
    ap.add_argument("--scale-mults", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.25, 2.0])
    ap.add_argument("--models-dir", default="results/models")
    ap.add_argument("--rec-tag", default="long")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--sta-lta", default=STA_LTA_JSON)
    ap.add_argument("--no-sta-lta", action="store_true")
    ap.add_argument("--out", default="results/mars.json")
    ap.add_argument("--scores", default="results/mars_scores.npz", help="per fold-model score series (for the timeline plot)")
    ap.add_argument("--check-windowing", action="store_true")
    a = ap.parse_args()

    if a.check_windowing:
        check_windowing(a)

    R.REC.update(epochs=None, patience=None, min_epochs=None, lr=1e-3)
    fs = chain.FS_APOLLO
    evaluator = R.make_evaluator()

    catalogue = read_catalogue(a.catalog)
    streams_by_arm, info = build_streams(catalogue, a.data_dir, arms=tuple(a.arms))

    print(f"{len(catalogue)} Mars waveforms: " + ", ".join(i["evid"] for i in info))

    for i in info:
        z = i["zone"]
        print(f"  {i['evid']}: onset {z['status']} ({z['onset_offset_s']:.0f} s), peak/bg {z['peak_ratio']:.1f}, "
              f"coda end {z['coda_end_s']:.0f} s{' (censored)' if z['coda_censored'] else ''}, "
              f"detect {i['target'].detect}, ignore ({i['target'].ignore[0]:.0f}, {i['target'].ignore[1]:.0f}), "
              f"usable onset {i['target'].has_positive}")

    records, audit, scores_out, meta = [], [], {}, dict(n0={}, apollo_n0={})
    n_w = len(catalogue)

    for arm in a.arms:

        streams = streams_by_arm[arm]
        by_id = {s.waveform_id: s for s in streams}
        wins = [w for s in streams for w in make_windows(s)]
        c_mars = calibrate_n0(streams)
        meta["n0"][arm] = c_mars
        sdur = {s.waveform_id: len(s.data) / fs for s in streams}
        print(f"\n=== arm {arm}: Mars N0 constant {c_mars:.4g} ({sum(len(w.data) > 0 for w in wins)} windows) ===")

        # ---- score every fold model once per (arm, mult)
        cache = {}

        for spec in a.models:

            kind, size = spec.split(":")
            base = config_name(kind, size, a.rec_tag)
            n_seeds = 1 if kind == "amp" else a.seeds
            t0 = time.time()

            for seed in range(n_seeds):
                for fold in range(a.folds):

                    det = R.build_detector(kind, size, "N0", f"{base}-s{seed}", Normalizer(NormConfig("N0")), fs, seed, a.epochs)
                    thr, _ = load_checkpoint(a.models_dir, base, seed, fold, det)
                    meta["apollo_n0"][base] = [n.fixed_scale for n in find_normalizers(det)][0]

                    for mult in a.scale_mults:
                        set_fixed_scale(det, c_mars * mult)
                        sc = np.asarray(det.score(wins, by_id), dtype=np.float64)
                        cache[(base, seed, fold, mult)] = (build_series(wins, sc), thr, sc)

            print(f"  scored {base} ({n_seeds} seed(s) x {a.folds} folds x {len(a.scale_mults)} scales) in {time.time() - t0:.0f} s; "
                  f"Apollo N0 {meta['apollo_n0'][base]:.3g}, Mars/Apollo ratio {c_mars / meta['apollo_n0'][base]:.3g}")

        # ---- policy D (one fold model per waveform and seed) + all-folds audit
        for spec in a.models:

            kind, size = spec.split(":")
            base = config_name(kind, size, a.rec_tag)
            n_seeds = 1 if kind == "amp" else a.seeds

            for mult in a.scale_mults:

                tag = f"{base}@mars-{arm}-x{mult:g}"

                for budget in a.budgets:
                    for seed in range(n_seeds):

                        res, used = [], []

                        for i, s in enumerate(streams):
                            fold = fold_for(i, seed, a.folds)
                            series, thr, _ = cache[(base, seed, fold, mult)]
                            res.append(evaluator.evaluate([s], series, thr[budget]))
                            used.append(float(thr[budget]))

                        pooled = pool_results(res)
                        records.append(dict(config=tag, budget=budget, seed=seed, recall=float(pooled.recall("all")),
                                            fa_per_day=float(pooled.fa_per_day), thresholds=used,
                                            n_alarms=int(pooled.n_alarms), n_false_alarms=int(pooled.n_false_alarms),
                                            scored_hours=float(pooled.background_hours),
                                            events=summarise_outcomes(pooled), outcomes=extract_tallies(pooled)))

                    for seed in range(n_seeds):
                        for fold in range(a.folds):
                            series, thr, _ = cache[(base, seed, fold, mult)]
                            r = evaluator.evaluate(streams, series, thr[budget])
                            audit.append(dict(config=tag, budget=budget, seed=seed, fold=fold, threshold=float(thr[budget]),
                                              events=summarise_outcomes(r), n_alarms=int(r.n_alarms),
                                              n_false_alarms=int(r.n_false_alarms),
                                              alarms={s.waveform_id: [float(t) for t in r_s.alarms]
                                                      for s, r_s in zip(streams, r.streams)}))

                if mult == 1.0:
                    for seed in range(min(n_seeds, 1)):
                        for fold in range(a.folds):
                            series, thr, sc = cache[(base, seed, fold, mult)]
                            for wid, ss in series.items():
                                scores_out[f"{base}|{arm}|s{seed}|f{fold}|{wid}|t"] = ss.times
                                scores_out[f"{base}|{arm}|s{seed}|f{fold}|{wid}|y"] = ss.scores
                            scores_out[f"{base}|{arm}|s{seed}|f{fold}|thr"] = np.array([thr[b] for b in a.budgets])

        # ---- frozen STA/LTA on the same streams (the 1 h file replaces the 'whole day so far' LTA: stated in the paper)
        if not a.no_sta_lta:

            from src.modules.detectors.frozen import FrozenSpec, evaluate_frozen

            spec = FrozenSpec.load(a.sta_lta)
            out = evaluate_frozen(spec, wins, streams, evaluator)

            for pname, r in out.items():
                budget = POINTS.get(pname, spec.points[pname].budget_fa_per_day)
                records.append(dict(config=f"sta_lta-frozen@mars-{arm}-x1", budget=budget, seed=0,
                                    recall=float(r.recall("all")), fa_per_day=float(r.fa_per_day),
                                    thresholds=[float(r.threshold)], n_alarms=int(r.n_alarms),
                                    n_false_alarms=int(r.n_false_alarms), scored_hours=float(r.background_hours),
                                    events=summarise_outcomes(r), outcomes=extract_tallies(r)))

    # ---------------- printed summary
    print("\n" + "=" * 100)
    print("Per-event detections (policy D: one fold model per waveform and seed; audit: every fold model), x1 scale")
    print("scored background hours are tiny: false alarms are counts, NOT rates")

    for arm in a.arms:
        for budget in a.budgets:
            print(f"\n--- arm {arm}, budget {budget:g}/day thresholds ---")
            print(f"  {'config':34s} " + " ".join(f"{i['evid']:>16s}" for i in info) + "   alarms   false  scored h")

            for r_cfg in sorted({r["config"] for r in records if r["config"].endswith(f"@mars-{arm}-x1") and r["budget"] == budget}):

                rs = [r for r in records if r["config"] == r_cfg and r["budget"] == budget]
                cells = []

                for i in info:
                    cells.append(f"{sum(bool(r['events'][i['evid']][0]) for r in rs)}/{len(rs)}")

                au = [x for x in audit if x["config"] == r_cfg and x["budget"] == budget]
                au_cells = [f"{sum(x['events'][i['evid']][0] for x in au)}/{len(au)}" if au else "-" for i in info]

                print(f"  {r_cfg.split('@')[0]:34s} " + " ".join(f"{c + ' [' + b + ']':>16s}" for c, b in zip(cells, au_cells))
                      + f"   {np.mean([r['n_alarms'] for r in rs]):6.1f} {np.mean([r['n_false_alarms'] for r in rs]):7.2f} "
                      + f"{np.mean([r['scored_hours'] for r in rs]):8.2f}")

    print("\ncells: detected / (seed runs) [detected / (all fold models, audit)]")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(records=records, audit=audit, meta=meta,
                                   targets={i["evid"]: dict(detect=i["target"].detect, ignore=i["target"].ignore,
                                                            arrival=i["target"].arrival_s, onset=i["target"].onset_s,
                                                            zone=i["zone"]) for i in info}), indent=1, default=float))
    np.savez_compressed(a.scores, **scores_out)
    print(f"wrote {out} and {a.scores}")


if __name__ == "__main__":
    main()