"""
Nested cross-validation of the feature models, the CNN and the amplitude control through the existing
evaluator (file name kept from the first DNN run).

    python -m tests.run_dnn_features                    # DNN 1k + 10k, N0 + N2, 5 seeds
    python -m tests.run_dnn_features --seeds 1          # quick smoke run
    python -m tests.run_dnn_features --rfc              # add the random forest
    python -m tests.run_dnn_features --ablate           # add N1 / N3 for the 10k DNN
    python -m tests.run_dnn_features --amp              # amplitude control: threshold on window RMS / peak-to-peak
    python -m tests.run_dnn_features --cnn              # 1D CNN on the normalised raw window (--cnn-sizes 1k 10k 100k)
    python -m tests.run_dnn_features --only amp cnn     # skip the default DNN configurations
    python -m tests.run_dnn_features --sizes 1k 10k 100k --norms N0 N2

Every seed is its OWN cross_validate_budgets call: that function keeps the best candidate per fold
by validation recall, so putting seeds into one call would silently select seeds.

Paths: edit the constants once (same values as in normal_inspect / visualize_normalization).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.models import (
    AmplitudeDetector,
    build_cnn_detector,
    build_dnn_detector,
    build_recurrent_detector,
    build_rfc_detector,
)
from src.modules.checkpoints import save_checkpoint, verify_roundtrip
from src.modules.models.report import format_table, summarise
from src.modules.normalization import NormConfig, Normalizer, ScaleCache
from src.modules.reporting import extract_tallies
from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.training import TrainConfig

# ---- EDIT THESE ONCE ----------------------------------------------------------
CATALOG_PATH = "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
DATA_DIR = "data/lunar/data/S12_GradeA/"
DERIVED_DIR = "data/lunar/derived"
OUT_PATH = "results/dnn_features.json"
# -------------------------------------------------------------------------------


def make_evaluator():
    """Same construction as tests/freeze_sta_lta.py. ADAPT IF YOURS DIFFERS (I have not seen that file)."""

    from src.modules.evaluation.module import Evaluator
    from src.modules.evaluation.types import EvalConfig

    return Evaluator(EvalConfig())


def make_normalizer(variant, cache):
    return Normalizer(NormConfig(variant), cache if variant in ("N2", "N3") else None)


def specs(args):

    only = set(args.only) if args.only else None
    out = []

    def want(kind):
        return only is None or kind in only

    if want("dnn"):
        out += [("dnn", s, n) for n in args.norms for s in args.sizes]

    if args.rfc or (only and "rfc" in only):
        out += [("rfc", "medium", n) for n in args.norms]

    if args.cnn or (only and "cnn" in only):
        out += [("cnn", s, n) for n in args.norms for s in args.cnn_sizes]

    for k in ("gru", "cfc", "gruE", "cfcE", "cfc0E"):
        if (args.rec and k in ("gru", "cfc")) or (args.rec_e and k in ("gruE", "cfcE", "cfc0E")) or (only and k in only):
            out += [(k, s, n) for n in args.norms for s in args.rec_sizes]

    if args.amp or (only and "amp" in only):
        out += [("amp", k, n) for n in args.norms for k in ("rms", "ptp")]

    if args.ablate:
        out += [("dnn", "10k", n) for n in ("N1", "N3") if n not in args.norms]

    return out


# schedule overrides for GRU / CfC only (set from the command line; CNN / DNN / RFC keep the phase-1 settings)
REC = dict(epochs=None, patience=None, min_epochs=None, lr=1e-3, tag="")


def build_detector(kind, size, norm, name, normalizer, fs, seed, epochs):

    if kind == "dnn":
        return build_dnn_detector(name, normalizer, fs, size, TrainConfig(epochs=epochs, seed=seed))

    if kind == "cnn":
        return build_cnn_detector(name, normalizer, size, TrainConfig(epochs=epochs, seed=seed))

    if kind in ("gru", "cfc", "gruE", "cfcE", "cfc0E"):
        cfg = TrainConfig(epochs=REC["epochs"] or epochs, seed=seed)
        if REC["patience"] is not None:
            cfg.patience = REC["patience"]
        if REC["min_epochs"] is not None:
            cfg.min_epochs = REC["min_epochs"]
        base = "gru" if kind.startswith("gru") else "cfc"
        variant = "E" if kind.endswith("E") else "W"
        # cfc: dt = 1 / fs | cfcE: dt = sub-window step (about 5 s) | cfc0E: ncps default dt = 1 (ablation)
        dt = {"cfc": 1 / fs, "cfc0E": 1.0}.get(kind)
        return build_recurrent_detector(name, base, normalizer, size, cfg, lr=REC["lr"], variant=variant, fs=fs, dt=dt)

    if kind == "rfc":
        return build_rfc_detector(name, normalizer, fs, size, seed=seed)

    if kind == "amp":
        return AmplitudeDetector(name, size, normalizer, fs)

    raise ValueError(kind)


def make_saver(a, bundle, config, seed, det, fresh):
    """Fold callback for cross_validate_budgets: saves the fold's fitted detector with its thresholds."""

    def callback(k, chosen):

        cands = {id(c) for c, _ in chosen.values()}

        if len(cands) != 1:
            raise RuntimeError("checkpoints need exactly one candidate per run")

        thresholds = {b: t for b, (_, t) in chosen.items()}
        path = save_checkpoint(a.save_models, config, seed, k, det, thresholds)

        if a.verify_models:
            val = bundle.fold_split(k).validation
            probe = val[:: max(1, len(val) // 100)][:100]
            diff = verify_roundtrip(det, fresh(), a.save_models, config, seed, k, probe)
            print(f"  saved {path} (reload check: max |score diff| {diff:.2g})")

    return callback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["1k", "10k"])
    ap.add_argument("--norms", nargs="+", default=["N0", "N2"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.25, 2.0])
    ap.add_argument("--rfc", action="store_true")
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--cnn-sizes", nargs="+", default=["1k", "10k"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--rec", action="store_true", help="GRU and CfC on the raw window (variant W)")
    ap.add_argument("--rec-e", action="store_true", help="GRU and CfC on the 12-step envelope sequence (variant E)")
    ap.add_argument("--rec-sizes", nargs="+", default=["1k", "10k"])
    ap.add_argument("--rec-epochs", type=int, default=None, help="epoch cap for GRU / CfC (default: --epochs)")
    ap.add_argument("--rec-patience", type=int, default=None, help="early-stopping patience for GRU / CfC")
    ap.add_argument("--rec-min-epochs", type=int, default=None, help="no early stopping before this epoch (GRU / CfC)")
    ap.add_argument("--rec-lr", type=float, default=1e-3)
    ap.add_argument("--rec-tag", default="", help="suffix for GRU / CfC config names, e.g. long -> gru-1k-N0-long "
                                                   "(keeps runs with different schedules apart in analyze_results)")
    ap.add_argument("--only", nargs="+", choices=["dnn", "rfc", "cnn", "amp", "gru", "cfc", "gruE", "cfcE", "cfc0E"], default=None,
                    help="run only these kinds (default: dnn plus the flagged ones)")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--catalog", default=CATALOG_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--derived-dir", default=DERIVED_DIR)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--save-models", default=None, metavar="DIR",
                    help="save every fitted detector (per config / seed / fold) with its thresholds under DIR (N0 / N1 only)")
    ap.add_argument("--verify-models", action="store_true",
                    help="after saving, reload each checkpoint into a fresh detector and compare scores on 100 windows")
    a = ap.parse_args()

    from src.modules.evaluation.cv import cross_validate_budgets

    REC.update(epochs=a.rec_epochs, patience=a.rec_patience, min_epochs=a.rec_min_epochs, lr=a.rec_lr)

    bundle = DataPipeline(PipelineConfig(a.catalog, a.data_dir, a.derived_dir)).build()
    streams = bundle.eval_streams(sorted(bundle._processed_by_id))
    fs = streams[0].sampling_rate

    print("N2 scale tracks are computed on demand (about 100 s per fold the first time they are needed)")
    cache = ScaleCache.from_streams(streams, NormConfig("N2"))

    evaluator = make_evaluator()
    example = bundle.windows[0]

    records, resources, fit_logs, extra = [], {}, {}, {}

    for kind, size, norm in specs(a):

        config = f"{kind}-{size}-{norm}" + (f"-{a.rec_tag}" if kind in ("gru", "cfc", "gruE", "cfcE", "cfc0E") and a.rec_tag else "")
        t0 = time.time()

        n_seeds = 1 if kind == "amp" else a.seeds          # the control has no randomness

        for seed in range(n_seeds):

            name = f"{config}-s{seed}"
            normalizer = make_normalizer(norm, cache)

            det = build_detector(kind, size, norm, name, normalizer, fs, seed, a.epochs)

            saver = None

            if a.save_models:
                if norm in ("N0", "N1"):
                    saver = make_saver(a, bundle, config, seed, det,
                                       lambda: build_detector(kind, size, norm, name, make_normalizer(norm, cache),
                                                              fs, seed, a.epochs))
                elif seed == 0:
                    print(f"  note: {config} is not saved (checkpoints support N0 / N1 only)")

            res = cross_validate_budgets(bundle, [det], evaluator, a.budgets, name=name, verbose=False,
                                         fold_callback=saver)

            for budget, cv in res.items():
                pooled = cv.pooled
                records.append(dict(
                    config=config, budget=budget, seed=seed,
                    recall=float(pooled.recall("all")),
                    fa_per_day=float(pooled.fa_per_day),
                    test_ap=float(np.nanmean([f.test_ap for f in cv.folds])),
                    thresholds=[float(f.threshold) for f in cv.folds],
                    outcomes=extract_tallies(pooled),          # per-event outcomes for tests.analyze_results
                ))

            if hasattr(det, "fit_infos"):
                fit_logs.setdefault(config, []).append([
                    dict(best_epoch=i.best_epoch, best_val_ap=i.best_val_ap,
                         early_stopping=i.early_stopping, stopped_early=i.stopped_early,
                         n_val_waveforms=i.n_val_waveforms, note=i.note)
                    for i in det.fit_infos
                ])
            elif kind == "rfc":
                extra.setdefault(config, []).append(det.importances())   # last fold's forest

            if config not in resources:
                resources[config] = det.resource_report(example)

            print(f"  {name}: "
                  + ", ".join(f"{r['budget']:g}/day recall {r['recall']:.2f} FA {r['fa_per_day']:.2f}"
                              for r in records[-len(a.budgets):]))

        print(f"{config} done in {time.time() - t0:.0f} s")

    rows = summarise(records)
    print(format_table(rows, resources))

    for config, res in resources.items():
        print(f"\n{config}: {res.summary()}")
        for n in res.notes:
            print(f"  note: {n}")

    for config, imps in extra.items():
        mean = {k: float(np.mean([i[k] for i in imps])) for k in imps[0]}
        print(f"\n{config} feature importances (mean over seeds, last fold's forest):")
        for k, v in sorted(mean.items(), key=lambda kv: -kv[1]):
            print(f"  {k:9s} {v:.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        records=records, summary=rows, fit_logs=fit_logs, importances=extra,
        resources={k: vars(v) for k, v in resources.items()},
    ), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()