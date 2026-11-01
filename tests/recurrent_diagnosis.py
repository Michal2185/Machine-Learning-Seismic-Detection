"""
Why does the CfC-W 1k peak early and then lose validation AP?  One fold, early stopping DISABLED, full trajectories.

    python -m tests.diagnose_recurrent                       # fold 1, seed 0, 60 epochs, 4 configurations
    python -m tests.diagnose_recurrent --fold 2 --seeds 0 1
    python -m tests.diagnose_recurrent --configs gru:1e-3 cfc:1e-3 cfc:1e-4

A configuration is kind:lr. Every fit uses the harness unchanged (same Sampler, same early-stopping set
carved out of the TRAINING waveforms, same weighting); only patience is set so large that all epochs run and
the best epoch is still restored at the end. Nothing here touches the validation or test streams of the fold
except one small score() check on validation windows at the end. Writes results/diagnose_recurrent.json.

What to look for in the output:
  loss falls, AP peaks early then drops    -> overfitting or too high learning rate (lower lr / more weight decay)
  loss flat or noisy, AP never above ~0.3  -> not learning (scale, init or lr too low)
  loss falls, AP keeps rising              -> the 1k CfC simply needs more epochs than the harness gives it
  logit spread at the best epoch near 0    -> collapsed to a constant output
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.models import build_recurrent_detector
from src.modules.normalization import NormConfig, Normalizer
from src.modules.pipeline import DataPipeline, PipelineConfig
from src.modules.training import TrainConfig
from tests.run_dnn_features import CATALOG_PATH, DATA_DIR, DERIVED_DIR

OUT_PATH = "results/diagnose_recurrent.json"


def history_rows(info):
    """(epoch, loss, val_ap, improved) from the harness EpochRecord, whatever its field names are."""

    rows = []

    for h in info.history:
        try:
            t = dataclasses.astuple(h)
        except TypeError:
            t = tuple(vars(h).values())
        rows.append((int(t[0]), float(t[1]), float(t[2]), bool(t[3])))

    return rows


def make_config(epochs, seed):

    cfg = TrainConfig(epochs=epochs, seed=seed)
    cfg.patience = 10 ** 6            # never stop early: we want the whole trajectory
    cfg.min_epochs = epochs

    return cfg


def spread(det, windows):
    """Logit mean of positives / background and overall std on a thin subsample of validation windows."""

    sub = list(windows)[::20]
    s = det.score(sub, None)
    y = np.array([w.label for w in sub])

    pos, neg = s[y == 1], s[y == 0]

    return dict(
        n=int(len(sub)),
        std=float(np.std(s[y >= 0])),
        mean_pos=float(pos.mean()) if len(pos) else float("nan"),
        mean_neg=float(neg.mean()) if len(neg) else float("nan"),
    )


def fmt(xs, nd=2):
    return " ".join(f"{x:.{nd}f}" for x in xs)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--size", default="1k")
    ap.add_argument("--configs", nargs="+", default=["gru:1e-3", "cfc:1e-3", "cfc:3e-4", "cfc:1e-4"])
    ap.add_argument("--every", type=int, default=3, help="print every n-th epoch")
    ap.add_argument("--catalog", default=CATALOG_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--derived-dir", default=DERIVED_DIR)
    ap.add_argument("--out", default=OUT_PATH)
    a = ap.parse_args()

    bundle = DataPipeline(PipelineConfig(a.catalog, a.data_dir, a.derived_dir)).build()
    split = bundle.fold_split(a.fold)
    fs = bundle.eval_stream(split.train_events[0] if hasattr(split, "train_events")
                            else split.validation_events[0]).sampling_rate

    print(f"fold {a.fold}: {len(split.train)} training windows, "
          f"{sum(w.label == 1 for w in split.train)} positive; fs {fs:g} Hz")

    out = []

    for spec in a.configs:

        kind, lr = spec.split(":")
        lr = float(lr)

        for seed in a.seeds:

            name = f"{kind}-{a.size}-lr{lr:g}-s{seed}"
            det = build_recurrent_detector(
                name, kind, Normalizer(NormConfig("N0")), a.size, make_config(a.epochs, seed),
                dt=1 / fs if kind == "cfc" else None, lr=lr,
            )

            t0 = time.time()
            det.fit(split.train)
            info = det.info
            rows = history_rows(info)

            loss = np.array([r[1] for r in rows])
            aps = np.array([r[2] for r in rows])

            best = int(np.nanargmax(aps))
            tail = float(np.nanmean(aps[-10:]))
            sp = spread(det, split.validation)           # weights = best epoch (restored by fit)

            print(f"\n=== {name}   ({time.time() - t0:.0f} s, {info.n_val_waveforms} held-out waveforms)")
            print(f"epochs {', '.join(str(r[0]) for r in rows[::a.every])}")
            print(f"loss   {fmt(loss[::a.every], 3)}")
            print(f"val AP {fmt(aps[::a.every])}")
            print(f"best AP {aps[best]:.2f} at epoch {best}; AP epoch 0 {aps[0]:.2f}; mean AP of last 10 epochs {tail:.2f}; "
                  f"loss first {loss[0]:.3f} -> last {loss[-1]:.3f} (min {loss.min():.3f} at epoch {int(loss.argmin())})")
            print(f"logits at best epoch (validation subsample): std {sp['std']:.2f}, "
                  f"mean positive {sp['mean_pos']:.2f}, mean background {sp['mean_neg']:.2f}")

            out.append(dict(config=spec, name=name, seed=seed, fold=a.fold, lr=lr, kind=kind,
                            loss=loss.tolist(), val_ap=aps.tolist(), best_epoch=best,
                            best_ap=float(aps[best]), tail_ap=tail, logits=sp))

    print("\nsummary (best AP / epoch, mean AP last 10 epochs, loss decrease):")

    for r in out:
        print(f"  {r['name']:24s} best {r['best_ap']:.2f} @ {r['best_epoch']:2d}   tail {r['tail_ap']:.2f}   "
              f"loss {r['loss'][0]:.3f} -> {r['loss'][-1]:.3f}")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()