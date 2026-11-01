"""
python -m tests.time_recurrent

Training step and scoring throughput of GRU-W / CfC-W on CPU, extrapolated to one nested-CV run.
Per fold and candidate: about 700 training windows per epoch x up to 60 epochs, plus scoring
about 240 000 windows (validation + test streams).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.models.recurrent import PRESETS, make_cfc, make_gru
from src.modules.training.learners import TorchLearner

EPOCH_WINDOWS, EPOCHS, SCORE_WINDOWS, FOLDS, SEEDS = 700, 60, 240_000, 5, 5


def main():
    x = np.random.default_rng(0).normal(0, 1, (256, 398, 1)).astype(np.float32)
    y = (np.arange(256) % 2).astype(np.float32)

    for size in ("1k", "10k"):
        for kind, make in (("gru", make_gru), ("cfc", lambda F, h: make_cfc(F, h, dt=1 / 6.625))):
            h = PRESETS[1][size][kind]
            lr = TorchLearner(lambda: make(1, h))
            lr.reset(0)

            t = time.time(); lr.train_step(x[:64], y[:64], 5.0); t_train = (time.time() - t) / 64   # s per window
            t = time.time(); lr.predict(x);                      t_pred = (time.time() - t) / 256

            train = t_train * EPOCH_WINDOWS * EPOCHS
            score = t_pred * SCORE_WINDOWS
            total = (train + score) * FOLDS * SEEDS / 3600
            print(f"{kind}-{size} h={h:3d}: train {t_train * 1e3:.2f} ms/window, score {t_pred * 1e3:.3f} ms/window "
                  f"-> per fold {train / 60:.1f} + {score / 60:.1f} min, 5 folds x 5 seeds = {total:.1f} h")


if __name__ == "__main__":
    main()