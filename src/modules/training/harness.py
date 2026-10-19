import numpy as np

from src.modules.detectors.types import Detector

from .metrics import average_precision
from .split import split_by_waveform
from .types import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    EpochRecord,
    FitInfo,
    TrainConfig,
)


class LearnerDetector(Detector):
    """
    Detector that trains any Learner (torch or numpy) on window inputs.

      fit(train_windows)  fresh weights; N0 / feature scaler fitted on the
                          training windows; a share of the TRAINING
                          waveforms is held out for early stopping; each
                          epoch draws NEW background windows with the
                          Sampler (all positives kept, ignore windows never
                          used); the epoch with the best validation average
                          precision is restored.
      score(windows, _)   one logit per window (same order); thresholds are
                          chosen by the evaluator, never here.

    cross_validate_budgets calls fit() on the same object for every fold,
    so everything learned is reset at the start of fit().
    """

    def __init__(self, name: str, learner, input_builder, config: TrainConfig | None = None,
                 sampler_factory=None):

        self.name = name
        self.learner = learner
        self.inputs = input_builder
        self.config = config or TrainConfig()
        self.sampler_factory = sampler_factory
        self.info: FitInfo | None = None
        self._fitted = False

    # ------------------------------------------------------------------

    def _make_sampler(self):

        if self.sampler_factory is not None:
            return self.sampler_factory(self.config)

        from src.modules.sampler.module import Sampler

        return Sampler(
            negatives_per_positive=self.config.negatives_per_positive,
            random_seed=self.config.seed,
        )

    def _validation_set(self, val_windows):

        pos = [w for w in val_windows if w.label == POSITIVE]
        neg = [w for w in val_windows if w.label == NEGATIVE][:: self.config.val_neg_stride]

        sel = pos + neg
        y = np.array([1.0] * len(pos) + [0.0] * len(neg))

        return sel, y

    def fit(self, train_windows) -> "LearnerDetector":

        cfg = self.config
        train_windows = list(train_windows)

        self.learner.reset(cfg.seed)
        self.inputs.fit(train_windows)

        train_part, val_part, val_ids = split_by_waveform(
            train_windows, cfg.val_fraction, cfg.seed
        )

        val_x = val_y = None
        note = ""

        if val_part:
            sel, val_y = self._validation_set(val_part)

            if val_y.sum() > 0 and len(sel) > val_y.sum():
                val_x = self.inputs.build(sel)
            else:
                note = "held-out waveforms have no positives: early stopping disabled"
                train_part = train_windows
                val_ids = []
        else:
            note = "early stopping disabled (val_fraction 0 or a single waveform)"

        early = val_x is not None

        sampler = self._make_sampler()

        w_pos = cfg.effective_pos_weight
        history = []
        best_ap, best_epoch, best_state = -np.inf, -1, None
        stopped = False

        for epoch in range(cfg.epochs):

            drawn = sampler.run(train_part, epoch)
            x = self.inputs.build(drawn.windows)
            y = np.array([w.label for w in drawn.windows], dtype=np.float32)

            order = np.random.default_rng([cfg.seed, epoch, 1]).permutation(len(y))

            total = 0.0

            for i in range(0, len(y), cfg.batch_size):
                idx = order[i:i + cfg.batch_size]
                total += self.learner.train_step(x[idx], y[idx], w_pos) * len(idx)

            loss = total / max(1, len(y))

            ap = float("nan")
            improved = False

            if early:
                ap = average_precision(val_y, self.learner.predict(val_x))
                improved = ap > best_ap + 1e-12

                if improved:
                    best_ap, best_epoch, best_state = ap, epoch, self.learner.get_state()

            history.append(EpochRecord(epoch, loss, ap, improved))

            if early and epoch + 1 >= cfg.min_epochs and epoch - best_epoch >= cfg.patience:
                stopped = True
                break

        if early:
            self.learner.set_state(best_state)
        else:
            best_epoch, best_ap = len(history) - 1, float("nan")

        self.info = FitInfo(
            n_train_waveforms=len({w.event.evid for w in train_part}),
            n_val_waveforms=len(val_ids),
            val_waveforms=val_ids,
            early_stopping=early,
            best_epoch=best_epoch,
            best_val_ap=best_ap if early else float("nan"),
            stopped_early=stopped,
            history=history,
            note=note,
        )

        self._fitted = True

        return self

    # ------------------------------------------------------------------

    def score(self, windows, streams_by_id=None) -> np.ndarray:

        if not self._fitted:
            raise RuntimeError("detector is not fitted")

        windows = list(windows)
        out = []

        for i in range(0, len(windows), self.config.score_batch):
            x = self.inputs.build(windows[i:i + self.config.score_batch])
            out.append(np.asarray(self.learner.predict(x), dtype=np.float64))

        return np.concatenate(out) if out else np.zeros(0)

    def resource_report(self, example_window, n_samples: int | None = None):
        """Static cost for one window (needs a fitted detector)."""

        if not self._fitted:
            raise RuntimeError("detector is not fitted")

        n = n_samples or len(example_window.data)
        ops = self.inputs.ops_per_window(n)

        return self.learner.resources(self.inputs.build([example_window]), input_ops=ops)