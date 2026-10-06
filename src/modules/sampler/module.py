import numpy as np

from src.modules.labeling import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    LabeledWindowRecord,
)

from .types import SamplingResult


class Sampler:
    """
    Builds a class-balanced TRAINING subset.

      * keeps ALL positive windows
      * draws `negatives_per_positive` x n_positive negatives at random,
        without replacement
      * never uses ignore windows (label -1)

    Use on the training split only. Validation and test stay at natural
    prevalence.

    `epoch` changes the random draw (deterministically: the same
    seed + epoch always gives the same subset), so calling run() once per
    epoch shows the model different background windows each time instead
    of the same few every epoch.
    """

    def __init__(
        self,
        negatives_per_positive: float = 5.0,
        random_seed: int = 42,
    ):

        if negatives_per_positive <= 0:
            raise ValueError("negatives_per_positive must be > 0")

        self.negatives_per_positive = negatives_per_positive
        self.random_seed = random_seed

    def run(
        self,
        windows: list[LabeledWindowRecord],
        epoch: int = 0,
    ) -> SamplingResult:

        positives = []
        negatives = []
        n_ignored = 0

        for w in windows:

            if w.label == POSITIVE:
                positives.append(w)

            elif w.label == NEGATIVE:
                negatives.append(w)

            elif w.label == IGNORE:
                n_ignored += 1

            else:
                raise ValueError(f"Unknown label: {w.label}")

        if not positives:
            raise ValueError("No positive windows to sample from.")

        n_negative = min(
            len(negatives),
            round(self.negatives_per_positive * len(positives)),
        )

        rng = np.random.default_rng([self.random_seed, epoch])

        chosen = rng.choice(
            len(negatives),
            size=n_negative,
            replace=False,
        )

        sampled = positives + [negatives[i] for i in chosen]

        order = rng.permutation(len(sampled))

        return SamplingResult(
            windows=[sampled[i] for i in order],
            epoch=epoch,
            n_positive=len(positives),
            n_negative=n_negative,
            n_negative_available=len(negatives),
            n_ignored_dropped=n_ignored,
        )