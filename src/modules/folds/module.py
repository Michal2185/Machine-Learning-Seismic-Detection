import random
from collections import defaultdict

from .types import FoldSet, make_folds


class FoldBuilder:
    """
    Grouped, stratified k-fold over DISTINCT WAVEFORMS.

    * Grouped: all catalogue events of a waveform stay together (and so do
      all windows), so nothing leaks between train, validation and test.
    * Stratified: every waveform gets a stratum and the strata are dealt
      round-robin over the folds, so rare event types are spread as evenly
      as possible:
          shallow      waveform holds a shallow moonquake
          deep         holds a deep moonquake (no shallow)
          impact_pos   impact events only, at least one with positives
          impact_weak  impact events only, none with positives
                       (no usable visible onset)
    * Rotating validation: fold k is the test set, fold k+1 the validation
      set, the rest is training.

    With only a handful of shallow / deep events some folds will contain
    none of them. Pool the out-of-fold predictions over all folds when
    reporting per-type results.
    """

    def __init__(self, n_folds: int = 5, random_seed: int = 42):

        if n_folds < 3:
            raise ValueError(
                "n_folds must be >= 3 (train, validation and test folds)"
            )

        self.n_folds = n_folds
        self.random_seed = random_seed

    @staticmethod
    def stratum_of(events, labeling=None) -> str:

        types = {e.event_type for e in events}

        if "shallow_mq" in types:
            return "shallow"

        if "deep_mq" in types:
            return "deep"

        if types - {"impact_mq"}:
            return "other"

        has_positive = any(
            labeling is None or labeling.positive_rule(e) is not None
            for e in events
        )

        return "impact_pos" if has_positive else "impact_weak"

    def run(self, event_groups: dict, labeling=None) -> FoldSet:
        """
        event_groups: primary waveform id -> catalogue events on it
        (WaveformGrouper.run(...).event_groups).
        labeling: the Labeling instance (decides which events have positives).
        """

        if len(event_groups) < self.n_folds:
            raise ValueError("Fewer waveforms than folds.")

        strata = {
            wid: self.stratum_of(events, labeling)
            for wid, events in event_groups.items()
        }

        by_stratum = defaultdict(list)

        for wid in sorted(event_groups):
            by_stratum[strata[wid]].append(wid)

        rng = random.Random(self.random_seed)

        assignment = {}
        counter = 0

        # the counter runs on across strata, so the fold sizes stay
        # balanced (differ by at most one) and each stratum is spread
        # over consecutive folds
        for name in sorted(by_stratum, key=lambda s: (len(by_stratum[s]), s)):

            ids = by_stratum[name]
            rng.shuffle(ids)

            for wid in ids:
                assignment[wid] = counter % self.n_folds
                counter += 1

        return FoldSet(
            folds=make_folds(assignment, self.n_folds),
            assignment=assignment,
            strata=strata,
            n_folds=self.n_folds,
            random_seed=self.random_seed,
        )