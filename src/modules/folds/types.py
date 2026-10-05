import csv
import os
from dataclasses import dataclass

from src.modules.dataset_builder import DatasetSplit
from src.modules.labeling import LabeledWindowRecord


@dataclass
class Fold:
    """One cross-validation fold. All lists hold PRIMARY waveform ids."""

    index: int
    train: list
    validation: list
    test: list


def make_folds(assignment: dict, n_folds: int) -> list:
    """
    Rotating scheme: for fold k the test set is fold k, the validation set
    is fold (k + 1) % n_folds and the training set is everything else.
    """

    folds = []

    for k in range(n_folds):

        val_fold = (k + 1) % n_folds

        test = sorted(w for w, f in assignment.items() if f == k)
        val = sorted(w for w, f in assignment.items() if f == val_fold)
        train = sorted(
            w for w, f in assignment.items() if f not in (k, val_fold)
        )

        folds.append(Fold(index=k, train=train, validation=val, test=test))

    return folds


@dataclass
class FoldSet:
    folds: list
    assignment: dict      # waveform id -> fold index (its TEST fold)
    strata: dict          # waveform id -> stratum name
    n_folds: int
    random_seed: int

    def split(
        self,
        labeled: list[LabeledWindowRecord],
        fold_index: int,
    ) -> DatasetSplit:
        """Window lists for one fold, in the DatasetSplit format."""

        f = self.folds[fold_index]

        train_set, val_set, test_set = (
            set(f.train), set(f.validation), set(f.test)
        )

        train, validation, test = [], [], []

        for w in labeled:

            wid = w.event.evid          # primary waveform id

            if wid in train_set:
                train.append(w)
            elif wid in val_set:
                validation.append(w)
            elif wid in test_set:
                test.append(w)
            else:
                raise RuntimeError(f"Unknown waveform id: {wid}")

        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
            train_events=list(f.train),
            validation_events=list(f.validation),
            test_events=list(f.test),
        )

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["waveform_id", "fold", "stratum", "n_folds", "seed"])
            for wid in sorted(self.assignment):
                w.writerow([
                    wid, self.assignment[wid], self.strata[wid],
                    self.n_folds, self.random_seed,
                ])

    @classmethod
    def load(cls, path: str) -> "FoldSet":

        assignment, strata = {}, {}
        n_folds = seed = None

        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                assignment[row["waveform_id"]] = int(row["fold"])
                strata[row["waveform_id"]] = row["stratum"]
                n_folds, seed = int(row["n_folds"]), int(row["seed"])

        return cls(
            folds=make_folds(assignment, n_folds),
            assignment=assignment,
            strata=strata,
            n_folds=n_folds,
            random_seed=seed,
        )