from dataclasses import dataclass

import numpy as np

from .module import Evaluator, average_precision, build_series, pool_results
from .types import EvalResult

IGNORE = -1


@dataclass
class FoldRecord:
    fold: int
    candidate: str
    threshold: float
    val_recall: float
    val_fa_per_day: float
    test_ap: float                  # window-level average precision (test)
    result: EvalResult


@dataclass
class CVResult:
    name: str
    folds: list
    pooled: EvalResult

    def print_summary(self, evaluator: Evaluator, intervals: bool = True) -> None:

        print()
        print("#" * 60)
        print(self.name)
        print("#" * 60)

        for f in self.folds:
            print(
                f"fold {f.fold}: {f.candidate} | threshold {f.threshold:.4g} | "
                f"validation recall {f.val_recall:.2f} at "
                f"{f.val_fa_per_day:.2f} FA/day | test AP {f.test_ap:.3f}"
            )

        aps = [f.test_ap for f in self.folds if not np.isnan(f.test_ap)]

        if aps:
            print(f"Window-level average precision (mean of folds): "
                  f"{np.mean(aps):.3f}")

        print()

        Evaluator.print_result(
            self.pooled,
            evaluator.bootstrap(self.pooled) if intervals else None,
        )


def cross_validate_budgets(
    bundle,
    candidates: list,
    evaluator: Evaluator,
    budgets: list,
    name: str = "cross-validation",
    select_group: str = "all",
    verbose: bool = True,
) -> dict:
    """
    For every fold:
      1. each candidate (hyper-parameter setting) is fitted on the TRAINING
         windows and scored on the VALIDATION streams (once),
      2. for every false-alarm budget the threshold is chosen on the
         validation curve and the best candidate is kept (highest
         validation recall, ties -> fewer false alarms),
      3. each chosen (candidate, threshold) is applied ONCE to the TEST
         streams of that fold.
    Test results of all folds are pooled (every waveform is tested once).

    Returns {budget: CVResult}; budgets are false alarms per 24 scored hours.
    """

    budgets = list(budgets)
    folds_by_budget = {b: [] for b in budgets}

    for k in range(bundle.folds.n_folds):

        split = bundle.fold_split(k)

        val_streams = bundle.eval_streams(split.validation_events)
        test_streams = bundle.eval_streams(split.test_events)

        val_by_id = {s.waveform_id: s for s in val_streams}
        test_by_id = {s.waveform_id: s for s in test_streams}

        cache = []

        for cand in candidates:

            cand.fit(split.train)

            val_series = build_series(
                split.validation, cand.score(split.validation, val_by_id)
            )

            cache.append(
                (cand, val_series, evaluator.curve(val_streams, val_series))
            )

        test_cache = {}

        labels = np.array([w.label for w in split.test])
        keep = labels != IGNORE

        for b in budgets:

            best = None

            for cand, val_series, curve in cache:

                thr = evaluator.select_threshold(
                    curve, max_fa_per_day=b, group=select_group
                )

                vr = evaluator.evaluate(val_streams, val_series, thr)

                key = (
                    np.nan_to_num(vr.recall(select_group), nan=-1.0),
                    -np.nan_to_num(vr.fa_per_day, nan=np.inf),
                )

                if best is None or key > best[0]:
                    best = (key, cand, thr, vr)

            _, cand, thr, vr = best

            if id(cand) not in test_cache:
                scores = cand.score(split.test, test_by_id)
                test_cache[id(cand)] = (
                    build_series(split.test, scores),
                    average_precision(labels[keep], scores[keep]),
                )

            test_series, ap = test_cache[id(cand)]

            result = evaluator.evaluate(test_streams, test_series, thr)

            rec = FoldRecord(
                fold=k,
                candidate=cand.name,
                threshold=thr,
                val_recall=vr.recall(select_group),
                val_fa_per_day=vr.fa_per_day,
                test_ap=ap,
                result=result,
            )

            folds_by_budget[b].append(rec)

            if verbose:
                print(f"[{name} @ {b:g}/day] fold {k}: {cand.name} | "
                      f"threshold {thr:.4g} | val recall {rec.val_recall:.2f} "
                      f"| test recall {result.recall(select_group):.2f} | "
                      f"test FA/day {result.fa_per_day:.2f}")

    return {
        b: CVResult(
            name=f"{name} @ {b:g} FA/day",
            folds=folds,
            pooled=pool_results([f.result for f in folds]),
        )
        for b, folds in folds_by_budget.items()
    }


def cross_validate(
    bundle,
    candidates: list,
    evaluator: Evaluator,
    name: str = "cross-validation",
    select_group: str = "all",
    verbose: bool = True,
) -> CVResult:
    """Single budget: the one in evaluator.config.max_fa_per_day."""

    budget = evaluator.config.max_fa_per_day

    result = cross_validate_budgets(
        bundle, candidates, evaluator, [budget],
        name=name, select_group=select_group, verbose=verbose,
    )[budget]

    result.name = name

    return result