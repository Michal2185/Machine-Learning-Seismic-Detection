import json
import os
from dataclasses import dataclass, field

import numpy as np

from src.modules.evaluation import build_series

from .sta_lta import StaLtaDetector, sta_lta_grid


@dataclass
class OperatingPoint:
    name: str                    # e.g. "strict" / "lenient"
    budget_fa_per_day: float     # false alarms per 24 scored hours it was set for
    threshold: float             # STA/LTA ratio


@dataclass
class FrozenSpec:
    """
    A frozen detector: ONE STA/LTA parameter set plus one threshold per
    operating point, chosen once on source waveforms and never retuned.
    """

    sta_s: float
    lta_s: float
    feature: str
    mask_gaps: bool = True
    delay_lta: bool = False
    min_lta_s: float = 600.0
    points: dict = field(default_factory=dict)     # name -> OperatingPoint
    info: dict = field(default_factory=dict)       # how it was selected

    def detector(self) -> StaLtaDetector:
        return StaLtaDetector(
            self.sta_s,
            self.lta_s,
            self.feature,
            mask_gaps=self.mask_gaps,
            delay_lta=self.delay_lta,
            min_lta_s=self.min_lta_s,
        )

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sta_s": self.sta_s,
            "lta_s": self.lta_s,
            "feature": self.feature,
            "mask_gaps": self.mask_gaps,
            "delay_lta": self.delay_lta,
            "min_lta_s": self.min_lta_s,
            "points": [
                {
                    "name": p.name,
                    "budget_fa_per_day": p.budget_fa_per_day,
                    "threshold": p.threshold,
                }
                for p in self.points.values()
            ],
            "info": self.info,
        }

    def save(self, path: str) -> None:

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "FrozenSpec":

        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)

        return cls(
            sta_s=d["sta_s"],
            lta_s=d["lta_s"],
            feature=d["feature"],
            mask_gaps=d["mask_gaps"],
            delay_lta=d["delay_lta"],
            min_lta_s=d["min_lta_s"],
            points={
                p["name"]: OperatingPoint(
                    p["name"], p["budget_fa_per_day"], p["threshold"]
                )
                for p in d["points"]
            },
            info=d.get("info", {}),
        )

    def describe(self) -> str:

        pts = ", ".join(
            f"{p.name}: threshold {p.threshold:.4g} "
            f"(budget {p.budget_fa_per_day:g}/day)"
            for p in self.points.values()
        )

        return (
            f"STA {self.sta_s:g} s / LTA {self.lta_s:g} s, {self.feature}, "
            f"gaps {'masked' if self.mask_gaps else 'not masked'} | {pts}"
        )


def freeze_sta_lta_sets(
    windows,
    streams_by_id: dict,
    evaluator,
    budgets: dict,
    sets: dict,
    grid: list | None = None,
    verbose: bool = True,
) -> dict:
    """
    Choose ONE STA/LTA parameter set and one threshold per operating point
    for each selection set.

    budgets : {point name: false alarms per 24 scored hours}
    sets    : {set name: [waveform ids used for the selection]}
              e.g. {"all": all ids, "fold0": train+validation ids of fold 0}
              Every set is selected independently, from its own waveforms
              only.
    Objective: mean recall (all events) over the operating points, each at
    its own threshold; ties -> fewer false alarms. One parameter set is
    shared by all operating points (one algorithm, a threshold per mission
    phase).

    Scores are computed once per candidate for all windows and reused for
    every set. Returns {set name: FrozenSpec}.
    """

    grid = list(grid) if grid is not None else sta_lta_grid()

    curves = {name: [] for name in sets}

    for n, cand in enumerate(grid):

        series = build_series(windows, cand.score(windows, streams_by_id))

        for name, ids in sets.items():
            curves[name].append(
                evaluator.curve(
                    [streams_by_id[i] for i in ids],
                    {i: series[i] for i in ids},
                )
            )

        if verbose and ((n + 1) % 10 == 0 or n + 1 == len(grid)):
            print(f"  scored {n + 1}/{len(grid)} candidates", flush=True)

    specs = {}

    for name, ids in sets.items():

        best = None

        for cand, curve in zip(grid, curves[name]):

            thresholds, recalls, fas = {}, [], []

            for pname, budget in budgets.items():

                t = evaluator.select_threshold(curve, max_fa_per_day=budget)
                i = int(np.flatnonzero(curve.thresholds == t)[0])

                thresholds[pname] = float(t)
                recalls.append(curve.recall_all[i])
                fas.append(curve.fa_per_day[i])

            key = (
                float(np.mean(np.nan_to_num(recalls, nan=-1.0))),
                -float(np.mean(np.nan_to_num(fas, nan=np.inf))),
            )

            if best is None or key > best[0]:
                best = (key, cand, thresholds, recalls, fas)

        _, cand, thresholds, recalls, fas = best

        specs[name] = FrozenSpec(
            sta_s=cand.sta_s,
            lta_s=cand.lta_s,
            feature=cand.feature,
            mask_gaps=cand.mask_gaps,
            delay_lta=cand.delay_lta,
            min_lta_s=cand.min_lta_s,
            points={
                pname: OperatingPoint(pname, float(b), thresholds[pname])
                for pname, b in budgets.items()
            },
            info={
                "selection_waveforms": len(ids),
                "grid_size": len(grid),
                "objective": "mean recall (all events) over the operating points",
                "selection_recall": {
                    p: float(r) for p, r in zip(budgets, recalls)
                },
                "selection_fa_per_day": {
                    p: float(f) for p, f in zip(budgets, fas)
                },
            },
        )

    return specs


def evaluate_frozen(spec: FrozenSpec, windows, streams, evaluator) -> dict:
    """
    Apply a frozen detector to waveforms. {point name: EvalResult}.
    Nothing is retuned: the parameters and thresholds are used as stored.
    """

    by_id = {s.waveform_id: s for s in streams}
    wins = [w for w in windows if w.event.evid in by_id]

    series = build_series(wins, spec.detector().score(wins, by_id))

    return {
        p.name: evaluator.evaluate(streams, series, p.threshold)
        for p in spec.points.values()
    }