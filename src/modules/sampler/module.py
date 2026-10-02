import random

from src.modules.labeling import LabeledWindowRecord

from .types import SamplingResult


class Sampler:
    """
    Samples training windows while preserving all positive examples.

    Example:
        positive_to_negative_ratio = 5

        For every positive window, at most 5 negative
        windows will be selected.
    """

    def __init__(
        self,
        positive_to_negative_ratio: int = 5,
        random_seed: int = 42,
    ):
        if positive_to_negative_ratio <= 0:
            raise ValueError(
                "positive_to_negative_ratio must be greater than 0."
            )

        self.positive_to_negative_ratio = (
            positive_to_negative_ratio
        )

        self.random_seed = random_seed

    def run(
        self,
        windows: list[LabeledWindowRecord],
    ) -> SamplingResult:

        positive_windows = [
            window
            for window in windows
            if window.label == 1
        ]

        negative_windows = [
            window
            for window in windows
            if window.label == 0
        ]

        if not positive_windows:
            raise ValueError(
                "No positive windows found."
            )

        if not negative_windows:
            raise ValueError(
                "No negative windows found."
            )

        # ----------------------------------------------------
        # Determine number of negatives to keep
        # ----------------------------------------------------

        desired_negative_count = (
            len(positive_windows)
            * self.positive_to_negative_ratio
        )

        negative_count = min(
            desired_negative_count,
            len(negative_windows),
        )

        # ----------------------------------------------------
        # Randomly sample negatives
        # ----------------------------------------------------

        rng = random.Random(
            self.random_seed
        )

        sampled_negative_windows = rng.sample(
            negative_windows,
            negative_count,
        )

        # ----------------------------------------------------
        # Combine positive and negative windows
        # ----------------------------------------------------

        sampled_windows = (
            positive_windows
            + sampled_negative_windows
        )

        # Shuffle the final dataset
        rng.shuffle(sampled_windows)

        return SamplingResult(
            windows=sampled_windows,
            positive_count=len(positive_windows),
            negative_count=len(
                sampled_negative_windows
            ),
            original_positive_count=len(
                positive_windows
            ),
            original_negative_count=len(
                negative_windows
            ),
        )