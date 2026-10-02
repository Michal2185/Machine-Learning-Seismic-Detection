import random

from src.modules.labeling import LabeledWindowRecord

from .types import DatasetSplit


class DatasetBuilder:

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        total = (
            train_ratio
            + validation_ratio
            + test_ratio
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Train, validation and test ratios "
                "must sum to 1.0."
            )

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def run(
        self,
        windows: list[LabeledWindowRecord],
    ) -> DatasetSplit:

        # ----------------------------------------------------
        # Find unique event IDs
        # ----------------------------------------------------

        event_ids = sorted(
            {
                window.event.evid
                for window in windows
            }
        )

        if not event_ids:
            raise ValueError(
                "No events found in dataset."
            )

        # ----------------------------------------------------
        # Shuffle events
        # ----------------------------------------------------

        rng = random.Random(
            self.random_seed
        )

        rng.shuffle(event_ids)

        # ----------------------------------------------------
        # Calculate split sizes
        # ----------------------------------------------------

        n_events = len(event_ids)

        n_train = round(
            n_events * self.train_ratio
        )

        n_validation = round(
            n_events * self.validation_ratio
        )

        # Everything remaining goes to test.
        n_test = (
            n_events
            - n_train
            - n_validation
        )

        if n_train == 0:
            raise ValueError(
                "Training split contains no events."
            )

        if n_validation == 0:
            raise ValueError(
                "Validation split contains no events."
            )

        if n_test == 0:
            raise ValueError(
                "Test split contains no events."
            )

        # ----------------------------------------------------
        # Split event IDs
        # ----------------------------------------------------

        train_events = event_ids[
            :n_train
        ]

        validation_events = event_ids[
            n_train:
            n_train + n_validation
        ]

        test_events = event_ids[
            n_train + n_validation:
        ]

        # ----------------------------------------------------
        # Convert to sets for fast lookup
        # ----------------------------------------------------

        train_set = set(train_events)
        validation_set = set(validation_events)
        test_set = set(test_events)

        # ----------------------------------------------------
        # Assign windows
        # ----------------------------------------------------

        train = []
        validation = []
        test = []

        for window in windows:

            event_id = window.event.evid

            if event_id in train_set:
                train.append(window)

            elif event_id in validation_set:
                validation.append(window)

            elif event_id in test_set:
                test.append(window)

            else:
                raise RuntimeError(
                    f"Unknown event ID: {event_id}"
                )

        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
            train_events=train_events,
            validation_events=validation_events,
            test_events=test_events,
        )