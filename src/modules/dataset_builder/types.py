from dataclasses import dataclass

from src.modules.labeling import LabeledWindowRecord


@dataclass
class DatasetSplit:
    train: list[LabeledWindowRecord]
    validation: list[LabeledWindowRecord]
    test: list[LabeledWindowRecord]

    train_events: list[str]
    validation_events: list[str]
    test_events: list[str]