from dataclasses import dataclass

from src.modules.labeling import LabeledWindowRecord


@dataclass
class SamplingResult:
    windows: list[LabeledWindowRecord]

    positive_count: int
    negative_count: int

    original_positive_count: int
    original_negative_count: int