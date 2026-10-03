from dataclasses import dataclass


@dataclass
class EpochResult:
    epoch: int

    train_loss: float
    validation_loss: float

    train_accuracy: float
    validation_accuracy: float


@dataclass
class TrainingResult:
    epochs: list[EpochResult]
    best_epoch: int
    best_validation_loss: float


@dataclass
class EvaluationResult:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    precision: float
    recall: float
    f1: float

    accuracy: float

    total_samples: int
    positive_samples: int
    negative_samples: int