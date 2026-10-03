from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


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


class Evaluator:
    """
    Evaluates a trained binary classifier.

    The evaluator operates on an already separated dataset.
    It does not modify the dataset or model.
    """

    def __init__(
        self,
        model,
        batch_size: int = 32,
        device: str | None = None,
        threshold: float = 0.5,
    ):
        self.model = model
        self.batch_size = batch_size
        self.threshold = threshold

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

    def evaluate(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                probabilities = torch.sigmoid(
                    logits
                )

                predictions = (
                    probabilities >= self.threshold
                ).float()

                true_positive += (
                    ((predictions == 1) & (y == 1))
                    .sum()
                    .item()
                )

                true_negative += (
                    ((predictions == 0) & (y == 0))
                    .sum()
                    .item()
                )

                false_positive += (
                    ((predictions == 1) & (y == 0))
                    .sum()
                    .item()
                )

                false_negative += (
                    ((predictions == 0) & (y == 1))
                    .sum()
                    .item()
                )

        total = (
            true_positive
            + true_negative
            + false_positive
            + false_negative
        )

        positive_samples = (
            true_positive + false_negative
        )

        negative_samples = (
            true_negative + false_positive
        )

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        if true_positive + false_positive > 0:
            precision = (
                true_positive
                / (true_positive + false_positive)
            )
        else:
            precision = 0.0

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        if true_positive + false_negative > 0:
            recall = (
                true_positive
                / (true_positive + false_negative)
            )
        else:
            recall = 0.0

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        if precision + recall > 0:
            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = (
            (true_positive + true_negative)
            / total
            if total > 0
            else 0.0
        )

        return EvaluationResult(
            true_positive=true_positive,
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            total_samples=total,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
        )