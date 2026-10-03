from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .types import EpochResult, TrainingResult


class Trainer:
    """
    Handles training and validation of a binary classifier.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        checkpoint_path: str = "checkpoints/seismic_cnn_best.pt",
        device: str | None = None,
    ):
        self.model = model

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.loss_function = (
            nn.BCEWithLogitsLoss()
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def _run_epoch(
        self,
        loader,
        training: bool,
    ):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:

            x = x.to(self.device)
            y = y.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):

                logits = self.model(x)

                loss = self.loss_function(
                    logits,
                    y,
                )

                if training:
                    loss.backward()
                    self.optimizer.step()

            total_loss += (
                loss.item() * x.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == y
            ).sum().item()

            total += y.numel()

        average_loss = (
            total_loss / len(loader.dataset)
        )

        accuracy = correct / total

        return average_loss, accuracy

    def fit(
        self,
        train_dataset,
        validation_dataset,
    ):

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        history = []

        best_validation_loss = float("inf")
        best_epoch = 0

        print()
        print("=" * 60)
        print("CNN TRAINING")
        print("=" * 60)

        print(
            f"Device:        {self.device}"
        )

        print(
            f"Train samples: {len(train_dataset)}"
        )

        print(
            f"Validation:    {len(validation_dataset)}"
        )

        print(
            f"Batch size:    {self.batch_size}"
        )

        print(
            f"Epochs:        {self.epochs}"
        )

        print()

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            train_loss, train_accuracy = (
                self._run_epoch(
                    train_loader,
                    training=True,
                )
            )

            validation_loss, validation_accuracy = (
                self._run_epoch(
                    validation_loader,
                    training=False,
                )
            )

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_accuracy=train_accuracy,
                validation_accuracy=validation_accuracy,
            )

            history.append(result)

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {validation_accuracy:.4f}"
            )

            # Save the best model
            if validation_loss < best_validation_loss:

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch

                self.checkpoint_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            self.model.state_dict(),
                        "optimizer_state_dict":
                            self.optimizer.state_dict(),
                        "validation_loss":
                            validation_loss,
                    },
                    self.checkpoint_path,
                )

        return TrainingResult(
            epochs=history,
            best_epoch=best_epoch,
            best_validation_loss=(
                best_validation_loss
            ),
        )