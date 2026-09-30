from pathlib import Path

from src.modules.data_loader import EventRecord

from .types import ValidationResult, ValidationStatus


class Validator:
    """
    Validates EventRecord objects before waveform loading.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def validate_event(self, event: EventRecord) -> ValidationResult:
        """
        Validate a single event.
        """

        if event.waveform_path.exists():
            return ValidationResult(
                event=event,
                status=ValidationStatus.VALID,
                message="Waveform file found.",
            )

        return ValidationResult(
            event=event,
            status=ValidationStatus.WARNING,
            message=(
                "Expected waveform file was not found: "
                f"{event.waveform_path.name}"
            ),
        )

    def run(
        self,
        events: list[EventRecord],
    ) -> list[ValidationResult]:
        """
        Validate all events.
        """

        results = []

        for event in events:
            result = self.validate_event(event)
            results.append(result)

        return results