from dataclasses import dataclass
from enum import Enum

from src.modules.data_loader import EventRecord


class ValidationStatus(Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    event: EventRecord
    status: ValidationStatus
    message: str


@dataclass
class ValidationReport:
    results: list[ValidationResult]

    @property
    def valid_events(self) -> list[EventRecord]:
        return [
            result.event
            for result in self.results
            if result.status == ValidationStatus.VALID
        ]

    @property
    def rejected_events(self) -> list[ValidationResult]:
        return [
            result
            for result in self.results
            if result.status != ValidationStatus.VALID
        ]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def valid_count(self) -> int:
        return len(self.valid_events)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_events)