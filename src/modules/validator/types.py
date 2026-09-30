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