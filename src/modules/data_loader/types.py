from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class EventRecord:
    """
    Metadata describing one seismic event and its associated waveform.
    """

    filename: str
    waveform_path: Path

    time_abs: datetime
    time_rel: float

    evid: str
    event_type: str