from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.modules.data_loader import EventRecord


@dataclass
class ProcessedWaveformRecord:
    event: EventRecord
    data: np.ndarray
    sampling_rate: float
    start_time: datetime