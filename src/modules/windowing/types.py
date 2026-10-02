from dataclasses import dataclass

import numpy as np

from src.modules.data_loader import EventRecord


@dataclass
class WindowRecord:
    event: EventRecord
    data: np.ndarray

    start_sample: int
    end_sample: int

    start_time: float
    end_time: float