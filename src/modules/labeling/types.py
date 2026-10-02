from dataclasses import dataclass

import numpy as np

from src.modules.windowing import WindowRecord


@dataclass
class LabeledWindowRecord:
    window: WindowRecord
    label: int

    @property
    def data(self) -> np.ndarray:
        return self.window.data

    @property
    def start_time(self) -> float:
        return self.window.start_time

    @property
    def end_time(self) -> float:
        return self.window.end_time

    @property
    def event(self):
        return self.window.event