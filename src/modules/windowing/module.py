from dataclasses import dataclass

import numpy as np

from src.modules.preprocessor import ProcessedWaveformRecord

from .types import WindowRecord


@dataclass
class WindowConfig:
    window_length: float
    step_size: float


class Windowing:
    """
    Splits processed seismic waveforms into fixed-length windows.
    """

    def __init__(self, config: WindowConfig):
        self.config = config

    def process(
        self,
        record: ProcessedWaveformRecord,
    ) -> list[WindowRecord]:

        sampling_rate = record.sampling_rate
        data = record.data

        window_samples = int(
            round(
                self.config.window_length
                * sampling_rate
            )
        )

        step_samples = int(
            round(
                self.config.step_size
                * sampling_rate
            )
        )

        if window_samples <= 0:
            raise ValueError(
                "Window length must produce at least one sample."
            )

        if step_samples <= 0:
            raise ValueError(
                "Step size must produce at least one sample."
            )

        if window_samples > len(data):
            raise ValueError(
                f"Window length ({self.config.window_length}s) "
                f"is longer than the waveform."
            )

        windows = []

        for start_sample in range(
            0,
            len(data) - window_samples + 1,
            step_samples,
        ):

            end_sample = (
                start_sample + window_samples
            )

            window_data = data[
                start_sample:end_sample
            ]

            start_time = (
                start_sample / sampling_rate
            )

            end_time = (
                end_sample / sampling_rate
            )

            windows.append(
                WindowRecord(
                    event=record.event,
                    data=window_data.copy(),
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        return windows

    def run(
        self,
        records: list[ProcessedWaveformRecord],
    ) -> list[WindowRecord]:

        windows = []

        for record in records:
            windows.extend(
                self.process(record)
            )

        return windows