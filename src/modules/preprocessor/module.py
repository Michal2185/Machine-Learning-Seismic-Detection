import numpy as np

from src.modules.waveform_loader import WaveformRecord

from .types import ProcessedWaveformRecord


class Preprocessor:
    """
    Performs basic preparation of loaded seismic waveforms.
    """

    def process(
        self,
        record: WaveformRecord,
    ) -> ProcessedWaveformRecord:

        if len(record.stream) != 1:
            raise ValueError(
                f"Expected exactly one trace, "
                f"got {len(record.stream)} "
                f"for event {record.event.evid}"
            )

        trace = record.stream[0]

        data = np.asarray(
            trace.data,
            dtype=np.float32,
        )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"Waveform contains NaN or infinite values "
                f"for event {record.event.evid}"
            )

        return ProcessedWaveformRecord(
            event=record.event,
            data=data,
            sampling_rate=trace.stats.sampling_rate,
            start_time=trace.stats.starttime,
        )

    def run(
        self,
        records: list[WaveformRecord],
    ) -> list[ProcessedWaveformRecord]:

        processed = []

        for record in records:
            processed_record = self.process(record)
            processed.append(processed_record)

        return processed