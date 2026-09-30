from pathlib import Path

from obspy import read

from src.modules.data_loader import EventRecord

from .types import WaveformRecord


class WaveformLoader:
    """
    Loads MiniSEED waveforms associated with EventRecord objects.
    """

    def load(self, event: EventRecord) -> WaveformRecord:
        """
        Load the MiniSEED waveform associated with one event.
        """

        stream = read(event.waveform_path)

        return WaveformRecord(
            event=event,
            stream=stream,
        )

    def run(self, events: list[EventRecord]) -> list[WaveformRecord]:
        """
        Load waveforms for a collection of events.
        """

        records = []

        for event in events:
            record = self.load(event)
            records.append(record)

        return records