from obspy import read

from src.modules.data_loader import EventRecord

from .types import WaveformRecord


class WaveformLoader:
    """
    Loads MiniSEED waveforms for validated events.
    """

    def load(self, event: EventRecord) -> WaveformRecord:
        """
        Load the waveform associated with one event.
        """

        if not event.waveform_path.exists():
            raise FileNotFoundError(
                f"Waveform file does not exist:\n"
                f"  {event.waveform_path}\n"
                f"Event ID: {event.evid}"
            )

        stream = read(event.waveform_path)

        return WaveformRecord(
            event=event,
            stream=stream,
        )

    def run(
        self,
        events: list[EventRecord],
    ) -> list[WaveformRecord]:
        """
        Load waveforms for all validated events.
        """

        records = []

        for event in events:
            record = self.load(event)
            records.append(record)

        return records