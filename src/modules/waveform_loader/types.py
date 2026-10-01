from dataclasses import dataclass

from obspy import Stream

from src.modules.data_loader import EventRecord


@dataclass
class WaveformRecord:
    """
    An event together with its loaded MiniSEED waveform.
    """

    event: EventRecord
    stream: Stream