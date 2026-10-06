from dataclasses import dataclass

from src.modules.data_loader import EventRecord
from src.modules.waveform_loader import WaveformRecord


@dataclass
class GroupingResult:
    """
    waveforms:
        One WaveformRecord per DISTINCT waveform (the record of its
        primary event). Feed these to Preprocessor / Windowing.

    event_groups:
        primary evid -> every catalogue event on that waveform
        (sorted by arrival time; always includes the primary event).
    """

    waveforms: list[WaveformRecord]
    event_groups: dict[str, list[EventRecord]]

    @property
    def group_of(self) -> dict[str, str]:
        """any evid -> primary evid of its waveform."""
        return {
            e.evid: primary
            for primary, events in self.event_groups.items()
            for e in events
        }

    def events_of(self, primary_evids) -> list[EventRecord]:
        """All catalogue events living on the given primary waveforms."""
        return [
            e
            for p in primary_evids
            for e in self.event_groups[p]
        ]

    def evids_of(self, primary_evids) -> list[str]:
        return [e.evid for e in self.events_of(primary_evids)]