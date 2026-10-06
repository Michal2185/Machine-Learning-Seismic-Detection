import hashlib

import numpy as np

from src.modules.waveform_loader import WaveformRecord

from .types import GroupingResult


class WaveformGrouper:
    """
    Detects catalogue events that share the SAME waveform (e.g. two
    events on the same day, each with its own copy of the daily file)
    and keeps a single record per distinct waveform.

    Why: duplicated waveforms (a) leak between train/validation/test,
    (b) are labeled against only one of their events, and (c) are
    double counted when fitting the normalisation scale.
    """

    @staticmethod
    def fingerprint(record: WaveformRecord) -> str:

        if len(record.stream) != 1:
            raise ValueError(
                f"Expected exactly one trace, "
                f"got {len(record.stream)} "
                f"for event {record.event.evid}"
            )

        trace = record.stream[0]

        h = hashlib.md5()
        h.update(np.ascontiguousarray(trace.data).tobytes())
        h.update(str(trace.stats.starttime).encode())
        h.update(str(trace.stats.sampling_rate).encode())

        return h.hexdigest()

    def run(self, records: list[WaveformRecord]) -> GroupingResult:

        primary_of_key = {}
        waveforms = []
        groups = {}

        for record in records:

            key = self.fingerprint(record)

            if key not in primary_of_key:
                primary_of_key[key] = record
                waveforms.append(record)
                groups[record.event.evid] = [record.event]
                continue

            primary = primary_of_key[key]

            # guard against hash collisions
            if not np.array_equal(
                primary.stream[0].data,
                record.stream[0].data,
            ):
                raise RuntimeError(
                    f"Fingerprint collision between "
                    f"{primary.event.evid} and {record.event.evid}"
                )

            groups[primary.event.evid].append(record.event)

        for events in groups.values():
            events.sort(key=lambda e: e.time_rel)

        return GroupingResult(
            waveforms=waveforms,
            event_groups=groups,
        )