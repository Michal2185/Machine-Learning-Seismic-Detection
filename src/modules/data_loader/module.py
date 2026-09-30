from asyncio import events
from pathlib import Path

import pandas as pd

from .types import EventRecord

class DataLoader:
    """
    Loads seismic event metadata from a catalogue and associates
    each catalogue entry with its corresponding MiniSEED file.
    """

    def __init__(self, catalog_path: str | Path, data_dir: str | Path):
        self.catalog_path = Path(catalog_path)
        self.data_dir = Path(data_dir)

    def run(self) -> list[EventRecord]:
        """
        Load the catalogue and return a list of EventRecord objects.
        """

        catalog = pd.read_csv(self.catalog_path)

        print(f"Catalogue shape: {catalog.shape}")
        print(f"Number of rows: {len(catalog)}")
        print()
        print(catalog.head())
        print()
        print(catalog.tail())

        events = []

        for _, row in catalog.iterrows():
            filename = str(row["filename"])

            waveform_path = self.data_dir / f"{filename}.mseed"

            event = EventRecord(
            filename=filename,
            waveform_path=waveform_path,
            time_abs=pd.to_datetime(
                row["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]
            ).to_pydatetime(),
            time_rel=float(row["time_rel(sec)"]),
            evid=str(row["evid"]),
            event_type=str(row["mq_type"]),
        )

            events.append(event)

        return events