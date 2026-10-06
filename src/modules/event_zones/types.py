import csv
import math
import os
from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class ZoneConfig:
    """
    Parameters of the envelope analysis (see tests/diagnostic_onset_coda.py
    for the definitions). All times in seconds.
    """

    analysis_low_hz: float = 0.2     # causal highpass used for the analysis
    filter_order: int = 4
    env_sec: float = 20.0            # RMS envelope length
    k_on: float = 3.0                # onset: envelope > k_on * local background
    k_end: float = 2.0               # coda end: envelope < k_end * day background
    dropout_rel: float = 0.05        # envelope < 5% of background = no data
    onset_search: tuple = (-900.0, 300.0)
    onset_min_dur_s: float = 30.0
    local_ref: tuple = (-2400.0, -600.0)
    peak_search_s: float = 10800.0
    coda_min_below_s: float = 600.0
    event_zone: tuple = (-600.0, 21600.0)   # excluded from the day background
    gap_min_run_s: float = 10.0      # interpolated runs recorded as gaps
    long_gap_s: float = 600.0        # gap length that matters for a coda


@dataclass
class EventZone:
    evid: str
    event_type: str
    waveform_id: str                 # primary evid of the waveform
    arrival_s: float                 # catalogue time (s from file start)

    onset_offset_s: float            # visible onset - catalogue (nan if none)
    onset_status: str                # ok | none | pre_active
    peak_delay_s: float
    peak_ratio: float                # peak envelope / day background
    background: float                # day background level (raw units)
    local_background_ratio: float

    coda_end_s: float                # relative to the catalogue arrival
    coda_censored: bool              # True -> coda_end_s is a lower bound
    censor_reason: str               # "", file_end, next_event
    gap_in_coda: bool                # a long interpolated gap lies in the coda
    tau_s: float                     # exponential decay constant (nan if n/a)


_ZONE_FIELDS = [f.name for f in fields(EventZone)]


@dataclass
class EventZoneTable:
    zones: dict                      # evid -> EventZone
    gaps: dict                       # waveform_id -> [(start_s, end_s), ...]
    config: ZoneConfig = field(default_factory=ZoneConfig)

    def get(self, evid):
        return self.zones.get(evid)

    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:

        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "event_zones.csv"),
                  "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_ZONE_FIELDS)
            for z in self.zones.values():
                w.writerow([getattr(z, name) for name in _ZONE_FIELDS])

        with open(os.path.join(directory, "waveform_gaps.csv"),
                  "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["waveform_id", "start_s", "end_s"])
            for wid, runs in self.gaps.items():
                for a, b in runs:
                    w.writerow([wid, a, b])

    @classmethod
    def load(cls, directory: str, config: ZoneConfig | None = None):

        def as_bool(v):
            return str(v) == "True"

        zones = {}

        with open(os.path.join(directory, "event_zones.csv"),
                  newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                zones[row["evid"]] = EventZone(
                    evid=row["evid"],
                    event_type=row["event_type"],
                    waveform_id=row["waveform_id"],
                    arrival_s=float(row["arrival_s"]),
                    onset_offset_s=float(row["onset_offset_s"]),
                    onset_status=row["onset_status"],
                    peak_delay_s=float(row["peak_delay_s"]),
                    peak_ratio=float(row["peak_ratio"]),
                    background=float(row["background"]),
                    local_background_ratio=float(row["local_background_ratio"]),
                    coda_end_s=float(row["coda_end_s"]),
                    coda_censored=as_bool(row["coda_censored"]),
                    censor_reason=row["censor_reason"],
                    gap_in_coda=as_bool(row["gap_in_coda"]),
                    tau_s=float(row["tau_s"]),
                )

        gaps = {}

        with open(os.path.join(directory, "waveform_gaps.csv"),
                  newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                gaps.setdefault(row["waveform_id"], []).append(
                    (float(row["start_s"]), float(row["end_s"]))
                )

        return cls(zones=zones, gaps=gaps, config=config or ZoneConfig())

    # ------------------------------------------------------------------

    def n_nan(self) -> int:
        return sum(
            int(math.isnan(z.peak_ratio) or math.isnan(z.coda_end_s))
            for z in self.zones.values()
        )