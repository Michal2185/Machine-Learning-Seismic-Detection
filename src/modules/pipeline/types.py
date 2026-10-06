from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.modules.dataset_builder import DatasetSplit
from src.modules.event_zones import EventZoneTable, ZoneConfig
from src.modules.folds import FoldSet
from src.modules.labeling import (
    IGNORE,
    POSITIVE,
    LabelConfig,
    Labeling,
)
from src.modules.preprocessor import PreprocessConfig
from src.modules.waveform_grouping import GroupingResult


@dataclass
class PipelineConfig:
    catalog_path: str
    data_dir: str
    derived_dir: str                  # zones/ and folds/ are cached here

    window_length: float = 60.0
    step_size: float = 10.0

    # causal highpass 0.2 Hz, no scaling: normalisation variants are
    # applied by the model code (fixed scale must be fitted on the
    # TRAINING waveforms of each fold, so it cannot be set here)
    preprocess: PreprocessConfig = field(
        default_factory=lambda: PreprocessConfig(low_hz=0.2)
    )
    labels: LabelConfig = field(default_factory=LabelConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)

    n_folds: int = 5
    random_seed: int = 42

    rebuild: bool = False             # ignore cached zones / folds


@dataclass
class EventTarget:
    """One catalogue event as the evaluator sees it (times in seconds)."""

    evid: str
    event_type: str
    arrival_s: float
    detect: tuple              # alarm inside this interval detects the event
    ignore: tuple              # onset region + coda: no false alarms counted
    onset_s: float | None      # visible onset (absolute), None if unusable
    has_positive: bool         # False -> weak event (no usable visible onset)
    peak_ratio: float | None   # peak envelope / day background


@dataclass
class EvalStream:
    """
    One continuous waveform plus everything needed to score a detector
    on it. Alarm time convention for the evaluator: the END time of the
    window that triggered the alarm.
    """

    waveform_id: str
    data: np.ndarray           # preprocessed, float32, continuous
    sampling_rate: float
    start_time: object
    events: list               # [EventTarget]
    gaps: list                 # [(start_s, end_s)] interpolated stretches


@dataclass
class DatasetBundle:
    config: PipelineConfig
    grouping: GroupingResult
    zones: EventZoneTable
    labeling: Labeling
    processed: list            # ProcessedWaveformRecord per distinct waveform
    windows: list              # LabeledWindowRecord, all waveforms
    folds: FoldSet

    def __post_init__(self):
        self._processed_by_id = {
            r.event.evid: r for r in self.processed
        }

    # ------------------------------------------------------------------

    def fold_split(self, fold_index: int) -> DatasetSplit:
        """Windows of one fold in the DatasetSplit format."""
        return self.folds.split(self.windows, fold_index)

    def target(self, event) -> EventTarget:

        z = self.zones.get(event.evid)
        anchor = self.labeling.anchor(event)

        onset = (
            float(anchor)
            if anchor is not None and np.isfinite(anchor)
            else None
        )

        return EventTarget(
            evid=event.evid,
            event_type=event.event_type,
            arrival_s=float(event.time_rel),
            detect=self.labeling.detect_interval(event),
            ignore=self.labeling.ignore_interval(event),
            onset_s=onset,
            has_positive=self.labeling.positive_rule(event) is not None,
            peak_ratio=None if z is None else float(z.peak_ratio),
        )

    def eval_stream(self, waveform_id: str) -> EvalStream:

        rec = self._processed_by_id[waveform_id]

        return EvalStream(
            waveform_id=waveform_id,
            data=rec.data,
            sampling_rate=float(rec.sampling_rate),
            start_time=rec.start_time,
            events=[
                self.target(e)
                for e in self.grouping.event_groups[waveform_id]
            ],
            gaps=list(self.zones.gaps.get(waveform_id, [])),
        )

    def eval_streams(self, waveform_ids) -> list:
        return [self.eval_stream(w) for w in waveform_ids]

    # ------------------------------------------------------------------

    def summary(self) -> None:

        c = Counter(w.label for w in self.windows)
        total = len(self.windows)

        events = [
            e
            for evs in self.grouping.event_groups.values()
            for e in evs
        ]

        n_pos_events = sum(
            self.labeling.positive_rule(e) is not None for e in events
        )

        print("=" * 60)
        print("DATASET BUNDLE")
        print("=" * 60)
        print(f"Catalogue events:     {len(events)}")
        print(f"Distinct waveforms:   {len(self.processed)}")
        print(f"Events with positives:{n_pos_events:4d} "
              f"(weak events: {len(events) - n_pos_events})")
        print(f"Windows:              {total} "
              f"({self.config.window_length:.0f} s, "
              f"step {self.config.step_size:.0f} s)")
        print(f"  positive (1):       {c[POSITIVE]}")
        print(f"  ignore (-1):        {c[IGNORE]} "
              f"({100 * c[IGNORE] / total:.2f}%, "
              f"of which gaps: {self.labeling.gap_ignored})")
        print(f"  negative (0):       {c[0]}")
        print(f"Folds: {self.folds.n_folds} (seed {self.folds.random_seed})")
        print(f"Preprocessing: {self.config.preprocess}")