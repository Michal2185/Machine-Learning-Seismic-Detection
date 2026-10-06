import os

from src.modules.data_loader import DataLoader
from src.modules.event_zones import EventZoneTable, ZoneEstimator
from src.modules.folds import FoldBuilder, FoldSet
from src.modules.labeling import Labeling
from src.modules.preprocessor import Preprocessor
from src.modules.validator import Validator
from src.modules.waveform_grouping import WaveformGrouper
from src.modules.waveform_loader import WaveformLoader
from src.modules.windowing import Windowing, WindowConfig

from .types import DatasetBundle, PipelineConfig


class DataPipeline:
    """
    Single entry point of the data branch:

        catalogue -> validation -> waveforms -> grouping of shared waveforms
        -> zone table (onset / coda / gaps, cached) -> preprocessing
        -> windows -> labels -> folds (cached)

        bundle = DataPipeline(PipelineConfig(...)).build()
        split  = bundle.fold_split(0)            # train / validation / test windows
        streams = bundle.eval_streams(split.test_events)   # for the evaluator

    The zone table and the fold assignment are written to
    <derived_dir>/zones and <derived_dir>/folds/folds.csv and reused on the
    next run (set PipelineConfig.rebuild=True to recompute them).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------

    def _zones(self, grouping):

        cfg = self.config
        path = os.path.join(cfg.derived_dir, "zones")

        events = [
            e
            for evs in grouping.event_groups.values()
            for e in evs
        ]

        if not cfg.rebuild and os.path.exists(
            os.path.join(path, "event_zones.csv")
        ):
            table = EventZoneTable.load(path, cfg.zones)

            if all(table.get(e.evid) is not None for e in events):
                return table

            print("Cached zone table is incomplete -> recomputing.")

        table = ZoneEstimator(cfg.zones).run(
            grouping.waveforms, grouping.event_groups
        )

        table.save(path)

        return table

    def _folds(self, grouping, labeling):

        cfg = self.config
        path = os.path.join(cfg.derived_dir, "folds", "folds.csv")

        if not cfg.rebuild and os.path.exists(path):

            folds = FoldSet.load(path)

            same = (
                set(folds.assignment) == set(grouping.event_groups)
                and folds.n_folds == cfg.n_folds
                and folds.random_seed == cfg.random_seed
            )

            if same:
                return folds

            print("Cached folds do not match the configuration -> rebuilding.")

        folds = FoldBuilder(cfg.n_folds, cfg.random_seed).run(
            grouping.event_groups, labeling
        )

        folds.save(path)

        return folds

    # ------------------------------------------------------------------

    def build(self) -> DatasetBundle:

        cfg = self.config

        if (
            cfg.preprocess.normalization == "fixed_scale"
            and cfg.preprocess.scale is None
        ):
            raise ValueError(
                "fixed_scale must be fitted on the training waveforms of "
                "each fold (Preprocessor.fit); give an explicit "
                "PreprocessConfig.scale or use normalization='none' here."
            )

        events = DataLoader(
            catalog_path=cfg.catalog_path,
            data_dir=cfg.data_dir,
        ).run()

        report = Validator(cfg.data_dir).run(events)

        waveforms = WaveformLoader().run(report.valid_events)

        grouping = WaveformGrouper().run(waveforms)

        zones = self._zones(grouping)

        processed = Preprocessor(cfg.preprocess).run(grouping.waveforms)

        windows = Windowing(
            WindowConfig(
                window_length=cfg.window_length,
                step_size=cfg.step_size,
            )
        ).run(processed)

        labeling = Labeling(
            cfg.labels,
            event_groups=grouping.event_groups,
            zones=zones,
        )

        labeled = labeling.run(windows)

        folds = self._folds(grouping, labeling)

        return DatasetBundle(
            config=cfg,
            grouping=grouping,
            zones=zones,
            labeling=labeling,
            processed=processed,
            windows=labeled,
            folds=folds,
        )