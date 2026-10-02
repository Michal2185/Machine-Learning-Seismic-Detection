from .types import LabeledWindowRecord
from src.modules.windowing import WindowRecord


class Labeling:
    """
    Assigns binary event labels to waveform windows.

    Label:
        1 = catalogue arrival falls inside the window
        0 = no catalogue arrival inside the window
    """

    def label(
        self,
        window: WindowRecord,
    ) -> LabeledWindowRecord:

        arrival_time = window.event.time_rel

        is_event = (
            window.start_time
            <= arrival_time
            < window.end_time
        )

        label = 1 if is_event else 0

        return LabeledWindowRecord(
            window=window,
            label=label,
        )

    def run(
        self,
        windows: list[WindowRecord],
    ) -> list[LabeledWindowRecord]:

        return [
            self.label(window)
            for window in windows
        ]