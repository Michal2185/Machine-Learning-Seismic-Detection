from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.modules.data_loader import EventRecord


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Defaults reproduce the previous behaviour (cast to float32 only),
    so existing tests keep passing until a config is chosen explicitly.

    low_hz / high_hz:
        Causal Butterworth filter (sosfilt, NOT filtfilt: zero-phase
        filtering uses future samples and is impossible onboard).
          both set  -> bandpass
          low only  -> highpass
          high only -> lowpass
          none      -> no filtering
        Choose the edges from the spectral diagnostic on training events.

    normalization:
        "none"         -> keep physical amplitude
        "fixed_scale"  -> divide by ONE constant (robust std of filtered
                          training data). Preserves relative amplitude
                          between events/windows and is deployable onboard.
                          The constant is obtained with Preprocessor.fit()
                          on TRAINING records only, or set via `scale`.
        Per-window standardisation is applied later at window level
        (see standardize_window in module.py).
    """

    low_hz: float | None = None
    high_hz: float | None = None
    filter_order: int = 4
    normalization: str = "none"
    scale: float | None = None


@dataclass
class ProcessedWaveformRecord:
    event: EventRecord
    data: np.ndarray
    sampling_rate: float
    start_time: datetime

    # traceability (defaults keep old constructor calls valid)
    scale: float = 1.0
    low_hz: float | None = None
    high_hz: float | None = None