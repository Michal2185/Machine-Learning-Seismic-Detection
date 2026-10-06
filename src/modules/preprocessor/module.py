import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from src.modules.waveform_loader import WaveformRecord

from .types import PreprocessConfig, ProcessedWaveformRecord


_NORMALIZATIONS = ("none", "fixed_scale")


def design_sos(
    sampling_rate: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int,
):
    """Butterworth second-order sections, or None if no filtering."""

    nyquist = sampling_rate / 2.0

    if low_hz is not None and not 0 < low_hz < nyquist:
        raise ValueError(f"low_hz={low_hz} must be in (0, {nyquist})")

    if high_hz is not None and not 0 < high_hz < nyquist:
        raise ValueError(f"high_hz={high_hz} must be in (0, {nyquist})")

    if low_hz is not None and high_hz is not None:
        if low_hz >= high_hz:
            raise ValueError("low_hz must be smaller than high_hz")
        return butter(
            order, [low_hz, high_hz],
            btype="bandpass", fs=sampling_rate, output="sos",
        )

    if low_hz is not None:
        return butter(
            order, low_hz,
            btype="highpass", fs=sampling_rate, output="sos",
        )

    if high_hz is not None:
        return butter(
            order, high_hz,
            btype="lowpass", fs=sampling_rate, output="sos",
        )

    return None


def apply_filter(
    data: np.ndarray,
    sampling_rate: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    """
    Causal filtering in float64. Output sample n depends only on input
    samples <= n. The initial state is set to the steady state of the
    first sample, which suppresses the start-up transient for DC offsets.
    """

    sos = design_sos(sampling_rate, low_hz, high_hz, order)

    x = np.asarray(data, dtype=np.float64)

    if sos is None:
        return x

    zi = sosfilt_zi(sos) * x[0]
    y, _ = sosfilt(sos, x, zi=zi)

    return y


def robust_std(x: np.ndarray) -> float:
    """1.4826 * MAD: robust to the event itself and to glitches."""
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def standardize_window(
    window: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-window standardisation (zero mean, unit std). Removes absolute
    amplitude, which helps cross-mission transfer but discards the
    strongest cue (amplitude). Use as an ablation at window level.
    """
    w = np.asarray(window, dtype=np.float64)
    return ((w - w.mean()) / (w.std() + eps)).astype(np.float32)


class Preprocessor:
    """
    Causal, deployable preparation of loaded seismic waveforms.

    Steps: causal Butterworth filter -> optional fixed-scale normalisation
    -> float32.
    """

    def __init__(self, config: PreprocessConfig | None = None):

        self.config = config or PreprocessConfig()

        if self.config.normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_NORMALIZATIONS}"
            )

        self.scale = self.config.scale

    # --------------------------------------------------------------

    def _filtered(self, record: WaveformRecord) -> tuple[np.ndarray, float]:

        if len(record.stream) != 1:
            raise ValueError(
                f"Expected exactly one trace, "
                f"got {len(record.stream)} "
                f"for event {record.event.evid}"
            )

        trace = record.stream[0]

        raw = np.asarray(trace.data, dtype=np.float64)

        if not np.all(np.isfinite(raw)):
            raise ValueError(
                f"Waveform contains NaN or infinite values "
                f"for event {record.event.evid}"
            )

        fs = float(trace.stats.sampling_rate)

        filtered = apply_filter(
            raw,
            fs,
            self.config.low_hz,
            self.config.high_hz,
            self.config.filter_order,
        )

        return filtered, fs

    def fit(self, records: list[WaveformRecord]) -> float:
        """
        Estimate the fixed scale from TRAINING records only
        (median over records of the robust std of the filtered data).
        """

        if self.config.normalization != "fixed_scale":
            raise ValueError(
                "fit() is only meaningful for normalization='fixed_scale'"
            )

        stds = [robust_std(self._filtered(r)[0]) for r in records]

        self.scale = float(np.median(stds))

        if not self.scale > 0:
            raise ValueError("Estimated scale is not positive")

        return self.scale

    def process(
        self,
        record: WaveformRecord,
    ) -> ProcessedWaveformRecord:

        filtered, fs = self._filtered(record)

        scale = 1.0

        if self.config.normalization == "fixed_scale":

            if self.scale is None:
                raise RuntimeError(
                    "fixed_scale needs a scale: call fit(train_records) "
                    "or set PreprocessConfig.scale"
                )

            scale = self.scale

        data = (filtered / scale).astype(np.float32)

        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"Non-finite values after preprocessing "
                f"for event {record.event.evid}"
            )

        trace = record.stream[0]

        return ProcessedWaveformRecord(
            event=record.event,
            data=data,
            sampling_rate=trace.stats.sampling_rate,
            start_time=trace.stats.starttime,
            scale=scale,
            low_hz=self.config.low_hz,
            high_hz=self.config.high_hz,
        )

    def run(
        self,
        records: list[WaveformRecord],
    ) -> list[ProcessedWaveformRecord]:

        processed = []

        for record in records:
            processed_record = self.process(record)
            processed.append(processed_record)

        return processed