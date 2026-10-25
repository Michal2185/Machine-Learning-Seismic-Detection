import numpy as np

from src.modules.detectors.types import Detector
from src.modules.dnn_features import FEATURE_NAMES, extract_features
from src.modules.training import ResourceReport

KINDS = ("rms", "ptp")


class AmplitudeDetector(Detector):
    """
    Control detector without learning: the score is the RMS or the peak-to-peak
    amplitude of the (optionally normalised) window. The evaluator chooses its
    threshold on validation like for any other detector.

    With N0 (or no normaliser) the score is a fixed multiple of the raw amplitude;
    with N2 it is the amplitude relative to the causal background estimate.
    Anything a learned model claims on top of this has to be measured against it.
    """

    def __init__(self, name: str, kind: str = "rms", normalizer=None, fs: float = 6.625):

        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")

        self.name = name
        self.kind = kind
        self.normalizer = normalizer
        self.fs = fs
        self._idx = FEATURE_NAMES.index(kind)

    def fit(self, train_windows) -> "AmplitudeDetector":

        if self.normalizer is not None:
            self.normalizer.fit(list(train_windows))          # N0 learns its constant; others no-op

        return self

    def score(self, windows, streams_by_id=None) -> np.ndarray:

        windows = list(windows)
        out = []

        for i in range(0, len(windows), 8192):

            chunk = windows[i:i + 8192]

            if self.normalizer is not None:
                x = self.normalizer.windows(chunk)
            else:
                x = np.stack([np.asarray(w.data, dtype=np.float64) for w in chunk])

            out.append(extract_features(x, self.fs)[:, self._idx].astype(np.float64))

        return np.concatenate(out) if out else np.zeros(0)

    def resource_report(self, example_window, n_samples: int | None = None) -> ResourceReport:

        n = n_samples or len(example_window.data)

        return ResourceReport(
            params=1, bytes_fp32=4, bytes_int8=4, macs_per_window=0, state_floats=0,
            input_ops=3 * n,
            notes=["one threshold, no weights; input ops: scaling n + amplitude 2n (estimate)"],
        )