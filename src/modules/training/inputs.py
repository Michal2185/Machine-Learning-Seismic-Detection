import numpy as np

from src.modules.dnn_features import FeatureScaler, extract_features

LAYOUTS = ("NT", "NCT", "NTC")


class RawWindowInput:
    """Normalised raw window as network input (1D CNN / GRU / CfC variant W)."""

    def __init__(self, normalizer, layout: str = "NCT"):

        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")

        self.normalizer = normalizer
        self.layout = layout

    def fit(self, train_windows) -> "RawWindowInput":
        self.normalizer.fit(train_windows)         # N0 learns its constant; others no-op
        return self

    def build(self, windows) -> np.ndarray:

        x = self.normalizer.windows(windows)

        if self.layout == "NCT":
            return x[:, None, :]

        if self.layout == "NTC":
            return x[:, :, None]

        return x

    def ops_per_window(self, n_samples: int = 398) -> int:
        return n_samples                            # one multiply per sample (scalar scale)


class FeatureInput:
    """6 features of the normalised window, log + standardised (DNN / RFC)."""

    def __init__(self, normalizer, fs: float, scaler: FeatureScaler | None = None,
                 scale: bool = True, fit_stride: int = 6):

        self.normalizer = normalizer
        self.fs = fs
        self.scaler = (scaler or FeatureScaler()) if scale else None
        self.fit_stride = fit_stride

    def fit(self, train_windows) -> "FeatureInput":

        self.normalizer.fit(train_windows)

        if self.scaler is not None:
            sub = list(train_windows)[:: self.fit_stride]
            f = extract_features(self.normalizer.windows(sub), self.fs)
            self.scaler.fit(f)

        return self

    def build(self, windows) -> np.ndarray:

        f = extract_features(self.normalizer.windows(windows), self.fs)

        return self.scaler.transform(f) if self.scaler is not None else f

    def ops_per_window(self, n_samples: int = 398) -> int:
        return feature_ops(n_samples)


def feature_ops(n: int) -> int:
    """
    Rough operation count of the 6 features of one window of n samples:
    normalisation n, mean n, energy 2n, ptp 2n, peak n, zero crossings 2n,
    Hann taper n, real FFT about 2.5 n log2 n, power spectrum n. An
    estimate for comparing with STA/LTA, not a measurement.
    """

    return int(10 * n + 2.5 * n * np.log2(n))