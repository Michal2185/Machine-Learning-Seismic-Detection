import numpy as np

from .module import FEATURE_NAMES

LOG_FEATURES = ("energy", "rms", "ptp")     # heavy tailed, span orders of magnitude


class FeatureScaler:
    """
    Log transform of the amplitude features + standardisation, fitted on
    TRAINING features only (the feature models should call fit once per fold).

    energy, rms, ptp -> log(max(x, floor)); every feature -> (x - mean) / std
    (std 0 -> 1); output clipped to +-clip (flat windows give an extreme log).
    """

    def __init__(self, log_features=LOG_FEATURES, floor: float = 1e-6, clip: float = 10.0):

        self.log_idx = [FEATURE_NAMES.index(n) for n in log_features]
        self.floor = floor
        self.clip = clip
        self.mean = None
        self.std = None

    def _log(self, f: np.ndarray) -> np.ndarray:

        g = np.array(f, dtype=np.float64, copy=True)

        if self.log_idx:
            g[:, self.log_idx] = np.log(np.maximum(g[:, self.log_idx], self.floor))

        return g

    def fit(self, features: np.ndarray) -> "FeatureScaler":

        f = np.asarray(features, dtype=np.float64)

        if f.ndim != 2 or f.shape[0] < 2:
            raise ValueError("need (N, 6) features with N >= 2 to fit the scaler")

        g = self._log(f)
        self.mean = g.mean(axis=0)
        std = g.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:

        if self.mean is None:
            raise RuntimeError("FeatureScaler is not fitted")

        g = (self._log(np.asarray(features, dtype=np.float64)) - self.mean) / self.std

        return np.clip(g, -self.clip, self.clip).astype(np.float32)