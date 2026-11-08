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

# ----------------------------------------------------------------------
# Variant E: sequence of sub-window envelope features (GRU / CfC)
# ----------------------------------------------------------------------

ENVELOPE_FEATURES = ("log_energy", "rms", "ptp", "zcr")
N_ENVELOPE = len(ENVELOPE_FEATURES)


def envelope_features(x: np.ndarray, n_sub: int, fs: float, eps: float = 1e-6,
                      log_amp: bool = False) -> np.ndarray:
    """
    Per sub-window features of NORMALISED windows x (N, T) -> (N, n_sub, 4) float32.

    The window is cut into n_sub sub-windows of L = T // n_sub samples, aligned to the END of the window
    (the oldest T - n_sub * L samples are dropped), so the last step always ends at the decision time.
    Per sub-window (mean removed first, as in extract_features):

      log_energy  log(sum of squares + eps)
      rms         sqrt(energy / L)
      ptp         peak-to-peak amplitude
      zcr         zero crossings per second (crossings / (L / fs))

    log_amp=True replaces rms and ptp by log(. + eps) (ablation; the default follows the plan).
    Fully vectorised: no Python loop over windows or sub-windows.
    """

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("x must be (N, T)")

    n_win, t = x.shape
    length = t // n_sub

    if n_sub < 1 or length < 2:
        raise ValueError(f"window of {t} samples cannot be cut into {n_sub} sub-windows of >= 2 samples")

    if fs <= 0:
        raise ValueError("fs must be > 0")

    if not np.isfinite(x).all():
        raise ValueError("windows contain non-finite samples")

    s = x[:, t - n_sub * length:].reshape(n_win, n_sub, length)
    s = s - s.mean(axis=2, keepdims=True)

    energy = (s ** 2).sum(axis=2)
    rms = np.sqrt(energy / length)
    ptp = s.max(axis=2) - s.min(axis=2)

    sign = s >= 0
    zcr = (sign[:, :, 1:] != sign[:, :, :-1]).sum(axis=2) / (length / fs)

    if log_amp:
        rms, ptp = np.log(rms + eps), np.log(ptp + eps)

    out = np.stack([np.log(energy + eps), rms, ptp, zcr], axis=2)

    return out.astype(np.float32)


class EnvelopeInput:
    """
    Sequence of sub-window envelope features of the normalised window, input (B, n_sub, 4) for the
    GRU / CfC variant E (12 steps of 5 s). Features are standardised per feature with the mean / std of
    the TRAINING windows of the fold (fit_stride keeps the fit cheap), like FeatureInput.

    step_s is the physical time between two steps (L / fs, about 5 s): the time step of the CfC.
    """

    def __init__(self, normalizer, fs: float, n_sub: int = 12, scale: bool = True,
                 fit_stride: int = 6, eps: float = 1e-6, log_amp: bool = False):

        self.normalizer = normalizer
        self.fs = float(fs)
        self.n_sub = n_sub
        self.scale = scale
        self.fit_stride = fit_stride
        self.eps = eps
        self.log_amp = log_amp
        self.mean_ = None
        self.std_ = None
        self._length = None

    @property
    def step_s(self) -> float:
        """Seconds between two steps (needs one build / fit call to know the window length)."""

        if self._length is None:
            raise RuntimeError("window length unknown: call fit() or build() first")

        return self._length / self.fs

    def _raw(self, windows) -> np.ndarray:

        x = self.normalizer.windows(windows)
        self._length = x.shape[1] // self.n_sub

        return envelope_features(x, self.n_sub, self.fs, self.eps, self.log_amp)

    def fit(self, train_windows) -> "EnvelopeInput":

        self.normalizer.fit(train_windows)

        sub = list(train_windows)[:: self.fit_stride]
        f = self._raw(sub)                                  # (N, n_sub, 4)

        if self.scale:
            flat = f.reshape(-1, f.shape[-1]).astype(np.float64)
            self.mean_ = flat.mean(axis=0)
            std = flat.std(axis=0)
            self.std_ = np.where(std > 1e-8, std, 1.0)

        return self

    def build(self, windows) -> np.ndarray:

        f = self._raw(windows)

        if not self.scale:
            return f

        if self.mean_ is None:
            raise RuntimeError("EnvelopeInput is not fitted")

        return ((f - self.mean_) / self.std_).astype(np.float32)

    def ops_per_window(self, n_samples: int = 398) -> int:
        """
        Recomputing the window from scratch: normalisation n, and per sub-window mean n, energy 2n,
        ptp 2n, zero crossings 2n over the n_sub * L used samples, plus about 6 operations per step
        (sqrt, log, standardisation). An estimate, not a measurement.
        """

        used = (n_samples // self.n_sub) * self.n_sub

        return int(n_samples + 7 * used + 6 * self.n_sub)

    def ops_streaming_per_10s(self, new_steps: int = 2) -> int:
        """Maintained in a streaming way: only the newest steps are computed (2 per 10 s at 5 s steps)."""

        length = self._length or 33

        return int(new_steps * (8 * length + 6))