import numpy as np

FEATURE_NAMES = ("energy", "rms", "ptp", "crest", "dom_freq", "zcr")
N_FEATURES = len(FEATURE_NAMES)


def extract_features(windows, fs: float) -> np.ndarray:
    """
    The six window features of the thesis (section 5.2.2), computed on
    the NORMALISED window (the caller normalises first).

      energy    sum of squares
      rms       sqrt(energy / n)
      ptp       peak-to-peak amplitude
      crest     max|x| / rms
      dom_freq  frequency [Hz] of the largest bin of the Hann-tapered
                power spectrum, DC excluded
      zcr       zero crossings per second

    The window mean is removed first (all features describe the
    fluctuation). A flat window gives all zeros. windows: (T,) or (N, T);
    returns (6,) or (N, 6) float32.
    """

    x = np.asarray(windows, dtype=np.float64)
    single = x.ndim == 1

    if single:
        x = x[None, :]

    if x.ndim != 2 or x.shape[1] < 4:
        raise ValueError("windows must be (T,) or (N, T) with T >= 4")

    if fs <= 0:
        raise ValueError("fs must be > 0")

    if not np.isfinite(x).all():
        raise ValueError("windows contain non-finite samples")

    n_win, n = x.shape

    amax = np.abs(x).max(axis=1)
    flat = (x.max(axis=1) - x.min(axis=1)) <= 1e-12 * amax
    flat |= amax == 0

    x = x - x.mean(axis=1, keepdims=True)

    energy = (x ** 2).sum(axis=1)
    rms = np.sqrt(energy / n)
    ptp = x.max(axis=1) - x.min(axis=1)
    peak = np.abs(x).max(axis=1)

    crest = np.zeros(n_win)
    ok = ~flat & (rms > 0)
    crest[ok] = peak[ok] / rms[ok]

    power = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    power[:, 0] = 0.0
    dom = power.argmax(axis=1) * fs / n

    sign = x >= 0
    crossings = (sign[:, 1:] != sign[:, :-1]).sum(axis=1)
    zcr = crossings / (n / fs)

    out = np.stack([energy, rms, ptp, crest, dom, zcr], axis=1)
    out[flat] = 0.0
    out = out.astype(np.float32)

    return out[0] if single else out