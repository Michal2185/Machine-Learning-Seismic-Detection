"""
Apollo-compatible preprocessing chain for InSight traces.

    raw 20 Hz  ->  resample_poly(53, 160) to 6.625 Hz  ->  causal Butterworth highpass 0.2 Hz (order 4)
               ->  [optional causal Butterworth lowpass 1.3 Hz (order 4)]

6.625 Hz = 53/8 and 20 Hz = 160/8, so the ratio is exact (no timing jitter). The anti-alias FIR of resample_poly is
linear phase and looks about 1.5 s ahead (resample_lookahead_s); this is not causal at the 1.5 s scale, which is far
below the 30-60 s timing resolution of the evaluation, and is stated in the paper. Highpass and lowpass are causal
(sosfilt with the steady-state initial condition of the first sample), identical to the Apollo apply_filter; the first ~2 minutes of every stream are filter start-up and
are masked as gaps by streams.py.
"""

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

FS_APOLLO = 6.625
HP_HZ = 0.2
LP_HZ = 1.3
ORDER = 4


def resample_ratio(fs_in: float) -> tuple:

    r = Fraction(53, 8) / Fraction(float(fs_in)).limit_denominator(10000)

    return r.numerator, r.denominator


def resample_lookahead_s(fs_in: float) -> float:
    """Half length of the default FIR (10 x max(up, down) taps at the upsampled rate), in seconds."""

    up, down = resample_ratio(fs_in)

    return 10.0 * max(up, down) / (fs_in * up)


def resample_to_apollo(x, fs_in: float) -> np.ndarray:

    up, down = resample_ratio(fs_in)

    return resample_poly(np.asarray(x, dtype=np.float64), up, down)


def _causal(sos, x) -> np.ndarray:
    """Same convention as src.modules.preprocessor.apply_filter: causal, initial state = steady state of the first sample
    (suppresses the start-up transient of a DC offset). Verified identical to apply_filter (max difference 0 after the
    start-up, tests.test_mars.apollo_apply_filter_agrees_if_repo_present)."""

    x = np.asarray(x, dtype=np.float64)
    y, _ = sosfilt(sos, x, zi=sosfilt_zi(sos) * x[0])

    return y


def highpass(x, fs: float = FS_APOLLO, hz: float = HP_HZ, order: int = ORDER) -> np.ndarray:
    return _causal(butter(order, hz, "highpass", fs=fs, output="sos"), x)


def lowpass(x, fs: float = FS_APOLLO, hz: float = LP_HZ, order: int = ORDER) -> np.ndarray:
    return _causal(butter(order, hz, "lowpass", fs=fs, output="sos"), x)


def preprocess(x, fs_in: float) -> np.ndarray:
    """Resample + highpass (the 'nolp' arm). float64; cast to float32 where the data enter a stream."""
    return highpass(resample_to_apollo(x, fs_in))


def arm_data(hp, arm: str) -> np.ndarray:
    """'nolp' = highpass only; 'lp1.3' = highpass + lowpass 1.3 Hz. float32 like the Apollo streams."""

    if arm == "nolp":
        return np.asarray(hp, dtype=np.float32)

    if arm.startswith("lp"):
        return lowpass(hp, FS_APOLLO, float(arm[2:])).astype(np.float32)

    raise ValueError(f"unknown arm {arm!r} (nolp | lp1.3)")