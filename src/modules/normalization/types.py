from dataclasses import dataclass

import numpy as np

VARIANTS = ("N0", "N1", "N2", "N3")
FIXED_SCALE_STATS = ("median_of_waveforms", "pooled")


@dataclass
class NormConfig:
    """
    variant
      N0  fixed scale (one constant, fitted on the TRAINING waveforms of a fold)
      N1  per-window zero mean / unit std
      N2  causal adaptive scale (trailing quantile of block levels)
      N3  N2 followed by asinh compression
    """

    variant: str = "N2"

    # N0
    fixed_scale_stat: str = "median_of_waveforms"   # or "pooled"

    # N1 / all: floor that keeps divisions finite on flat input
    min_scale: float = 1e-30         # absolute floor; data are ~1e-10, so it must be tiny

    # N2 / N3: causal trailing-quantile background estimate
    block_s: float = 10.0            # one level estimate per block
    quantile: float = 0.65           # quantile of the trailing block levels
    window_s: float = 21600.0        # trailing window (6 h)
    min_valid_blocks: int = 30       # fewer valid blocks in the window -> no estimate yet
    max_gap_fraction: float = 0.5    # blocks with more interpolated samples are skipped
    elevated_ratio: float = 2.0      # diagnostic only: level > ratio x estimate is "elevated"

    # N3
    asinh_knee: float = 1.0          # asinh(z / knee), z in background-sigma units

    def __post_init__(self):

        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")

        if self.fixed_scale_stat not in FIXED_SCALE_STATS:
            raise ValueError(f"fixed_scale_stat must be one of {FIXED_SCALE_STATS}")

        if self.min_scale <= 0 or self.block_s <= 0 or self.window_s <= 0:
            raise ValueError("min_scale, block_s and window_s must be > 0")

        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")

        if self.min_valid_blocks < 1:
            raise ValueError("min_valid_blocks must be >= 1")

        if self.elevated_ratio <= 1.0:
            raise ValueError("elevated_ratio must be > 1")

        if self.asinh_knee <= 0:
            raise ValueError("asinh_knee must be > 0")

        if not 0.0 < self.max_gap_fraction <= 1.0:
            raise ValueError("max_gap_fraction must be in (0, 1]")


@dataclass
class ScaleTrack:
    """
    Causal background estimate of one continuous waveform.

    scale[i] is the estimate held BEFORE block i was seen (blocks 0..i-1
    only), so it can be applied to the samples of block i. Blocks before
    the first estimate (start-up only); blocks after a stretch without
    enough valid data carry the last estimate.
    """

    waveform_id: str
    block_len: int             # samples per block
    n_samples: int
    scale: np.ndarray          # (n_blocks,) float64
    level: np.ndarray          # (n_blocks,) robust std of each block (NaN if skipped)
    elevated: np.ndarray       # (n_blocks,) bool, level > elevated_ratio x scale (diagnostic)
    skipped: np.ndarray        # (n_blocks,) bool, excluded from the estimate (gap / flat / invalid)

    def block_of(self, sample: int) -> int:
        sample = min(max(int(sample), 0), self.n_samples - 1)
        return sample // self.block_len

    def at_end(self, end_sample: int) -> float:
        """Scale a window ending at end_sample (exclusive) is normalised with."""
        return float(self.scale[self.block_of(end_sample - 1)])

    def per_sample(self) -> np.ndarray:
        return np.repeat(self.scale, self.block_len)[: self.n_samples]