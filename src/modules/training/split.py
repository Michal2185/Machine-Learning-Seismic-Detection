import numpy as np

from .types import POSITIVE


def group_by_waveform(windows) -> dict:

    groups = {}

    for w in windows:
        groups.setdefault(w.event.evid, []).append(w)

    return groups


def split_by_waveform(windows, val_fraction: float, seed: int):
    """
    Hold out whole waveforms (never single windows: neighbouring windows
    overlap by 50 s) of a TRAINING fold for early stopping.

    Waveforms with positives and without are split separately so both
    sides get positives when possible; at least one waveform with
    positives always stays in training.

    Returns (train_windows, val_windows, val_waveform_ids).
    """

    windows = list(windows)

    if val_fraction <= 0.0:
        return windows, [], []

    groups = group_by_waveform(windows)
    ids = sorted(groups)

    if len(ids) < 2:
        return windows, [], []

    pos_ids = [i for i in ids if any(w.label == POSITIVE for w in groups[i])]
    neg_ids = [i for i in ids if i not in set(pos_ids)]

    n_val = min(max(1, int(round(val_fraction * len(ids)))), len(ids) - 1)

    n_val_pos = int(round(n_val * len(pos_ids) / len(ids)))
    n_val_pos = min(n_val_pos, max(0, len(pos_ids) - 1))

    if len(pos_ids) >= 2 and n_val >= 2:
        n_val_pos = max(1, n_val_pos)

    n_val_neg = min(n_val - n_val_pos, len(neg_ids))
    n_val_pos = min(n_val - n_val_neg, max(0, len(pos_ids) - 1))

    rng = np.random.default_rng([seed, 9137])

    val_ids = set()

    if n_val_pos:
        val_ids |= set(rng.choice(pos_ids, size=n_val_pos, replace=False).tolist())

    if n_val_neg:
        val_ids |= set(rng.choice(neg_ids, size=n_val_neg, replace=False).tolist())

    train = [w for w in windows if w.event.evid not in val_ids]
    val = [w for w in windows if w.event.evid in val_ids]

    return train, val, sorted(val_ids)