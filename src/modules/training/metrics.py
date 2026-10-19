import numpy as np


def average_precision(labels, scores) -> float:
    """
    Average precision with tied scores grouped (same definition as
    sklearn.metrics.average_precision_score). labels are 0/1; NaN if there
    is no positive.
    """

    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)

    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")

    n_pos = int(y.sum())

    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]

    tps = np.cumsum(y)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]

    precision = tps[last] / (last + 1)
    recall = tps[last] / n_pos

    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))