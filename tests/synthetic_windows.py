"""Synthetic streams, windows and labels shared by the training tests."""

from dataclasses import dataclass

import numpy as np

FS = 6.625
WIN = 398
STEP = 66


@dataclass
class Ev:
    evid: str


@dataclass
class Win:
    data: np.ndarray
    end_sample: int
    label: int
    event: Ev


@dataclass
class Stream:
    waveform_id: str
    data: np.ndarray
    sampling_rate: float = FS
    gaps: list = None

    def __post_init__(self):
        if self.gaps is None:
            self.gaps = []


def make_stream(wid, hours=3.0, sigma=1.0, onset_s=None, gain=8.0, dur_s=900.0, seed=0):
    """Gaussian noise, optional burst (event) starting at onset_s."""

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sigma, int(hours * 3600 * FS))

    if onset_s is not None:
        i0, i1 = int(onset_s * FS), int((onset_s + dur_s) * FS)
        decay = np.linspace(1.0, 0.3, i1 - i0)
        x[i0:i1] *= 1.0 + (gain - 1.0) * decay

    return Stream(wid, x.astype(np.float32))


def make_windows(stream, onset_s=None, step=STEP):
    """
    label 1  : window ends 30-60 s after the onset (>= 30 s of signal inside)
    label -1 : around the event (60 s before onset .. 20 min after), not positive
    label 0  : background
    """

    out = []

    for end in range(WIN, len(stream.data) + 1, step):
        t_end = end / FS
        lab = 0

        if onset_s is not None:
            if onset_s + 30 <= t_end <= onset_s + 60:
                lab = 1
            elif onset_s - 60 <= t_end <= onset_s + 1200:
                lab = -1

        out.append(Win(stream.data[end - WIN:end], end, lab, Ev(stream.waveform_id)))

    return out


def make_dataset(n_event=5, n_quiet=3, hours=3.0, seed=0):
    """Returns (streams, windows); event waveforms have sigma 1 and 8x bursts at 1.2 h."""

    streams, windows = [], []

    for i in range(n_event + n_quiet):
        onset = min(1.2 * 3600, 0.4 * hours * 3600) if i < n_event else None
        sigma = [1.0, 3.0, 0.2, 1.5, 0.7, 2.0, 1.0, 0.5][i % 8]
        s = make_stream(f"w{i:02d}", hours, sigma, onset, seed=seed + i)
        streams.append(s)
        windows += make_windows(s, onset)

    return streams, windows