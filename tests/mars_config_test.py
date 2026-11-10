import inspect
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter, filtfilt
from src.modules.preprocessor import apply_filter

print(inspect.getsource(apply_filter))

fs = 6.625
x = np.random.default_rng(1).normal(size=20000)
a = apply_filter(x, fs, 0.2, None, 4)
sos = butter(4, 0.2, "highpass", fs=fs, output="sos")
b_, a_ = butter(4, 0.2, "highpass", fs=fs)
cands = {"sosfilt (mine)": sosfilt(sos, x), "sosfiltfilt": sosfiltfilt(sos, x),
         "lfilter": lfilter(b_, a_, x), "filtfilt": filtfilt(b_, a_, x)}
for k, v in cands.items():
    print(f"{k:16s} all {np.max(np.abs(a - v)) / np.max(np.abs(a)):.2e}   after 2000 samples {np.max(np.abs(a[2000:] - v[2000:])) / np.max(np.abs(a)):.2e}")