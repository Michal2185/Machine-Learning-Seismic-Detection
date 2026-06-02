import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read

path = "F:/Git/Machine-Learning-Seismic-Detection/data/lunar/"

#catalog loading
df = pd.read_csv(path +"catalogs/apollo12_catalog_GradeA_final.csv")
df.columns = df.columns.str.strip()

#finding first occurrence of each event type
try:
    deep_idx = df[df['mq_type'].str.contains('deep', case=False)].index[0]
    shallow_idx = df[df['mq_type'].str.contains('shallow', case=False)].index[0]
    impact_idx = df[df['mq_type'].str.contains('impact', case=False)].index[0]
except IndexError:
    print("Could not find one of each event type in your catalog. Check your 'mq_type' labels!")
    # Fallback to manual indices if string matching fails
    deep_idx, shallow_idx, impact_idx = 0, 1, 2 

targets = [
    {"title": "Deep Moonquake", "idx": deep_idx, "color": "blue"},
    {"title": "Shallow Moonquake", "idx": shallow_idx, "color": "orange"},
    {"title": "Meteoroid Impact", "idx": impact_idx, "color": "purple"}
]

# Set up the subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(targets):
    row = df.iloc[target["idx"]]
    filename = path + "data/S12_GradeA/" + row['filename'] + ".mseed"
    
    # Load seismic file
    st = read(filename)
    trace = st[0]

    trace.filter("bandstop", freqmin=0.81, freqmax=0.83) #removing 0.82 Hz noise from the spectrum caused by the lunar module's instrument
    
    signal = trace.data
    sampling_rate = trace.stats.sampling_rate
    n = len(signal)
    
    # Compute FFT
    fft_values = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_magnitude = np.abs(fft_values)
    
    # ========================================================
    # NUMERICAL CALCULATION OF DOMINANT FREQUENCY
    # ========================================================
    # Find the index of the maximum amplitude
    dominant_idx = np.argmax(fft_magnitude)
    
    # Get the corresponding frequency and amplitude value
    dominant_freq = fft_freqs[dominant_idx]
    peak_amplitude = fft_magnitude[dominant_idx]
    
    # Print the results to the console
    print(f"{target['title']:<18} | Index: {target['idx']} | Dominant Freq: {dominant_freq:.3f} Hz (Amp: {peak_amplitude:.1f})")
    # ========================================================
    
    # Plotting
    ax = axes[i]
    ax.plot(fft_freqs, fft_magnitude, color=target["color"], linewidth=0.8, label="Spectrum")
    
    # Draw a vertical line at the calculated dominant frequency
    ax.axvline(x=dominant_freq, color='red', linestyle=':', linewidth=1.5, 
               label=f"Peak: {dominant_freq:.2f} Hz")
    
    # Styling
    ax.set_title(f"{target['title']}\n(ID: {row['evid']})")
    ax.set_xlabel("Frequency (Hz)")
    if i == 0:
        ax.set_ylabel("Spectral Amplitude")
    ax.set_xlim(0, sampling_rate / 2)  # Limit to Nyquist frequency
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper right")

print("------------------------------------------------")
plt.tight_layout()
plt.show()