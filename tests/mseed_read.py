import pandas as pd
import matplotlib.pyplot as plt
from obspy import read

INDEX = 10

path = "F:/Git/Machine-Learning-Seismic-Detection/data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
data_dir = "F:/Git/Machine-Learning-Seismic-Detection/data/lunar/data/S12_GradeA/"

df = pd.read_csv(path)
row = df.iloc[INDEX]

filename = data_dir + row['filename'] + ".mseed"
print(f"Reading file: {filename}")

arrival_time = row['time_rel(sec)']
event_type = row['mq_type']

st = read(filename)
trace = st[0]

plt.figure(figsize=(10, 4))
plt.plot(trace.times(), trace.data, color='black', linewidth=0.5)
plt.axvline(x=arrival_time, color='red', linestyle='--', label='Arrival Time')

plt.title(f"Index {INDEX} | Event Type: {event_type}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()