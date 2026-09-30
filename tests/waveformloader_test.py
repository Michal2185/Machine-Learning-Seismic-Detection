from src.modules.data_loader import DataLoader
from src.modules.waveform_loader import WaveformLoader


catalog_path = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

data_dir = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


# Load catalogue
data_loader = DataLoader(
    catalog_path=catalog_path,
    data_dir=data_dir,
)

events = data_loader.run()

print(f"Loaded {len(events)} events")


# Load waveforms
waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(events)

print(f"Loaded {len(waveforms)} waveforms")


# Inspect first waveform
record = waveforms[0]

print()
print("Event:")
print(record.event)

print()
print("Stream:")
print(record.stream)

print()
print("Number of traces:", len(record.stream))