from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader


catalog_path = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

data_dir = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


# --------------------------------------------------
# 1. Load catalogue
# --------------------------------------------------

data_loader = DataLoader(
    catalog_path=catalog_path,
    data_dir=data_dir,
)

events = data_loader.run()

print(f"Catalogue events: {len(events)}")


# --------------------------------------------------
# 2. Validate catalogue
# --------------------------------------------------

validator = Validator(data_dir)

validation_report = validator.run(events)

print(f"Valid events:     {validation_report.valid_count}")
print(f"Rejected events:  {validation_report.rejected_count}")


# --------------------------------------------------
# 3. Load validated waveforms
# --------------------------------------------------

waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(
    validation_report.valid_events
)

print(f"Loaded waveforms: {len(waveforms)}")


# --------------------------------------------------
# 4. Inspect first waveform
# --------------------------------------------------

record = waveforms[0]

print()
print("Event:")
print(record.event)

print()
print("Stream:")
print(record.stream)

print()
print(f"Number of traces: {len(record.stream)}")

trace = record.stream[0]
"""
for record in waveforms:
    print(
        record.event.evid,
        len(record.stream),
        record.stream[0].stats.sampling_rate,
        record.stream[0].stats.npts,
    )
"""