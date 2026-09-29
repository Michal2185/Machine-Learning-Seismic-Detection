from src.modules.data_loader import DataLoader


catalog_path = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

data_dir = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


loader = DataLoader(
    catalog_path=catalog_path,
    data_dir=data_dir,
)

events = loader.run()


print(f"Loaded {len(events)} events")

for event in events[:5]:
    print(event)