from src.modules.data_loader import DataLoader
from src.modules.validator import Validator


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

print(f"Loaded {len(events)} catalogue events")


# Validate
validator = Validator(data_dir)

report = validator.run(events)


print()
print("Validation summary")
print("------------------")
print(f"Total:    {report.total}")
print(f"Valid:    {report.valid_count}")
print(f"Rejected: {report.rejected_count}")


print()
print("Rejected events")
print("----------------")

for result in report.rejected_events:
    print(
        f"{result.event.evid}: "
        f"{result.message}"
    )