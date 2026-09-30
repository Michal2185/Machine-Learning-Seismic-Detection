from src.modules.data_loader import DataLoader
from src.modules.validator import Validator, ValidationStatus


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


# Validate events
validator = Validator(data_dir)

results = validator.run(events)


# Summary
valid = sum(
    result.status == ValidationStatus.VALID
    for result in results
)

warnings = sum(
    result.status == ValidationStatus.WARNING
    for result in results
)

invalid = sum(
    result.status == ValidationStatus.INVALID
    for result in results
)


print()
print("Validation summary")
print("------------------")
print(f"Total:    {len(results)}")
print(f"Valid:    {valid}")
print(f"Warnings: {warnings}")
print(f"Invalid:  {invalid}")


# Print warnings
print()
print("Warnings")
print("--------")

for result in results:
    if result.status == ValidationStatus.WARNING:
        print(
            f"{result.event.evid}: "
            f"{result.message}"
        )