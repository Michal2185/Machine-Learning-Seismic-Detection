from src.modules.data_loader import DataLoader
from src.modules.validator import Validator
from src.modules.waveform_loader import WaveformLoader
from src.modules.preprocessor import Preprocessor
from src.modules.windowing import Windowing, WindowConfig
from src.modules.labeling import Labeling
from src.modules.dataset_builder import DatasetBuilder
from src.modules.sampler import Sampler

from src.modules.models.cnn_classifier.model import SeismicCNN

from src.modules.training import (
    SeismicDataset,
    Trainer,
)


CATALOG_PATH = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/catalogs/apollo12_catalog_GradeA_final.csv"
)

DATA_DIR = (
    "F:/Git/Machine-Learning-Seismic-Detection/"
    "data/lunar/data/S12_GradeA/"
)


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

loader = DataLoader(
    catalog_path=CATALOG_PATH,
    data_dir=DATA_DIR,
)

events = loader.run()


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

validator = Validator(DATA_DIR)

validation_report = validator.run(events)


# ------------------------------------------------------------
# WAVEFORM LOADING
# ------------------------------------------------------------

waveform_loader = WaveformLoader()

waveforms = waveform_loader.run(
    validation_report.valid_events
)


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

preprocessor = Preprocessor()

processed = preprocessor.run(
    waveforms
)


# ------------------------------------------------------------
# WINDOWING
# ------------------------------------------------------------

windowing = Windowing(
    WindowConfig(
        window_length=60,
        step_size=10,
    )
)

windows = windowing.run(processed)


# ------------------------------------------------------------
# LABELING
# ------------------------------------------------------------

labeling = Labeling()

labeled_windows = labeling.run(
    windows
)


# ------------------------------------------------------------
# DATASET SPLIT
# ------------------------------------------------------------

dataset_builder = DatasetBuilder(
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)

dataset = dataset_builder.run(
    labeled_windows
)


# ------------------------------------------------------------
# SAMPLING
# ------------------------------------------------------------

sampler = Sampler(
    positive_to_negative_ratio=5,
    random_seed=42,
)

sampling_result = sampler.run(
    dataset.train
)


# ------------------------------------------------------------
# PYTORCH DATASETS
# ------------------------------------------------------------

train_dataset = SeismicDataset(
    sampling_result.windows
)

validation_dataset = SeismicDataset(
    dataset.validation
)

test_dataset = SeismicDataset(
    dataset.test
)


print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    f"Training:   {len(train_dataset)}"
)

print(
    f"Validation: {len(validation_dataset)}"
)

print(
    f"Test:       {len(test_dataset)}"
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = SeismicCNN()


# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------

trainer = Trainer(
    model=model,
    learning_rate=0.001,
    batch_size=32,
    epochs=20,
    checkpoint_path=(
        "checkpoints/seismic_cnn_best.pt"
    ),
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

result = trainer.fit(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best epoch: "
    f"{result.best_epoch}"
)

print(
    f"Best validation loss: "
    f"{result.best_validation_loss:.4f}"
)

print()
print(
    "Checkpoint saved to:"
)

print(
    trainer.checkpoint_path
)