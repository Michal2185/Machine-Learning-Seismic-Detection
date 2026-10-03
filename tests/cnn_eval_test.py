import torch
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
    Evaluator,
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

model = SeismicCNN()

checkpoint = torch.load(
    "checkpoints/seismic_cnn_best.pt",
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

evaluator = Evaluator(
    model=model,
    batch_size=32,
    threshold=0.5,
)

result = evaluator.evaluate(
    test_dataset
)

print()
print("=" * 60)
print("CNN EVALUATION")
print("=" * 60)

print()

print("Confusion Matrix")
print("-----------------")

print(
    f"True Negative:  {result.true_negative}"
)

print(
    f"False Positive: {result.false_positive}"
)

print(
    f"False Negative: {result.false_negative}"
)

print(
    f"True Positive:  {result.true_positive}"
)

print()

print("Metrics")
print("-------")

print(
    f"Accuracy:  {result.accuracy:.4f}"
)

print(
    f"Precision: {result.precision:.4f}"
)

print(
    f"Recall:    {result.recall:.4f}"
)

print(
    f"F1:        {result.f1:.4f}"
)

print()

print("Dataset")
print("-------")

print(
    f"Total:     {result.total_samples}"
)

print(
    f"Positive:  {result.positive_samples}"
)

print(
    f"Negative:  {result.negative_samples}"
)