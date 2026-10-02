import torch

from src.modules.models.cnn_classifier.model import SeismicCNN


print()
print("=" * 60)
print("CLASSIFIER MODEL TEST")
print("=" * 60)


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------

model = SeismicCNN()

print()
print(model)


# ------------------------------------------------------------
# Create example input
# ------------------------------------------------------------

batch_size = 8
number_of_samples = 398

x = torch.randn(
    batch_size,
    number_of_samples,
    1,
)


print()
print(
    "Input shape:",
    tuple(x.shape),
)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

with torch.no_grad():

    output = model(x)


print(
    "Output shape:",
    tuple(output.shape),
)


# ------------------------------------------------------------
# Convert logits to probabilities
# ------------------------------------------------------------

probabilities = torch.sigmoid(output)


print()
print("Probabilities:")

for probability in probabilities:

    print(
        f"{probability.item():.4f}"
    )


# ------------------------------------------------------------
# Assertions
# ------------------------------------------------------------

assert output.shape == (
    batch_size,
    1,
)

assert torch.all(
    probabilities >= 0
)

assert torch.all(
    probabilities <= 1
)


print()
print("Model test passed.")