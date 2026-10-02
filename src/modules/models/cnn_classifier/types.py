from dataclasses import dataclass


@dataclass
class Prediction:
    logit: float
    probability: float