import numpy as np

from src.modules.detectors.types import Detector
from src.modules.dnn_features import FEATURE_NAMES
from src.modules.training import FeatureInput, ResourceReport

PRESETS = {"small": dict(n_estimators=50, max_depth=6), "medium": dict(n_estimators=100, max_depth=8)}

# packed node: feature id 1 B, left 2 B, right 2 B, + threshold and leaf value
NODE_BYTES_FP32 = 1 + 2 + 2 + 4 + 4
NODE_BYTES_INT8 = 1 + 2 + 2 + 1 + 1


class RandomForestDetector(Detector):
    """
    Random forest on the 6 window features (trees need no feature scaling).
    Training set per fit: all positive windows + the union of the background
    windows of `n_draws` Sampler draws (new negatives per draw); ignore windows
    are never used. Score = forest probability of "event".
    """

    def __init__(self, name: str, input_builder, n_estimators: int = 100, max_depth: int = 8,
                 min_samples_leaf: int = 5, n_draws: int = 10, negatives_per_positive: float = 5.0,
                 seed: int = 0, n_jobs: int = 1, sampler_factory=None):

        self.name = name
        self.inputs = input_builder
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_draws = n_draws
        self.negatives_per_positive = negatives_per_positive
        self.seed = seed
        self.n_jobs = n_jobs
        self.sampler_factory = sampler_factory
        self.forest = None

    def _sampler(self):

        if self.sampler_factory is not None:
            return self.sampler_factory(self)

        from src.modules.sampler.module import Sampler

        return Sampler(self.negatives_per_positive, self.seed)

    def fit(self, train_windows) -> "RandomForestDetector":

        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as e:
            raise ImportError("RandomForestDetector needs scikit-learn") from e

        train_windows = list(train_windows)
        self.inputs.fit(train_windows)

        sampler = self._sampler()
        positives, negatives = {}, {}

        for epoch in range(self.n_draws):
            for w in sampler.run(train_windows, epoch).windows:
                (positives if w.label == 1 else negatives)[id(w)] = w

        chosen = list(positives.values()) + list(negatives.values())
        y = np.array([1] * len(positives) + [0] * len(negatives))

        self.forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=self.n_jobs,
        ).fit(self.inputs.build(chosen), y)

        return self

    def score(self, windows, streams_by_id=None) -> np.ndarray:

        if self.forest is None:
            raise RuntimeError("detector is not fitted")

        windows = list(windows)
        out = []

        for i in range(0, len(windows), 8192):
            out.append(self.forest.predict_proba(self.inputs.build(windows[i:i + 8192]))[:, 1])

        return np.concatenate(out) if out else np.zeros(0)

    def importances(self) -> dict:
        """Mean decrease in impurity per feature (sums to 1)."""

        if self.forest is None:
            raise RuntimeError("detector is not fitted")

        return dict(zip(FEATURE_NAMES, self.forest.feature_importances_.tolist()))

    def resource_report(self, example_window, n_samples: int | None = None) -> ResourceReport:

        if self.forest is None:
            raise RuntimeError("detector is not fitted")

        trees = [e.tree_ for e in self.forest.estimators_]
        nodes = int(sum(t.node_count for t in trees))
        compares = int(sum(t.max_depth for t in trees))      # worst-case comparisons per window

        n = n_samples or len(example_window.data)

        return ResourceReport(
            params=nodes,
            bytes_fp32=nodes * NODE_BYTES_FP32,
            bytes_int8=nodes * NODE_BYTES_INT8,
            macs_per_window=compares,
            state_floats=0,
            input_ops=self.inputs.ops_per_window(n),
            notes=["params = tree nodes; 'MACs' = worst-case comparisons (no multiplies); "
                   "bytes are estimates for a packed node array, not the sklearn pickle size"],
        )


def build_rfc_detector(name: str, normalizer, fs: float, size: str = "medium", seed: int = 0, **kw):
    """RFC on unscaled 6 features of the normalised window (variant in `normalizer`)."""

    return RandomForestDetector(
        name, FeatureInput(normalizer, fs, scale=False), seed=seed, **{**PRESETS[size], **kw}
    )