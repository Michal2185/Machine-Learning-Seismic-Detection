import copy

import numpy as np

from .types import ResourceReport


class NumpyLogisticLearner:
    """
    Logistic regression trained with plain SGD: dependency-free baseline and
    test double for the harness. Learner interface:

        reset(seed)                       fresh weights
        train_step(x, y, pos_weight)      one minibatch, returns the mean loss
        predict(x)                        logits (higher = more event)
        get_state() / set_state(state)    for restoring the best epoch
    """

    def __init__(self, lr: float = 0.1, l2: float = 1e-4):
        self.lr = lr
        self.l2 = l2
        self.w = None
        self.b = 0.0

    def reset(self, seed: int) -> None:
        self._n_in = None
        self.w = None
        self.b = 0.0
        self._rng = np.random.default_rng([seed, 5])

    def _init(self, n_in: int) -> None:
        self.w = self._rng.normal(0.0, 0.01, n_in)
        self.b = 0.0

    def train_step(self, x, y, pos_weight: float) -> float:

        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
        y = np.asarray(y, dtype=np.float64)

        if self.w is None:
            self._init(x.shape[1])

        z = x @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        sw = np.where(y > 0.5, pos_weight, 1.0)

        eps = 1e-12
        loss = -np.mean(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

        g = sw * (p - y) / len(y)
        self.w -= self.lr * (x.T @ g + self.l2 * self.w)
        self.b -= self.lr * g.sum()

        return float(loss)

    def predict(self, x) -> np.ndarray:

        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)

        if self.w is None:
            return np.zeros(len(x))

        return x @ self.w + self.b

    def get_state(self):
        return copy.deepcopy((self.w, self.b))

    def set_state(self, state) -> None:
        self.w, self.b = copy.deepcopy(state)

    def resources(self, example_x, input_ops: int = 0) -> ResourceReport:

        n = int(np.prod(np.asarray(example_x).shape[1:])) + 1

        return ResourceReport(
            params=n, bytes_fp32=4 * n, bytes_int8=n + 4, macs_per_window=n - 1,
            state_floats=0, input_ops=input_ops,
        )


class TorchLearner:
    """
    PyTorch learner (same interface). model_factory() must return a fresh
    nn.Module whose forward gives one logit per window, shape (B,) or (B, 1).
    Weights are re-created in reset(seed), so every fold and seed starts fresh.
    """

    def __init__(self, model_factory, lr: float = 1e-3, weight_decay: float = 1e-4,
                 device: str = "cpu", grad_clip: float | None = 1.0):

        self.model_factory = model_factory
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.grad_clip = grad_clip
        self.model = None
        self.opt = None

    def reset(self, seed: int) -> None:

        import torch

        torch.manual_seed(seed)
        self.model = self.model_factory().to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_step(self, x, y, pos_weight: float) -> float:

        import torch
        import torch.nn.functional as F

        self.model.train()

        xb = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        yb = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)

        logits = self.model(xb).reshape(len(yb))

        loss = F.binary_cross_entropy_with_logits(
            logits, yb, pos_weight=torch.tensor(float(pos_weight), device=self.device)
        )

        self.opt.zero_grad()
        loss.backward()

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.opt.step()

        return float(loss.item())

    def predict(self, x) -> np.ndarray:

        import torch

        self.model.eval()
        out = []

        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.as_tensor(np.asarray(x[i:i + 4096]), dtype=torch.float32,
                                     device=self.device)
                out.append(self.model(xb).reshape(len(xb)).cpu().numpy())

        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def get_state(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state) -> None:
        self.model.load_state_dict(state)

    def resources(self, example_x, input_ops: int = 0) -> ResourceReport:

        from .resources import count_model_resources

        return count_model_resources(self.model, example_x, input_ops=input_ops)