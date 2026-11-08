"""
Saving and restoring FITTED detectors (one checkpoint per configuration, seed and fold).

A checkpoint holds everything a frozen detector needs to score new data without retraining:
  * the learner state (network weights; logistic weights for the numpy test learner),
  * the input state (N0 constant, feature standardisation, envelope length) so the same numbers go in,
  * the decision thresholds chosen on the CLEAN validation streams of that fold (one per false-alarm budget),
  * a little metadata (names, seed, fold, format version).

Detectors built on the Learner interface (LearnerDetector: DNN, CNN, GRU, CfC) are stored explicitly as plain
data. Anything else (RFC, amplitude control) falls back to pickling the fitted detector object.
Files are Python pickles: load only checkpoints you wrote yourself.

N2 / N3 are not supported: their scale tracks depend on the whole recording and live outside the detector.
"""

import pickle
import time
from pathlib import Path

import numpy as np

FORMAT = 1


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def is_learner_detector(det) -> bool:
    """Duck-typed LearnerDetector: has a learner (get_state / set_state / reset) and an input builder."""

    lr = getattr(det, "learner", None)

    return (
        lr is not None
        and hasattr(det, "inputs")
        and all(hasattr(lr, m) for m in ("get_state", "set_state", "reset"))
    )


def _to_cpu(obj):
    """Recursively move tensors to the CPU (checkpoints must load on a machine without the GPU)."""

    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().clone()

    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)

    return obj


def _normalizer_state(n) -> dict:

    variant = n.config.variant

    if variant in ("N2", "N3"):
        raise NotImplementedError(f"{variant} is not supported by checkpoints (scale tracks live outside the detector)")

    return dict(variant=variant, fixed_scale=None if n.fixed_scale is None else float(n.fixed_scale))


def _load_normalizer(n, s) -> None:

    if n.config.variant != s["variant"]:
        raise ValueError(f"checkpoint was saved with {s['variant']}, detector uses {n.config.variant}")

    n.fixed_scale = s["fixed_scale"]


def _inputs_state(inp) -> dict:

    st = dict(class_name=type(inp).__name__, normalizer=_normalizer_state(inp.normalizer))

    if hasattr(inp, "n_sub") and hasattr(inp, "mean_"):                       # EnvelopeInput
        st.update(
            kind="envelope", n_sub=int(inp.n_sub), log_amp=bool(inp.log_amp), scale=bool(inp.scale),
            mean=None if inp.mean_ is None else np.asarray(inp.mean_, dtype=np.float64),
            std=None if inp.std_ is None else np.asarray(inp.std_, dtype=np.float64),
            length=inp._length,
        )

    elif hasattr(inp, "scaler") and hasattr(inp, "fs"):                        # FeatureInput
        st.update(kind="features", fs=float(inp.fs),
                  scaler=None if inp.scaler is None else pickle.dumps(inp.scaler))

    elif hasattr(inp, "layout"):                                               # RawWindowInput
        st.update(kind="raw", layout=inp.layout)

    else:
        raise TypeError(f"unsupported input builder {type(inp).__name__}")

    return st


def _load_inputs(inp, st) -> None:

    if type(inp).__name__ != st["class_name"]:
        raise ValueError(f"checkpoint input is {st['class_name']}, detector uses {type(inp).__name__}")

    _load_normalizer(inp.normalizer, st["normalizer"])

    if st["kind"] == "envelope":
        if (inp.n_sub, inp.log_amp, inp.scale) != (st["n_sub"], st["log_amp"], st["scale"]):
            raise ValueError("envelope input settings differ from the checkpoint")
        inp.mean_, inp.std_, inp._length = st["mean"], st["std"], st["length"]

    elif st["kind"] == "features":
        if abs(float(inp.fs) - st["fs"]) > 1e-9:
            raise ValueError("feature input sampling rate differs from the checkpoint")
        inp.scaler = None if st["scaler"] is None else pickle.loads(st["scaler"])

    elif st["kind"] == "raw":
        if inp.layout != st["layout"]:
            raise ValueError("raw window layout differs from the checkpoint")


# ----------------------------------------------------------------------
# detector <-> state
# ----------------------------------------------------------------------

def detector_state(det, thresholds=None, meta=None) -> dict:
    """Plain-data state of a FITTED detector, with the thresholds chosen for it."""

    if is_learner_detector(det):

        if not getattr(det, "_fitted", False):
            raise RuntimeError("detector is not fitted")

        body = dict(
            mode="learner",
            learner_class=type(det.learner).__name__,
            learner=_to_cpu(det.learner.get_state()),
            inputs=_inputs_state(det.inputs),
            seed=int(det.config.seed),
        )

    else:
        try:
            body = dict(mode="pickle", blob=pickle.dumps(det, protocol=4))
        except Exception as e:                                                  # lambdas, open handles...
            raise TypeError(f"cannot checkpoint {type(det).__name__}: {e}") from e

    return dict(
        format=FORMAT,
        name=getattr(det, "name", ""),
        thresholds={float(b): float(t) for b, t in (thresholds or {}).items()},
        meta=dict(meta or {}, saved=time.strftime("%Y-%m-%d %H:%M:%S")),
        **body,
    )


def load_detector_state(det, state) -> tuple:
    """Restores a detector IN PLACE from a state (the detector must have been built like the saved one)."""

    if state.get("format") != FORMAT:
        raise ValueError(f"unknown checkpoint format {state.get('format')!r}")

    if state["mode"] == "learner":

        if not is_learner_detector(det):
            raise TypeError("checkpoint holds a learner detector, the target is not one")

        if type(det.learner).__name__ != state["learner_class"]:
            raise ValueError(f"checkpoint learner is {state['learner_class']}, target uses {type(det.learner).__name__}")

        _load_inputs(det.inputs, state["inputs"])
        det.learner.reset(state["seed"])                    # builds the architecture of the target
        det.learner.set_state(state["learner"])             # raises if the weights do not fit it
        det._fitted = True

    else:
        loaded = pickle.loads(state["blob"])

        if type(loaded) is not type(det):
            raise TypeError(f"checkpoint holds {type(loaded).__name__}, target is {type(det).__name__}")

        det.__dict__.clear()
        det.__dict__.update(loaded.__dict__)

    return dict(state["thresholds"]), dict(state["meta"])


# ----------------------------------------------------------------------
# files
# ----------------------------------------------------------------------

def checkpoint_path(root, config: str, seed: int, fold: int) -> Path:
    return Path(root) / config / f"s{seed}" / f"fold{fold}.pkl"


def save_checkpoint(root, config: str, seed: int, fold: int, det, thresholds=None, meta=None) -> Path:

    path = checkpoint_path(root, config, seed, fold)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = detector_state(det, thresholds, dict(meta or {}, config=config, seed=seed, fold=fold))
    tmp = path.with_suffix(".tmp")

    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=4)

    tmp.replace(path)                                        # never leave a half-written checkpoint

    return path


def load_checkpoint(root, config: str, seed: int, fold: int, det) -> tuple:
    """Restores `det` in place; returns (thresholds {budget: value}, meta)."""

    path = checkpoint_path(root, config, seed, fold)

    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {path}")

    with open(path, "rb") as f:
        state = pickle.load(f)

    return load_detector_state(det, state)


def verify_roundtrip(live_det, fresh_det, root, config: str, seed: int, fold: int, windows,
                     streams_by_id=None, rtol: float = 1e-5) -> float:
    """
    Loads the saved checkpoint into `fresh_det` and compares its scores with the live detector's on `windows`.
    Returns the largest absolute difference; raises if it exceeds rtol x the score range.
    """

    load_checkpoint(root, config, seed, fold, fresh_det)

    a = np.asarray(live_det.score(windows, streams_by_id or {}), dtype=np.float64)
    b = np.asarray(fresh_det.score(windows, streams_by_id or {}), dtype=np.float64)

    diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    scale = max(float(np.max(np.abs(a))) if len(a) else 0.0, 1e-12)

    if diff > rtol * scale:
        raise AssertionError(f"loaded checkpoint scores differ from the live detector: max |diff| {diff:.3g} "
                             f"(score scale {scale:.3g})")

    return diff