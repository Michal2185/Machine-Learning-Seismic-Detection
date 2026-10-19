import math

from .types import ResourceReport

# ---------------------------------------------------------------- pure formulas
# Multiply-accumulates for ONE sample (batch size 1); biases, activations,
# normalisation layers and elementwise gating are not counted.


def linear_macs(n_in: int, n_out: int, positions: int = 1) -> int:
    return positions * n_in * n_out


def conv1d_macs(in_ch: int, out_ch: int, kernel: int, out_len: int, groups: int = 1) -> int:
    return out_ch * out_len * (in_ch // groups) * kernel


def gru_macs(n_in: int, hidden: int, steps: int, layers: int = 1, bidirectional: bool = False) -> int:

    dirs = 2 if bidirectional else 1
    total = 0

    for layer in range(layers):
        layer_in = n_in if layer == 0 else hidden * dirs
        total += dirs * steps * 3 * (layer_in * hidden + hidden * hidden)

    return total


def lstm_macs(n_in: int, hidden: int, steps: int, layers: int = 1, bidirectional: bool = False) -> int:
    return gru_macs(n_in, hidden, steps, layers, bidirectional) * 4 // 3


def rnn_state_floats(hidden: int, layers: int = 1, bidirectional: bool = False, lstm: bool = False) -> int:
    return hidden * layers * (2 if bidirectional else 1) * (2 if lstm else 1)


def model_bytes(n_params: int, n_tensors: int, bits: int) -> int:
    """fp32: 4 bytes per weight. int8: 1 byte per weight + one fp32 scale per tensor."""

    if bits == 32:
        return 4 * n_params

    if bits == 8:
        return n_params + 4 * n_tensors

    raise ValueError("bits must be 32 or 8")


# ---------------------------------------------------------------- torch hooks

def count_model_resources(model, example_x, input_ops: int = 0) -> ResourceReport:
    """
    Parameters, sizes, MACs of one forward pass over ONE window and
    recurrent state size. Supported leaf layers: Linear, Conv1d, GRU, LSTM,
    GRUCell. Other parametric layers are listed in `notes` (their MACs are
    NOT counted: add a formula before trusting a number). Custom layers
    (CfC, LTC) may define `macs_per_step(n_in)` / `state_size`; see handoff.
    """

    import numpy as np
    import torch
    import torch.nn as nn

    macs = [0]
    state = [0]
    notes = []
    handles = []

    def hook(mod, inp, out):

        x = inp[0]

        if isinstance(mod, nn.Linear):
            positions = int(np.prod(x.shape[1:-1])) if x.dim() > 2 else 1
            macs[0] += linear_macs(mod.in_features, mod.out_features, positions)

        elif isinstance(mod, nn.Conv1d):
            macs[0] += conv1d_macs(mod.in_channels, mod.out_channels, mod.kernel_size[0],
                                   out.shape[-1], mod.groups)

        elif isinstance(mod, (nn.GRU, nn.LSTM)):
            steps = x.shape[1] if mod.batch_first else x.shape[0]
            f = gru_macs if isinstance(mod, nn.GRU) else lstm_macs
            macs[0] += f(mod.input_size, mod.hidden_size, steps, mod.num_layers, mod.bidirectional)
            state[0] += rnn_state_floats(mod.hidden_size, mod.num_layers, mod.bidirectional,
                                         isinstance(mod, nn.LSTM))

        elif isinstance(mod, nn.GRUCell):
            macs[0] += 3 * (mod.input_size * mod.hidden_size + mod.hidden_size ** 2)
            state[0] += mod.hidden_size

        elif hasattr(mod, "macs_per_step"):
            steps = x.shape[1] if x.dim() == 3 else 1
            macs[0] += steps * int(mod.macs_per_step(x.shape[-1]))
            state[0] += int(getattr(mod, "state_size", 0))

        elif any(True for _ in mod.parameters(recurse=False)):
            if type(mod).__name__ not in notes:
                notes.append(type(mod).__name__)

    for mod in model.modules():
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(hook))

    was_training = model.training
    model.eval()

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(example_x)[:1], dtype=torch.float32)
        model(x)

    model.train(was_training)

    for h in handles:
        h.remove()

    tensors = list(model.parameters())
    n_params = int(sum(p.numel() for p in tensors))

    return ResourceReport(
        params=n_params,
        bytes_fp32=model_bytes(n_params, len(tensors), 32),
        bytes_int8=model_bytes(n_params, len(tensors), 8),
        macs_per_window=int(macs[0]),
        state_floats=int(state[0]),
        input_ops=input_ops,
        notes=[f"MACs not counted for layer type {n}" for n in notes],
    )