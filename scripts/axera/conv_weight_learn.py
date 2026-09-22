"""Extend `emitter.py`'s bit-permutation weight learner with the one term it
omits for a real, biased convolution -- and the scale extraction a real
Pulsar2 build needs it fed from.

`emitter.py`'s `requant_block()` derives the per-channel `(bias, M)` block a
quantised convolution's `npu_params` carries from `x_scale`, `w_scale`,
`y_scale`, `x_zero`, `y_zero` and the weight codes alone. That is exactly
right for a `Conv` with **no** float bias input -- every existing test and
the README's own validated derivation used one. Every real ResNet18 `Conv`
node has a bias (confirmed directly against `t6-r18fold/step.onnx`: all 20
forward `Conv`s carry a third input). A biased convolution's quantised output
is

    y_code = zy + [ sum_i (x_code_i - zx) * q_i * (x_scale * w_scale) + bias_float ] / y_scale

so the stored per-channel bias term needs one more addend than
`requant_block()` computes:

    bias[c] = zy - zx * sum(q_c) * M_c + bias_float[c] / y_scale

Read directly off a real `Conv(cin=64, cout=64, k=3x3, stride=1, pad=1,
batch=16, spatial=56)` build with a real bias input: the *uncorrected*
formula was off by exactly `bias_float / y_scale` (max residual 1.9e-4 across
48 builds after subtracting it, the same float32-rounding order of magnitude
`emitter.py`'s own no-bias derivation reports as 6.1e-05). See
`docs/axera-conv-weight-learn-stem.md` for the full campaign.
"""

from __future__ import annotations

import json
import os

import numpy as np


def requant_block_biased(codes, x_scale, x_zero, y_scale, y_zero, w_scale, bias_float):
    """`emitter.requant_block`, plus the float-bias term a real `Conv` needs.

    Import `emitter` yourself and pass `emitter.requant_block` semantics
    through this module rather than editing `emitter.py` -- this function
    only adds the missing addend on top of it.
    """
    import emitter

    block = emitter.requant_block(codes, x_scale, x_zero, y_scale, y_zero, w_scale)
    block = np.array(block, copy=True)
    view = block.view(np.float32)
    channels = len(bias_float)
    view[:channels] += (
        np.asarray(bias_float, np.float32) / np.float32(y_scale)
    ).astype(np.float32)
    return block


def emit_biased(
    reference_table, origin, w, bias, x_scale, x_zero, y_scale, y_zero, block_at
):
    """`emitter.emit`, with the biased requant block above in place of
    `emitter.requant_block`'s direct call."""
    import emitter

    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes)
    w_scale = emitter.weight_scales(w)
    block = requant_block_biased(codes, x_scale, x_zero, y_scale, y_zero, w_scale, bias)
    table = np.array(table, copy=True)
    table[block_at : block_at + len(block)] = block
    return table


def tensor_scales(quant_axmodel_json_path: str) -> dict:
    """`{tensor_name: {"scale": [...], "zero_point": [...]}}` from a Pulsar2
    build's own `out/quant/quant_axmodel.json` -- x/w/b/y for a single-`Conv`
    reference. There is one `tensor_configs` entry per graph *output*; a
    single-node model has exactly one, keyed by that output's name."""
    with open(quant_axmodel_json_path) as f:
        doc = json.load(f)
    out = {}
    for cfg in doc["tensor_configs"].values():
        for tensor, entry in cfg.items():
            value = doc["values"].get(str(entry.get("hash")))
            if value:
                out[tensor] = value
    return out


def build_scales(build_dir: str, output_name: str = "y") -> dict:
    """`tensor_scales` for one `pulsar2 build` output directory."""
    return tensor_scales(os.path.join(build_dir, "quant", "quant_axmodel.json"))
