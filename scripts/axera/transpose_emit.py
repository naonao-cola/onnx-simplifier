"""Emit AX650 ``Transpose`` models by patching a measured template (untiled regime).

Scope, and only this: float32 ``Transpose(x[1,1,R,C], perm=[0,1,3,2])`` for the
``(R, q)`` pairs in ``fixtures/transpose_untiled/index.json``, where
``q = ceil(C/8)`` and ``C`` is not a multiple of 8. Pulsar2 7.0-lite compiles
every ``C`` of one ``(R, q)`` block to the same MCode form; only a few bytes
(the low and high bytes of ``4*C-1``, ``4*C``, ``4*R*C-1`` and ``4*R*C``, at 11-16
positions) change. ``scripts/axera/transpose_fields.py`` fits those positions
from several builds of one block; the committed index stores the fitted field
map and one compiled template per block, and this module patches the template
for another ``C`` in the same block.

What it does not do: predict which form a *new* ``(R, q)`` block takes. That
class is not a simple function of the shape (it recurs irregularly in ``q``), so
a block that is not in the index is rejected instead of guessed. The tiled
regime (larger tensors, including every Transpose in the ResNet18 step) is out
of scope too; see ``docs/axera-transpose-untiled.md``.
"""

from __future__ import annotations

import gzip
import json
import os

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, "fixtures", "transpose_untiled")
_NOISE_START = 301
_NOISE_END = 326

_FEATURES = {
    "4RC-1": lambda r, c: 4 * r * c - 1,
    "4RC": lambda r, c: 4 * r * c,
    "4C-1": lambda r, c: 4 * c - 1,
    "4C": lambda r, c: 4 * c,
    "32q": lambda r, c: 32 * ((c + 7) // 8),
    "4q": lambda r, c: 4 * ((c + 7) // 8),
    "2q": lambda r, c: 2 * ((c + 7) // 8),
    "2q-2": lambda r, c: 2 * ((c + 7) // 8) - 2,
    "q-1": lambda r, c: (c + 7) // 8 - 1,
    "Rq-1": lambda r, c: r * ((c + 7) // 8) - 1,
}


def _index() -> dict[tuple[int, int], dict]:
    with open(os.path.join(_DIR, "index.json")) as f:
        entries = json.load(f)
    return {(e["R"], e["q"]): e for e in entries}


def measured_blocks() -> list[tuple[int, int]]:
    """The ``(R, q)`` blocks a template exists for."""
    return sorted(_index())


def _neu(model: onnx.ModelProto):
    matches = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *_neu initializer, found {len(matches)}"
        )
    return matches[0]


def _set_dims(value_info, shape: list[int]) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for size in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = size


def _dims(value_info) -> list[int]:
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _check_structure(model: onnx.ModelProto) -> tuple[int, int]:
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("reference must contain exactly one compiled 'neu mode' node")
    if list(model.graph.node[0].input) != ["x"] or list(model.graph.node[0].output) != [
        "y"
    ]:
        raise ValueError("reference NPU node must map input 'x' to output 'y'")
    if [v.name for v in model.graph.input] != ["x"] or [
        v.name for v in model.graph.output
    ] != ["y"]:
        raise ValueError("reference must have one input 'x' and one output 'y'")
    x, y = model.graph.input[0], model.graph.output[0]
    for v in (x, y):
        if v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("reference must be float32")
    xs, ys = _dims(x), _dims(y)
    if len(xs) != 4 or xs[:2] != [1, 1] or ys != [1, 1, xs[3], xs[2]]:
        raise ValueError(
            f"reference is not a [1,1,R,C] -> [1,1,C,R] Transpose: {xs}->{ys}"
        )
    return xs[2], xs[3]


def emit_transpose_axmodel(
    output_path: str, *, shape: tuple[int, int, int, int]
) -> str:
    """Write a compiled ``Transpose(x[shape], perm=[0,1,3,2])`` axmodel.

    ``shape`` is ``(1, 1, R, C)`` with ``C`` not a multiple of 8, and ``(R,
    ceil(C/8))`` must be a block in the index (``measured_blocks()``); anything
    else raises ``ValueError``. The result is the block's compiled template with
    the fitted MCode bytes rewritten for ``C`` and the tensor dims updated.
    """
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
        raise ValueError("shape must be (1, 1, R, C)")
    r, c = shape[2], shape[3]
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (r, c)) or c < 1:
        raise ValueError("R and C must be positive integers")
    if c % 8 == 0:
        raise ValueError("C is a multiple of 8; only unaligned C blocks are measured")
    q = (c + 7) // 8
    entry = _index().get((r, q))
    if entry is None:
        raise ValueError(
            f"(R={r}, q={q}) is not a measured block; measured: {measured_blocks()}"
        )

    with gzip.open(os.path.join(_DIR, entry["template"]), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    tr, tc = _check_structure(model)
    if (tr, (tc + 7) // 8) != (r, q):
        raise ValueError("template does not match its index entry")

    neu = _neu(model)
    code = bytearray(neu.raw_data)
    for pos_s, (name, idx) in entry["fields"].items():
        code[int(pos_s)] = (_FEATURES[name](r, c) >> (8 * idx)) & 0xFF
    neu.raw_data = bytes(code)

    _set_dims(model.graph.input[0], [1, 1, r, c])
    _set_dims(model.graph.output[0], [1, 1, c, r])
    for v in model.graph.value_info:
        if v.name == "x":
            _set_dims(v, [1, 1, r, c])
        elif v.name == "y":
            _set_dims(v, [1, 1, c, r])
    for attr in model.graph.node[0].attribute:
        if attr.name == "outputs_info":
            attr.s = json.dumps({"y": ["FP32", [1, 1, c, r]]}).encode()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
