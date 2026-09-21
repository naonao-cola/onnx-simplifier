"""Evidence-scoped emitter for one non-fused AX650 ``Reshape`` DMA family.

Most non-fused ``Reshape -> Relu`` programs change many independent fields
whenever a dimension changes (see ``docs/axera-reshape-dma.md``), so they cannot
be produced by patching. One family is different: the weight fold
``[Co,8,3,3] -> [1,Co,8,9]`` (``Cin=8``) followed by ``Relu`` has, for ``Co`` in
``EXACT_CO``, exactly four bytes of segment 2 that depend on ``Co`` and nothing
else does, plus two 16-bit tensor-size words in the loader tail. All are affine in
``Co`` and were verified against compiler-built models at held-out ``Co`` values.

This module retargets a committed compiled model of that family to another
``Co`` in ``EXACT_CO`` by rewriting those four bytes and the graph
dimensions. It refuses every other shape: outside ``EXACT_CO`` the program
changes layout, and ``Cin`` other than 8 was not characterized.

All offsets are absolute positions in the ``*_neu`` blob.
"""

from __future__ import annotations

import gzip
import json
import os

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.join(_HERE, "fixtures", "reshape_dma", "w1_co40.axmodel.gz")

CIN = 8
TEMPLATE_CO = 40
# Co values whose compiler-built program equals the prediction below outside the
# 301-325 noise window. The formulas were fitted on 40, 44, 48, 52, 56; the other
# nine were held out. The set is NOT an interval: Co=36 and Co=60 (both even, next
# to exact values) change layout, as do all odd Co and every Co >= 66 measured
# (docs/axera-reshape-dma.md), so nothing is interpolated.
EXACT_CO = frozenset({34, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 62, 64})

_NOISE = (301, 326)
# (absolute offset in the MCode blob, function of Co) for the four bytes inside
# the segment-2 program (segment 2 starts at blob offset 380).
FIELDS = (
    (936, lambda co: (co - 1) & 0xFF),
    (1192, lambda co: (8 * co - 1) & 0xFF),
    (1212, lambda co: (co // 2) & 0xFF),
    (1344, lambda co: co & 0xFF),
)
# Two little-endian uint16 tensor-size words in the loader tail: 4 * 8 * 9 * Co.
SIZE_OFFSETS = (2000, 2160)


def _size_word(co: int) -> bytes:
    return (4 * CIN * 9 * co).to_bytes(2, "little")


def _neu(model: onnx.ModelProto):
    matches = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(f"expected one *_neu initializer, found {len(matches)}")
    return matches[0]


def predict_mcode(template: bytes, co: int) -> bytes:
    """MCode of ``Reshape([co,8,3,3] -> [1,co,8,9]) -> Relu`` from the Co=40 build."""
    out = bytearray(template)
    for offset, fn in FIELDS:
        out[offset] = fn(co)
    for offset in SIZE_OFFSETS:
        out[offset : offset + 2] = _size_word(co)
    return bytes(out)


def _set_dims(value_info, shape) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for size in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = size


def emit_weight_fold_axmodel(output_path: str, *, co: int, cin: int = CIN) -> str:
    """Emit the compiled ``Reshape([co,8,3,3] -> [1,co,8,9]) -> Relu`` model."""
    if not isinstance(co, int) or isinstance(co, bool):
        raise ValueError("co must be an integer")
    if cin != CIN:
        raise ValueError(f"only cin={CIN} was characterized, got {cin}")
    if co not in EXACT_CO:
        raise ValueError(
            f"co={co} is not one of the measured-exact values {sorted(EXACT_CO)}; "
            "the program changes layout elsewhere"
        )
    with gzip.open(_FIXTURE, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    node = model.graph.node[0]
    if len(model.graph.node) != 1 or node.op_type != "neu mode":
        raise ValueError("template must contain exactly one 'neu mode' node")
    neu = _neu(model)
    neu.raw_data = predict_mcode(bytes(neu.raw_data), co)

    in_shape, out_shape = [co, CIN, 3, 3], [1, co, CIN, 9]
    _set_dims(model.graph.input[0], in_shape)
    _set_dims(model.graph.output[0], out_shape)
    for value_info in model.graph.value_info:
        if value_info.name == "x":
            _set_dims(value_info, in_shape)
        elif value_info.name == "y":
            _set_dims(value_info, out_shape)
    for attr in node.attribute:
        if attr.name == "outputs_info":
            attr.s = json.dumps({"y": ["FP32", out_shape]}).encode()
        elif attr.name == "inputs_info":
            attr.s = json.dumps({"x": ["FP32", in_shape]}).encode()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
