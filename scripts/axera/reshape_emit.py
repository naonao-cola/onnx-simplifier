"""Evidence-scoped emitter for Reshape that Pulsar2 fuses away (AX650).

Pulsar2 7.0-lite compiles a ``Reshape`` next to a ``Relu`` in one of two ways,
depending only on the (input shape, output shape) pair:

* **Fused.** The compiled MCode is byte-identical (outside the known 301-325
  noise window) to a plain ``Relu`` compiled at the *other* tensor's shape:
  ``Reshape -> Relu`` equals ``Relu`` at the output shape, ``Relu -> Reshape``
  equals ``Relu`` at the input shape. The only differences are the graph
  input dims (or output dims and ``outputs_info``) and the matching
  ``value_info`` entry. This module retargets exactly those fields.
* **Not fused.** The Reshape becomes a real DMA program (a longer MCode blob,
  and for large tensors a byte-offset table in ``npu_params``) that depends on
  the shape pair in a way not decoded here. Those pairs are rejected.

Standalone ``Reshape`` (no neighbouring op) does not compile at all
(``ZeroDivisionError`` in the NPU backend), so a consumer is required.

The fused/not-fused split was measured pair by pair (see
``docs/axera-reshape.md``); no closed-form rule was found, so ``FUSED_BEFORE``
and ``FUSED_AFTER`` are lookup tables of measured pairs, not a predicate. Pairs
in neither table are unmeasured and are rejected rather than guessed.
Only ``Relu`` neighbours were measured.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Sequence

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fixtures", "reshape")
_NOISE_START = 301
_NOISE_END = 326

Shape = tuple[int, ...]

# (input shape, output shape) pairs where ``Reshape -> Relu`` compiled to the
# same MCode as ``Relu`` at the output shape.
FUSED_BEFORE: frozenset[tuple[Shape, Shape]] = frozenset(
    [
        ((1, 8, 4, 4), (1, 1, 8, 16)),
        ((1, 1, 8, 16), (1, 8, 4, 4)),
        ((1, 8), (8,)),
        ((16, 4, 4, 4), (16, 1, 4, 16)),
        ((1, 4, 4, 4), (1, 1, 4, 16)),
        ((1, 8, 4, 8), (1, 1, 8, 32)),
        ((1, 8, 8, 8), (1, 1, 8, 64)),
        ((1, 8, 16, 16), (1, 1, 8, 256)),
        ((1, 8, 2, 2), (1, 1, 8, 4)),
        ((1, 16, 8, 8), (1, 1, 16, 64)),
        ((1, 4, 4, 8), (1, 1, 4, 32)),
        ((1, 4, 4, 16), (1, 1, 4, 64)),
        ((1, 4, 4, 32), (1, 1, 4, 128)),
        ((1, 1, 8, 16), (1, 8, 2, 8)),
        ((1, 3, 4, 4), (1, 1, 3, 16)),
        ((1, 8, 4, 4), (1, 4, 8, 4)),
        ((1, 6, 4, 4), (1, 1, 6, 16)),
        ((1, 1, 8, 16), (1, 8, 16)),
        ((1, 8, 16), (1, 8, 4, 4)),
        ((1, 1, 8, 48), (1, 8, 48)),
        ((1, 1, 8, 48), (1, 8, 6, 8)),
        ((1, 8, 6, 8), (1, 1, 8, 48)),
        ((1, 1, 8, 16), (1, 2, 4, 16)),
        ((1, 64), (64,)),
        ((1, 512), (512,)),
    ]
)

# Pairs where ``Relu -> Reshape`` compiled to the same MCode as ``Relu`` at the
# input shape.
FUSED_AFTER: frozenset[tuple[Shape, Shape]] = frozenset(
    [
        ((1, 8, 4, 4), (1, 1, 8, 16)),
        ((1, 1, 8, 16), (1, 8, 4, 4)),
        ((1, 8), (8,)),
        ((16, 4, 4, 4), (16, 1, 4, 16)),
        ((1, 4, 4, 16), (1, 1, 4, 64)),
        ((1, 1, 8, 16), (1, 8, 2, 8)),
        ((1, 1, 8, 48), (1, 8, 6, 8)),
        ((1, 64), (64,)),
    ]
)

# Pairs measured to emit a real DMA program (not fused), including every
# ResNet18-step convolution/weight Reshape family except the bias flatten.
NOT_FUSED: frozenset[tuple[Shape, Shape]] = frozenset(
    [
        ((8, 8, 3, 3), (1, 8, 8, 9)),
        ((4, 4, 3, 3), (1, 4, 4, 9)),
        ((8, 8, 3, 3), (8, 8, 9)),
        ((1, 8, 8, 9), (8, 8, 3, 3)),
        ((1, 4, 4, 9), (1, 1, 4, 36)),
        ((1, 4, 4, 9), (1, 4, 36)),
        ((16, 64, 56, 56), (16, 1, 64, 3136)),
        ((16, 1, 64, 3136), (16, 64, 56, 56)),
        ((1, 8, 4, 9), (1, 1, 8, 36)),
        ((1, 8, 4, 5), (1, 1, 8, 20)),
        ((1, 8, 7, 7), (1, 1, 8, 49)),
        ((1, 4, 4, 12), (1, 1, 4, 48)),
        ((1, 1, 6, 16), (1, 2, 3, 16)),
        ((1, 1, 8, 12), (1, 8, 4, 3)),
        ((16, 1, 64, 28224), (16, 1, 576, 3136)),
        ((16, 1, 512, 441), (16, 1, 4608, 49)),
        ((16, 1, 256, 1764), (16, 1, 2304, 196)),
        ((16, 1, 128, 7056), (16, 1, 1152, 784)),
        ((16, 1, 128, 784), (16, 128, 28, 28)),
        ((16, 256, 14, 14), (16, 1, 256, 196)),
        ((16, 512, 7, 7), (16, 1, 512, 49)),
        ((64, 64, 3, 3), (1, 64, 64, 9)),
        ((1, 64, 64, 9), (1, 1, 64, 576)),
        ((1, 64, 576), (64, 64, 3, 3)),
        ((512, 512, 3, 3), (1, 512, 512, 9)),
    ]
)


def _dims(shape: Sequence[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def _shape(value: Sequence[int], what: str) -> Shape:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{what} must be a sequence of integers")
    if not value or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in value
    ):
        raise ValueError(f"{what} must be a non-empty sequence of positive integers")
    return tuple(value)


def _numel(shape: Shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _set_dims(value_info, shape: Shape) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for size in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = size


def _get_dims(value_info) -> Shape:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def _load_relu_template(shape: Shape) -> onnx.ModelProto:
    path = os.path.join(_FIXTURES, f"relu_{_dims(shape)}.axmodel.gz")
    if not os.path.exists(path):
        raise ValueError(f"no compiled Relu template for shape {shape}")
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


def _validate_relu_template(model: onnx.ModelProto, shape: Shape) -> onnx.NodeProto:
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("template must contain exactly one compiled 'neu mode' node")
    node = model.graph.node[0]
    if list(node.input) != ["x"] or list(node.output) != ["y"]:
        raise ValueError("template NPU node must map input 'x' to output 'y'")
    if [v.name for v in model.graph.input] != ["x"]:
        raise ValueError("template must have exactly one input named 'x'")
    if [v.name for v in model.graph.output] != ["y"]:
        raise ValueError("template must have exactly one output named 'y'")
    for value in (model.graph.input[0], model.graph.output[0]):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("template tensors must be float32")
        if _get_dims(value) != shape:
            raise ValueError(f"template shape {_get_dims(value)} is not {shape}")
    inits = {item.name: item for item in model.graph.initializer}
    params = inits.get("npu_params")
    if params is None or bytes(params.raw_data) != bytes(40):
        raise ValueError("template must have the 40-byte all-zero npu_params")
    dynamic = inits.get("npu_dyn_params")
    if dynamic is None or dynamic.raw_data or dynamic.dims != [0]:
        raise ValueError("template must have the empty npu_dyn_params initializer")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if json.loads(attrs.get("outputs_info", b"{}")) != {"y": ["FP32", list(shape)]}:
        raise ValueError("template outputs_info disagrees with its output shape")
    return node


def emit_fused_reshape_axmodel(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    output_path: str,
    *,
    position: str = "before",
) -> str:
    """Emit a compiled ``Reshape`` + ``Relu`` model for a measured fused pair.

    ``position="before"`` models ``Relu(Reshape(x))``: the committed ``Relu``
    compiled at ``output_shape`` is reused with its graph input relabelled to
    ``input_shape``. ``position="after"`` models ``Reshape(Relu(x))``: the
    ``Relu`` compiled at ``input_shape`` is reused with its output dims and
    ``outputs_info`` relabelled to ``output_shape``. The MCode and parameter
    tables are kept unchanged. Only pairs in ``FUSED_BEFORE`` / ``FUSED_AFTER``
    are accepted; a pair measured to need real Reshape MCode, or one never
    measured, raises ``ValueError``.
    """
    if position not in ("before", "after"):
        raise ValueError("position must be 'before' or 'after'")
    source = _shape(input_shape, "input_shape")
    target = _shape(output_shape, "output_shape")
    if _numel(source) != _numel(target):
        raise ValueError(f"cannot reshape {source} to {target}: element counts differ")
    pair = (source, target)
    fused = FUSED_BEFORE if position == "before" else FUSED_AFTER
    if pair not in fused:
        if pair in NOT_FUSED:
            raise ValueError(
                f"Reshape {source} -> {target} compiles to a real DMA program "
                "(not fused); its MCode cannot be produced by relabelling"
            )
        raise ValueError(
            f"unmeasured Reshape pair {source} -> {target} for position {position!r}"
        )

    reference_shape = target if position == "before" else source
    model = _load_relu_template(reference_shape)
    node = _validate_relu_template(model, reference_shape)

    if position == "before":
        _set_dims(model.graph.input[0], source)
        for value_info in model.graph.value_info:
            if value_info.name == "x":
                _set_dims(value_info, source)
    else:
        _set_dims(model.graph.output[0], target)
        for value_info in model.graph.value_info:
            if value_info.name == "y":
                _set_dims(value_info, target)
        outputs_info = next(a for a in node.attribute if a.name == "outputs_info")
        outputs_info.s = json.dumps({"y": ["FP32", list(target)]}).encode()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
