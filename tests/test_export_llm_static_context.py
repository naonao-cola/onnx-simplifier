"""Unit tests for scripts/apple/export_llm_to_coreml.py's static-context path.

`_pin_sequence_dims` is plain ONNX shape editing with only an `onnx`
dependency, so it's tested directly here. A MIL-level check (the pinned model
builds with no `dynamic_shapes` and carries no symbolic dims) runs too, gated
on coremltools like the rest of the Core ML export tests.
"""

import os
import sys

import onnx
import pytest
from onnx import helper

_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)

from export_llm_to_coreml import _pin_sequence_dims  # noqa: E402


def _decoder_model() -> onnx.ModelProto:
    # Built with helper, not onnx.parser: the text format cannot express the
    # composite 'past_sequence_length + sequence_length' dim_param.
    x = helper.make_tensor_value_info(
        "x", onnx.TensorProto.FLOAT, [1, 3, "sequence_length", 8]
    )
    past = helper.make_tensor_value_info(
        "past", onnx.TensorProto.FLOAT, [1, 3, "past_sequence_length", 8]
    )
    logits = helper.make_tensor_value_info(
        "logits",
        onnx.TensorProto.FLOAT,
        [1, 3, "past_sequence_length + sequence_length", 8],
    )
    node = helper.make_node("Concat", ["x", "past"], ["logits"], axis=2)
    graph = helper.make_graph([node], "g", [x, past], [logits])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _shapes(model: onnx.ModelProto):
    out = {}
    for vi in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
        out[vi.name] = dims
    return out


def test_pin_sequence_dims_concretizes_all_three():
    model = _decoder_model()
    assert _shapes(model)["past"] == [1, 3, "past_sequence_length", 8]
    _pin_sequence_dims(model, sequence_length=1, past_sequence_length=511)
    assert _shapes(model) == {
        "x": [1, 3, 1, 8],
        "past": [1, 3, 511, 8],
        "logits": [1, 3, 512, 8],
    }


ct = pytest.importorskip("coremltools", reason="coremltools is not installed")


def test_pinned_model_builds_mil_with_static_shapes():
    from onnxsim import coreml_export

    model = _decoder_model()
    _pin_sequence_dims(model, sequence_length=1, past_sequence_length=511)
    mb, types, Function, Program, RangeDim, TensorType = coreml_export._import_mil()
    prog, _ = coreml_export._build_mil_program(
        model, mb, types, Function, Program, RangeDim, TensorType
    )
    out = prog.functions["main"].outputs[0]
    assert tuple(out.shape) == (1, 3, 512, 8)
    seen = [out]
    for op in prog.functions["main"].operations:
        for v in op.inputs.values():
            seen.extend(v if isinstance(v, (list, tuple)) else [v])
        seen.extend(op.outputs)
    assert seen
    for v in seen:
        for d in v.shape:
            assert isinstance(d, int), v.shape
