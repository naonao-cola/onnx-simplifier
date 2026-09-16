"""Static checks for `scripts/edgeai/quantize_for_tidl.py`'s QDQ output.

Verifies that `onnxsim.calibration.quantize_static`/`quantize_static_int16`
structurally matches `docs/quantization.md`'s per-layer quantization
scheme (symmetric per-channel weights, asymmetric per-tensor activations)
and that the QOperator-format blocker `tidl_ops.py` gained fires on the
op types edgeai-tidl-tools' own `docs/operators.md` has no entry for.

Needs no vendor package or device -- correctness of the *real* compiler's
handling of this output (including a confirmed real crash, see
`scripts/edgeai/quantize_for_tidl.py`'s docstring) is
`tests/test_edgeai_tidl_real_compile.py`'s job, not this file's.
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_EDGEAI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edgeai"
)
_AXERA_DIR = os.path.join(os.path.dirname(_EDGEAI_DIR), "axera")
for _dir in (_EDGEAI_DIR, _AXERA_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import quantize_for_tidl as qft  # noqa: E402
import tidl_backend as tidl  # noqa: E402
import tidl_ops  # noqa: E402
from _local_import import fresh  # noqa: E402

models = fresh("models", _EDGEAI_DIR)


def _conv_relu_float():
    w = numpy_helper.from_array(
        np.random.RandomState(1).randn(8, 3, 3, 3).astype(np.float32), "w"
    )
    b = numpy_helper.from_array(np.zeros(8, np.float32), "b")
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "conv_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 16, 16])],
        [w, b],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.checker.check_model(model)
    return model


def test_quantize_for_tidl_int8_matches_documented_scheme():
    model = _conv_relu_float()
    quantized = qft.quantize_for_tidl(model, precision="int8")
    onnx.checker.check_model(quantized)

    assert qft.check_tidl_qdq_scheme(quantized) == []
    op_types = {n.op_type for n in quantized.graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= op_types
    assert tidl.coverage(quantized) == "full"


def test_quantize_for_tidl_int8_weight_is_int8_symmetric_per_channel():
    """The specific values, not just "no violations" -- `check_tidl_qdq_scheme`
    could in principle pass on an empty model."""
    model = _conv_relu_float()
    quantized = qft.quantize_for_tidl(model, precision="int8")

    weight_dq = next(
        n
        for n in quantized.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in {i.name for i in quantized.graph.initializer}
    )
    scale = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == weight_dq.input[1])
    )
    zero_point = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == weight_dq.input[2])
    )
    assert zero_point.dtype == np.int8
    assert np.all(zero_point == 0)
    assert scale.shape == (8,)  # one scale per output channel


def test_quantize_for_tidl_int8_a16_widens_activation_to_uint16():
    model = _conv_relu_float()
    quantized = qft.quantize_for_tidl(model, precision="int8_a16")
    onnx.checker.check_model(quantized)
    assert qft.check_tidl_qdq_scheme(quantized) == []

    act_quantize = next(
        n for n in quantized.graph.node if n.op_type == "QuantizeLinear"
    )
    zero_point = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == act_quantize.input[2])
    )
    assert zero_point.dtype == np.uint16


def test_quantize_for_tidl_rejects_unknown_precision():
    model = _conv_relu_float()
    try:
        qft.quantize_for_tidl(model, precision="int4")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unsupported precision")


def test_check_tidl_qdq_scheme_flags_asymmetric_weight_and_per_channel_activation():
    """A deliberately-wrong QDQ graph -- asymmetric weight zero_point and a
    per-channel (not per-tensor) activation scale -- must both be flagged."""
    w = numpy_helper.from_array(np.zeros((4, 3, 3, 3), np.int8), "w")
    w_scale = numpy_helper.from_array(np.array([0.1, 0.2, 0.3, 0.4], np.float32), "ws")
    w_zp = numpy_helper.from_array(np.array([1, 0, 0, 0], np.int8), "wz")
    x_scale = numpy_helper.from_array(np.array([0.1, 0.2], np.float32), "xs")
    x_zp = numpy_helper.from_array(np.array([0, 0], np.uint8), "xz")

    nodes = [
        helper.make_node("DequantizeLinear", ["x", "xs", "xz"], ["xf"]),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"], axis=0),
        helper.make_node("Conv", ["xf", "wf"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "bad_qdq",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [w, w_scale, w_zp, x_scale, x_zp],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.checker.check_model(model)

    issues = qft.check_tidl_qdq_scheme(model)
    assert len(issues) == 2
    assert any("symmetric" in issue for issue in issues)
    assert any("per-tensor" in issue for issue in issues)


def test_qoperator_format_ops_are_blocked():
    """QLinearConv (and friends) have no entry in docs/operators.md --
    only plain QuantizeLinear/DequantizeLinear (QDQ format) do."""
    node = helper.make_node(
        "QLinearConv",
        ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"],
        ["y"],
    )
    graph = helper.make_graph(
        [node],
        "qlinearconv_leaf",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 1, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    assert tidl.coverage(model) == "partial"
    assert tidl_ops.blocking_op_types(model) == {"QLinearConv"}


def test_no_blockers_in_clean_synthetic_models_still_holds():
    """Adding QOPERATOR_OPS must not spuriously flag the existing suite."""
    for name in models.names():
        if name == "edgeai_dynamic_batch_leaf":
            continue
        model = models.build(name)
        assert tidl.coverage(model) == "full", (name, tidl.blockers(model))
