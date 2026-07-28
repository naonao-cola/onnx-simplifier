"""End-to-end regression tests against real NVIDIA ModelOpt output.

``tests/test_simple.py`` already covers the fp8 QDQ handling with *synthetic*
graphs that hand-build ``float8_e4m3fn`` zero points to mimic what NVIDIA
ModelOpt emits (GitHub issue #348). Those guard the code path but not the
assumption behind it: that ModelOpt actually produces graphs of that shape.

This module closes that gap by driving the real
``modelopt.onnx.quantization.quantize`` API -- the same call TensorRT users run
to produce int8/fp8 QDQ ONNX models -- and then feeding its output straight
through ``onnxsim.simplify``. If a future ModelOpt release changes its quantized
graph in a way onnxsim can no longer simplify (a new unhashable element type, a
QDQ layout the optimizers fold away, ...), these tests fail where the synthetic
ones would keep passing.

ModelOpt (and its torch / onnxruntime dependencies) is heavy and not part of
onnxsim's test requirements, so the whole module is skipped when it is not
installed. The dedicated ``modelopt-integration`` CI workflow installs it and
runs these tests; the regular build-and-test matrix skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

# Skip the entire module unless NVIDIA ModelOpt's ONNX quantization is available.
# Checked before importing onnxsim so a ModelOpt-less run skips cleanly.
moq = pytest.importorskip(
    "modelopt.onnx.quantization",
    reason="nvidia-modelopt[onnx] is not installed",
)

import onnxsim  # noqa: E402  (imported after the ModelOpt availability check)


def _build_fp32_model(path: str) -> None:
    """Write a small Conv + MatMul fp32 model that ModelOpt will quantize.

    ModelOpt only inserts QDQ around layers big enough to benefit from
    low-precision TensorRT kernels: Conv needs input/output channels > 16 and a
    filter size <= 32, MatMul/Gemm needs its contracted/output dims >= 16 (see
    ``find_nodes_from_convs_to_exclude`` / the ``_MIN_MATMUL_DIM`` guard in
    modelopt.onnx.quantization.graph_utils). Undersized layers are silently left
    in fp16 and produce no QDQ, which would make these tests vacuous, so the
    32-channel Conv and 32x32 MatMul below are deliberately above those floors.
    """
    rng = np.random.RandomState(0)
    conv_w = helper.make_tensor(
        "conv_w",
        TensorProto.FLOAT,
        [32, 32, 3, 3],
        rng.randn(32 * 32 * 3 * 3).astype(np.float32).tolist(),
    )
    mm_w = helper.make_tensor(
        "mm_w",
        TensorProto.FLOAT,
        [32, 32],
        rng.randn(32 * 32).astype(np.float32).tolist(),
    )
    nodes = [
        helper.make_node(
            "Conv", ["X", "conv_w"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("GlobalAveragePool", ["r"], ["p"]),
        helper.make_node("Flatten", ["p"], ["f"], axis=1),
        helper.make_node("MatMul", ["f", "mm_w"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "modelopt_src",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 32])],
        [conv_w, mm_w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _quantize_with_modelopt(tmp_path, quantize_mode: str) -> onnx.ModelProto:
    """Quantize the fp32 model with real ModelOpt and return the QDQ model.

    Calibration is pinned to the CPU EP so the test runs on CPU-only CI.
    """
    src = str(tmp_path / "fp32.onnx")
    out = str(tmp_path / f"quant_{quantize_mode}.onnx")
    _build_fp32_model(src)
    moq.quantize(
        onnx_path=src,
        quantize_mode=quantize_mode,
        calibration_data={
            "X": np.random.RandomState(1).randn(4, 32, 8, 8).astype(np.float32)
        },
        calibration_eps=["cpu"],
        output_path=out,
    )
    return onnx.load(out)


def _qdq_counts(model: onnx.ModelProto):
    op_types = [n.op_type for n in model.graph.node]
    return op_types.count("QuantizeLinear"), op_types.count("DequantizeLinear")


def _has_float8_tensor(model: onnx.ModelProto) -> bool:
    """Whether any initializer/attribute tensor uses the float8 element type."""
    if any(
        init.data_type == TensorProto.FLOAT8E4M3FN for init in model.graph.initializer
    ):
        return True
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.HasField("t") and attr.t.data_type == TensorProto.FLOAT8E4M3FN:
                return True
            if any(t.data_type == TensorProto.FLOAT8E4M3FN for t in attr.tensors):
                return True
    return False


def test_modelopt_int8_qdq_simplifies(tmp_path):
    # int8 zero points are a hashable element type, so this exercises the
    # ordinary path: onnxsim must simplify a real ModelOpt int8 QDQ graph, keep
    # ``check_ok`` True (ModelOpt wrapped simplify in a try/except that dropped
    # the simplified model on error -- issue #348), and preserve every QDQ pair
    # that TensorRT relies on.
    model = _quantize_with_modelopt(tmp_path, "int8")
    n_q, n_dq = _qdq_counts(model)
    assert n_q > 0 and n_dq > 0, "ModelOpt produced no int8 QDQ nodes to test"

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # onnxsim must neither fold nor dedupe the QDQ structure away.
    assert _qdq_counts(sim_model) == (n_q, n_dq)


def test_modelopt_fp8_qdq_simplifies(tmp_path):
    # The regression that motivated this: ModelOpt's fp8 mode emits
    # ``float8_e4m3fn`` (element type 17) zero points. onnxoptimizer's
    # tensor-value hashing passes cannot hash that type and used to abort the
    # whole simplification with "no supported data type: 17" (issue #348).
    # onnxsim now detects such tensors and transparently skips those passes.
    model = _quantize_with_modelopt(tmp_path, "fp8")
    n_q, n_dq = _qdq_counts(model)
    assert n_q > 0 and n_dq > 0, "ModelOpt produced no fp8 QDQ nodes to test"
    # Guard that this test really exercises the unhashable-tensor path: if a
    # future ModelOpt stops emitting float8 zero points here, we want to know.
    assert _has_float8_tensor(model), "expected float8_e4m3fn zero points"

    # Must not raise "no supported data type: 17".
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # The fp8 QDQ structure TensorRT consumes must survive simplification.
    assert _qdq_counts(sim_model) == (n_q, n_dq)
    assert _has_float8_tensor(sim_model)
