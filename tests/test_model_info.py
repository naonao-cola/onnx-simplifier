"""Tests for MACs / FLOPs counting in ``onnxsim.model_info``.

Every model is built directly with ``onnx.helper`` (no torch dependency). MAC
counts are asserted against hand-computed values. The symbolic (sympy) path is
exercised when sympy is installed and skipped otherwise, mirroring the optional
``onnxsim[symbolic]`` extra.
"""
import numpy as np
from onnx import TensorProto, helper
import pytest

from onnxsim import model_info
from onnxsim.model_info import ModelInfo, human_readable_num, human_readable_size


def _model(nodes, inputs, outputs, initializers=None, opset=23):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model


def _vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _macs(nodes, inputs, outputs, initializers=None, opset=23):
    return ModelInfo(_model(nodes, inputs, outputs, initializers, opset)).macs


def _weight(name, shape):
    return helper.make_tensor(
        name, TensorProto.FLOAT, shape, np.zeros(shape, np.float32).flatten()
    )


# --------------------------------------------------------------------------- #
# Core compute operators
# --------------------------------------------------------------------------- #
def test_conv_macs():
    # output 1*4*8*8, cin/group 3, kernel 3*3 -> 256 * 3 * 9
    x = _vi("x", [1, 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", [1, 4, 8, 8])
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    assert _macs([node], [x], [y], [w]) == 1 * 4 * 8 * 8 * 3 * (3 * 3)


def test_conv_grouped_macs():
    # depthwise: groups=4, weight [4, 1, 3, 3]; cin/group = 1
    x = _vi("x", [1, 4, 8, 8])
    w = _weight("w", [4, 1, 3, 3])
    y = _vi("y", [1, 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=4
    )
    assert _macs([node], [x], [y], [w]) == 1 * 4 * 8 * 8 * 1 * (3 * 3)


def test_conv_transpose_macs():
    # input 1*3*8*8, out_channels/group 4, kernel 3*3
    x = _vi("x", [1, 3, 8, 8])
    w = _weight("w", [3, 4, 3, 3])  # [in, out/group, kH, kW]
    y = _vi("y", [1, 4, 10, 10])
    node = helper.make_node("ConvTranspose", ["x", "w"], ["y"], kernel_shape=[3, 3])
    assert _macs([node], [x], [y], [w]) == 1 * 3 * 8 * 8 * 4 * (3 * 3)


def test_gemm_macs():
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert _macs([node], [a, b], [y]) == 5 * 3 * 7


def test_gemm_transposed_macs():
    # transB=1: b is [N, K] = [3, 7]; M=5, N=3, K=7
    a = _vi("a", [5, 7])
    b = _vi("b", [3, 7])
    y = _vi("y", [5, 3])
    node = helper.make_node("Gemm", ["a", "b"], ["y"], transB=1)
    assert _macs([node], [a, b], [y]) == 5 * 3 * 7


def test_matmul_batched_macs():
    # A [2, 5, 7], B [7, 3] -> Y [2, 5, 3]; K=7
    a = _vi("a", [2, 5, 7])
    b = _weight("b", [7, 3])
    y = _vi("y", [2, 5, 3])
    node = helper.make_node("MatMul", ["a", "b"], ["y"])
    assert _macs([node], [a], [y], [b]) == 2 * 5 * 3 * 7


# --------------------------------------------------------------------------- #
# Attention (ai.onnx opset 23+)
# --------------------------------------------------------------------------- #
def _attention_4d(hq, hkv):
    b, sq, skv, d, dv = 2, 16, 16, 64, 64
    q = _vi("q", [b, hq, sq, d])
    k = _vi("k", [b, hkv, skv, d])
    v = _vi("v", [b, hkv, skv, dv])
    y = _vi("y", [b, hq, sq, dv])
    node = helper.make_node("Attention", ["q", "k", "v"], ["y"])
    expected = b * hq * sq * skv * d + b * hq * sq * skv * dv
    return _macs([node], [q, k, v], [y]), expected


def test_attention_4d_mha():
    got, expected = _attention_4d(hq=8, hkv=8)
    assert got == expected


def test_attention_4d_gqa_uses_query_heads():
    # kv heads < q heads, but all q heads are evaluated.
    got, expected = _attention_4d(hq=8, hkv=2)
    assert got == expected


def test_attention_3d_uses_head_attrs():
    b, sq, skv, hidden, heads = 2, 16, 16, 512, 8
    q = _vi("q", [b, sq, hidden])
    k = _vi("k", [b, skv, hidden])
    v = _vi("v", [b, skv, hidden])
    y = _vi("y", [b, sq, hidden])
    node = helper.make_node(
        "Attention", ["q", "k", "v"], ["y"], q_num_heads=heads, kv_num_heads=heads
    )
    d = hidden // heads
    expected = b * heads * sq * skv * d + b * heads * sq * skv * d
    assert _macs([node], [q, k, v], [y]) == expected


def test_attention_3d_without_head_attrs_is_zero():
    # Head split is unknowable without q_num_heads / kv_num_heads.
    b, sq, hidden = 2, 16, 512
    q = _vi("q", [b, sq, hidden])
    k = _vi("k", [b, sq, hidden])
    v = _vi("v", [b, sq, hidden])
    y = _vi("y", [b, sq, hidden])
    node = helper.make_node("Attention", ["q", "k", "v"], ["y"])
    assert _macs([node], [q, k, v], [y]) == 0


# --------------------------------------------------------------------------- #
# Quantized twins reuse the float formulas at the right operand indices
# --------------------------------------------------------------------------- #
def test_matmul_integer_macs():
    a = _vi("a", [4, 8], TensorProto.UINT8)
    b = _vi("b", [8, 16], TensorProto.UINT8)
    y = _vi("y", [4, 16], TensorProto.INT32)
    node = helper.make_node("MatMulInteger", ["a", "b"], ["y"])
    assert _macs([node], [a, b], [y], opset=10) == 4 * 16 * 8


def test_qlinearconv_weight_at_input3():
    # QLinearConv packs weight at input[3]: x, x_s, x_z, w, w_s, w_z, y_s, y_z
    x = _vi("x", [1, 3, 8, 8], TensorProto.UINT8)
    w = _vi("w", [4, 3, 3, 3], TensorProto.UINT8)
    y = _vi("y", [1, 4, 8, 8], TensorProto.UINT8)
    scalars = [
        _vi("x_s", []), _vi("x_z", [], TensorProto.UINT8),
        _vi("w_s", [4]), _vi("w_z", [4], TensorProto.UINT8),
        _vi("y_s", []), _vi("y_z", [], TensorProto.UINT8),
    ]
    node = helper.make_node(
        "QLinearConv",
        ["x", "x_s", "x_z", "w", "w_s", "w_z", "y_s", "y_z"],
        ["y"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    assert _macs([node], [x] + scalars + [w], [y], opset=10) == 1 * 4 * 8 * 8 * 3 * 9


# --------------------------------------------------------------------------- #
# Unknown / dynamic shapes
# --------------------------------------------------------------------------- #
def test_unnamed_dynamic_dim_counts_per_sample():
    # A batch dim with neither a value nor a name: ONNX shape inference assigns
    # it a generated symbol (e.g. "unk__0"), so the count is linear in that axis
    # and collapses to the per-sample MACs when the axis is set to 1.
    x = _vi("x", [None, 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", [None, 4, 8, 8])
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    macs = _macs([node], [x], [y], [w])
    assert model_info._representative_number(macs) == 1 * 4 * 8 * 8 * 3 * 9


def test_uninferrable_node_contributes_zero():
    # When a required operand has no shape at all (empty shape map), the counter
    # returns 0 rather than guessing.
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert model_info._gemm_macs(node, {}) == 0


def test_flops_is_twice_macs():
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    info = ModelInfo(_model([helper.make_node("Gemm", ["a", "b"], ["y"])], [a, b], [y]))
    assert info.flops == 2 * info.macs


# --------------------------------------------------------------------------- #
# Symbolic (sympy) path
# --------------------------------------------------------------------------- #
def test_dynamic_dim_symbolic():
    sympy = pytest.importorskip("sympy")
    x = _vi("x", ["batch", 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", ["batch", 4, 8, 8])
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    macs = ModelInfo(_model([node], [x], [y], [w])).macs
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert sympy.simplify(macs - 1 * 4 * 8 * 8 * 3 * 9 * batch) == 0


def test_symbolic_dims_unify_across_tensors():
    # A shared dim_param name ("seq") must collapse to one symbol so the two
    # attention matmuls combine into a single seq**2 term.
    sympy = pytest.importorskip("sympy")
    b, hq, d = 2, 8, 64
    q = _vi("q", [b, hq, "seq", d])
    k = _vi("k", [b, hq, "seq", d])
    v = _vi("v", [b, hq, "seq", d])
    y = _vi("y", [b, hq, "seq", d])
    node = helper.make_node("Attention", ["q", "k", "v"], ["y"])
    macs = ModelInfo(_model([node], [q, k, v], [y])).macs
    seq = sympy.Symbol("seq", positive=True, integer=True)
    assert sympy.simplify(macs - 2 * b * hq * d * seq ** 2) == 0


def test_symbolic_human_readable_num():
    sympy = pytest.importorskip("sympy")
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert human_readable_num(9472 * batch) == "9472*batch"


def test_print_simplifying_info_symbolic_does_not_raise():
    pytest.importorskip("sympy")
    x = _vi("x", ["batch", 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", ["batch", 4, 8, 8])
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    model = _model([node], [x], [y], [w])
    model_info.print_simplifying_info(model, model)  # must not raise on symbolic "<"


def test_dynamic_dim_without_sympy_assumes_one(monkeypatch):
    # With sympy unavailable, dynamic dims are assumed 1 (per-sample MACs).
    monkeypatch.setattr(model_info, "sympy", None)
    x = _vi("x", ["batch", 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", ["batch", 4, 8, 8])
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    assert ModelInfo(_model([node], [x], [y], [w])).macs == 1 * 4 * 8 * 8 * 3 * 9


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def test_human_readable_num_units():
    assert human_readable_num(0) == "0.0"
    assert human_readable_num(9472) == "9.5K"
    assert human_readable_num(3_000_000) == "3.0M"


def test_human_readable_size_units():
    assert human_readable_size(512) == "512.0B"
    assert human_readable_size(1024) == "1.0KiB"


# --------------------------------------------------------------------------- #
# Warnings replace silent failures
# --------------------------------------------------------------------------- #
def test_warns_when_shape_inference_fails(monkeypatch):
    def boom(_model):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_info.shape_inference, "infer_shapes", boom)
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    with pytest.warns(UserWarning, match="Shape inference failed"):
        ModelInfo(_model([node], [x], [y]))


def test_warns_when_counter_raises(monkeypatch):
    def boom(node, shapes):
        raise ValueError("bad counter")

    monkeypatch.setitem(model_info._MAC_COUNTERS, "Relu", boom)
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"], name="r")
    with pytest.warns(UserWarning, match="Failed to count MACs"):
        ModelInfo(_model([node], [x], [y]))
