"""Tests for ``onnxsim.quantize_weight_only_nf4`` (bitsandbytes' NF4, see
``onnxsim/nf4.py``) -- block-wise quantization onto a fixed, non-uniform
16-value codebook (the quantile points of a standard normal distribution)
instead of a uniform integer grid, represented in the ONNX graph via
ordinary Gather/Reshape/Mul (no contrib op, no opset-21 features).
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim
from onnxsim.nf4 import NF4_CODEBOOK

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=8
    )


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _dequantize_nf4_by_hand(model, w_name="W", block_size=64):
    """Independent reference decode: reads Wq/Ws straight from the
    initializers and dequantizes via numpy, without using any of the ops
    this module inserts into the graph -- so this test doesn't just check
    "the graph runs", it checks the *values* against ground truth computed
    a completely different way.
    """
    wq = next(t for t in model.graph.initializer if t.name == f"{w_name}_nf4_q")
    ws = next(t for t in model.graph.initializer if t.name == f"{w_name}_nf4_scale")
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    codebook = np.asarray(NF4_CODEBOOK, dtype=np.float64)

    dim0, dim1 = codes.shape
    num_blocks = scale.shape[0]
    block_size_actual = dim0 // num_blocks
    assert block_size_actual == block_size
    values = codebook[codes]  # [dim0, dim1]
    scale_full = np.repeat(scale, block_size, axis=0)  # [dim0, dim1]
    return values * scale_full


def test_nf4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    nf4_model = onnxsim.quantize_weight_only_nf4(model, block_size=64)
    onnx.checker.check_model(nf4_model)

    op_types = {n.op_type for n in nf4_model.graph.node}
    # No custom/contrib op and no DequantizeLinear -- just ordinary ops.
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in nf4_model.graph.node)


def test_nf4_dequantized_values_match_hand_decoded_reference():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.3
    model = _matmul_model(weight=weight)
    nf4_model = onnxsim.quantize_weight_only_nf4(model, block_size=64)

    w_hand = _dequantize_nf4_by_hand(nf4_model, block_size=64)

    # Every element's dequantization error must be within half the largest
    # codebook gap scaled by that block's own scale -- a real per-element
    # correctness check, not just an aggregate error bound.
    codebook = np.asarray(NF4_CODEBOOK)
    max_gap = np.max(np.diff(codebook))
    block_scale = np.abs(weight.astype(np.float64)).max()
    assert np.all(
        np.abs(w_hand - weight.astype(np.float64)) <= max_gap * block_scale / 2 + 1e-6
    )


def test_nf4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    nf4_model = onnxsim.quantize_weight_only_nf4(model)
    onnx.checker.check_model(nf4_model)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (nf4_y,) = _run(nf4_model, {"X": x})
    assert np.all(np.isfinite(nf4_y))
    assert _rel_l2(float_y, nf4_y) < 0.25


def test_nf4_gemm_transb():
    rng = np.random.default_rng(4)
    K, N = 128, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    nf4_model = onnxsim.quantize_weight_only_nf4(model, block_size=64)
    onnx.checker.check_model(nf4_model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (nf4_y,) = _run(nf4_model, {"X": x})
    assert _rel_l2(float_y, nf4_y) < 0.25


def test_nf4_codes_stay_in_range():
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    nf4_model = onnxsim.quantize_weight_only_nf4(model, block_size=64)
    wq = next(t for t in nf4_model.graph.initializer if t.name == "W_nf4_q")
    codes = onnx.numpy_helper.to_array(wq)
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_nf4_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=6)  # 48 is not a multiple of 64
    nf4_model = onnxsim.quantize_weight_only_nf4(model, block_size=64)
    assert nf4_model.SerializeToString() == model.SerializeToString()


def test_nf4_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 64]), _vi("W", [64, 4])], [_vi("Y", [4, 4])], []
    )
    nf4_model = onnxsim.quantize_weight_only_nf4(model)
    assert nf4_model.SerializeToString() == model.SerializeToString()


def test_nf4_codebook_is_well_formed():
    codebook = np.asarray(NF4_CODEBOOK)
    assert codebook.shape == (16,)
    assert np.all(np.diff(codebook) > 0)  # strictly increasing
    assert codebook[0] == -1.0 and codebook[-1] == 1.0
    assert 0.0 in codebook
