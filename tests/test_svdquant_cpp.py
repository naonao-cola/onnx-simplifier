"""Tests for ``onnxsim.apply_svdquant_cpp`` -- the C++-backed port of
SVDQuant's own low-rank-branch-plus-residual-quantization (see
``onnxsim/svdquant_entry.h``). Like ``tests/test_low_rank_compensation_cpp.py``,
this exercises the hand-rolled Jacobi SVD this codebase uses in place of
LAPACK.

``onnxsim.svdquant.apply_svdquant`` (the pure-Python entry point) is now a
thin wrapper around this same C++ function -- see ``onnxsim/svdquant.py`` and
``tests/test_svdquant.py`` (which exercises the identical public contract
through that wrapper). There is no longer a separate pure-Python SVD to
compare against for exact parity -- ``svdquant_entry.h``'s own "ACCEPTED,
PERMANENT DIVERGENCE" note explains why a hand-rolled Jacobi SVD need not
agree sign-for-sign or bit-for-bit with LAPACK's own routine. This file
instead verifies what IS guaranteed regardless of which valid SVD either
side's algorithm lands on: the reconstructed low-rank-branch-plus-residual
weight is close to the float reference, quantizing a low-rank/outlier-heavy
weight this way beats plain round-to-nearest INT4, and documented edge cases
(smoothing on/off, opset gating, rank clamping) behave as specified.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_svdquant_cpp

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=64, N=16, weight=None, seed=0, opset=21):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
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


def _unpack_int4(tensor):
    dims = list(tensor.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(tensor.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    return codes.reshape(dims).astype(np.float64)


def _dequantize_int4_generic(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    init = {t.name: t for t in model.graph.initializer}
    wq = init[dq_node.input[0]]
    ws = init[dq_node.input[1]]
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    codes = _unpack_int4(wq)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def _reconstruct_svdquant_weight(model):
    residual = _dequantize_int4_generic(model)
    init = {t.name: t for t in model.graph.initializer}
    matmul_nodes = [n for n in model.graph.node if n.op_type == "MatMul"]
    l1 = onnx.numpy_helper.to_array(init[matmul_nodes[1].input[1]]).astype(np.float64)
    l2 = onnx.numpy_helper.to_array(init[matmul_nodes[2].input[1]]).astype(np.float64)
    return residual + l1 @ l2


def _outlier_weight(K=64, N=16, seed=0, spike_scale=25.0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((K, N)) * 0.05
    u = rng.standard_normal((K, 2))
    v = rng.standard_normal((2, N))
    spike = spike_scale * (u @ v)
    return (base + spike).astype(np.float32)


def test_cpp_reduces_reconstruction_error_vs_plain_int4_on_outlier_weight():
    K, N = 64, 16
    weight = _outlier_weight(K=K, N=N, seed=0)
    model = _matmul_model(K=K, N=N, weight=weight)
    w64 = weight.astype(np.float64)

    plain = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(plain)
    plain_err = np.linalg.norm(w64 - _dequantize_int4_generic(plain))

    sv = apply_svdquant_cpp(model, rank=4, block_size=32, smooth_alpha=None)
    onnx.checker.check_model(sv)
    sv_err = np.linalg.norm(w64 - _reconstruct_svdquant_weight(sv))

    assert sv_err < plain_err
    assert sv_err < plain_err / 5


def test_cpp_matches_plain_int4_at_rank_zero_equivalent_scale():
    K, N = 64, 16
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)
    w64 = weight.astype(np.float64)

    plain = onnxsim.quantize_weight_only_int4(model)
    plain_err = np.linalg.norm(w64 - _dequantize_int4_generic(plain))

    sv = apply_svdquant_cpp(model, rank=4, block_size=32, smooth_alpha=None)
    sv_err = np.linalg.norm(w64 - _reconstruct_svdquant_weight(sv))

    assert sv_err <= plain_err + 1e-6


def test_cpp_structural_nodes_inserted():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=2)
    sv = apply_svdquant_cpp(model, rank=4, block_size=32, smooth_alpha=None)
    onnx.checker.check_model(sv)

    op_types = [n.op_type for n in sv.graph.node]
    assert op_types.count("MatMul") == 3
    assert "DequantizeLinear" in op_types
    assert "Add" in op_types


def test_cpp_with_smoothing_output_stays_close_to_float_via_onnxruntime():
    K, N = 64, 16
    weight = _outlier_weight(K=K, N=N, seed=3)
    model = _matmul_model(K=K, N=N, weight=weight)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((32, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    sv = apply_svdquant_cpp(
        model,
        rank=8,
        block_size=32,
        smooth_alpha=0.5,
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(sv)

    (float_y,) = _run(model, {"X": x})
    (sv_y,) = _run(sv, {"X": x})
    assert np.all(np.isfinite(sv_y))
    assert _rel_l2(float_y, sv_y) < 0.25


def test_cpp_smooth_alpha_none_skips_smoothing():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=5)
    sv = apply_svdquant_cpp(model, rank=4, block_size=32, smooth_alpha=None)
    onnx.checker.check_model(sv)
    assert all(n.op_type != "Mul" for n in sv.graph.node)


def test_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(6)
    K, N = 64, 12
    weight = _outlier_weight(K=K, N=N, seed=6).T  # store as [N, K] for transB
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    onnx.checker.check_model(model)

    sv = apply_svdquant_cpp(model, rank=4, block_size=32, smooth_alpha=None)
    onnx.checker.check_model(sv)

    rng2 = np.random.default_rng(7)
    x = rng2.standard_normal((8, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (sv_y,) = _run(sv, {"X": x})
    assert np.all(np.isfinite(sv_y))
    assert _rel_l2(float_y, sv_y) < 0.25


def test_cpp_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=8)
    result = apply_svdquant_cpp(model, block_size=32, smooth_alpha=None)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result = apply_svdquant_cpp(model, smooth_alpha=None)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_svdquant_cpp(model, smooth_alpha=None)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = apply_svdquant_cpp(model, smooth_alpha=None)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_rank_clamped_to_min_dimension():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=9)
    sv = apply_svdquant_cpp(model, rank=1000, block_size=32, smooth_alpha=None)
    onnx.checker.check_model(sv)


def test_cpp_missing_graph_input_raises_value_error_when_smoothing():
    model = _matmul_model(K=16, N=4, seed=10)
    with pytest.raises(ValueError):
        apply_svdquant_cpp(
            model,
            smooth_alpha=0.5,
            calibration_data=[{"WrongName": np.zeros((2, 16), dtype=np.float32)}],
        )
