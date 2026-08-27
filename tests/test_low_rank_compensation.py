"""Tests for ``onnxsim.apply_low_rank_compensation`` (LoRC, from
ZeroQuant-V2, see ``onnxsim/low_rank_compensation.py``) -- adds a
truncated-SVD rank-``r`` correction of a quantized layer's own existing
reconstruction error (``float_weight - dequantized_weight``), computed
directly from the two weights with no calibration data.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _matmul_model(K=64, N=16, seed=0):
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


def test_lorc_reduces_reconstruction_error():
    model = _matmul_model(K=64, N=16, seed=0)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 64)).astype(np.float32)

    lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=8)
    onnx.checker.check_model(lorc_model)

    # A correction was actually inserted -- otherwise this is just
    # re-testing plain RTN.
    op_types = [n.op_type for n in lorc_model.graph.node]
    assert op_types.count("MatMul") == 3  # base + the two correction matmuls

    (float_y,) = _run(model, {"X": x})
    (rtn_y,) = _run(quant, {"X": x})
    (lorc_y,) = _run(lorc_model, {"X": x})
    rtn_err = np.linalg.norm(float_y.astype(np.float64) - rtn_y.astype(np.float64))
    lorc_err = np.linalg.norm(float_y.astype(np.float64) - lorc_y.astype(np.float64))
    assert lorc_err < rtn_err


def test_lorc_higher_rank_never_increases_error():
    # Eckart-Young: a larger truncation rank can only capture more of the
    # residual's own energy, never less -- so reconstruction error must be
    # monotonically non-increasing as rank grows.
    model = _matmul_model(K=64, N=16, seed=2)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})

    errors = []
    for rank in (1, 4, 8, 16):
        lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=rank)
        (y,) = _run(lorc_model, {"X": x})
        errors.append(np.linalg.norm(float_y.astype(np.float64) - y.astype(np.float64)))

    assert all(errors[i] >= errors[i + 1] - 1e-6 for i in range(len(errors) - 1))


def test_lorc_exact_at_full_rank():
    # rank = min(K, N) recovers the residual exactly, so the compensated
    # layer should reproduce the float model almost exactly (up to
    # fp32/SVD numerical precision). Requires
    # workaround_ort_matmul_nbits_axis0_bug: onnxruntime 1.29.0's default
    # graph optimization level otherwise silently miscomputes axis=0
    # block-quantized DequantizeLinear feeding a plain (non-transposed)
    # MatMul -- a bug in quantize_weight_only_int4's own output, not in
    # apply_low_rank_compensation, but one this exact-reconstruction
    # property is precise enough to notice (see
    # onnxsim/ort_matmul_nbits_workaround.py's own docstring).
    K, N = 32, 16
    model = _matmul_model(K=K, N=N, seed=4)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, K)).astype(np.float32)

    lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=min(K, N))
    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(lorc_model)
    (float_y,) = _run(model, {"X": x})
    (lorc_y,) = _run(fixed, {"X": x})
    assert _rel_l2(float_y, lorc_y) < 1e-4


def test_lorc_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=6)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 64)).astype(np.float32)

    lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=8)
    onnx.checker.check_model(lorc_model)

    (float_y,) = _run(model, {"X": x})
    (lorc_y,) = _run(lorc_model, {"X": x})
    assert np.all(np.isfinite(lorc_y))
    assert _rel_l2(float_y, lorc_y) < 0.25


def test_lorc_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 48, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = rng.standard_normal((8, K)).astype(np.float32)
    lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=6)
    onnx.checker.check_model(lorc_model)

    (float_y,) = _run(model, {"X": x})
    (lorc_y,) = _run(lorc_model, {"X": x})
    assert _rel_l2(float_y, lorc_y) < 0.25


def test_lorc_noop_when_no_int4_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_low_rank_compensation(model, model, rank=4)
    assert result.SerializeToString() == model.SerializeToString()
