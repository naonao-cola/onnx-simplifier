"""Tests for ``onnxsim.apply_tesseraq_cpp`` -- the C++-backed port of
``onnxsim.apply_tesseraq`` (TesseraQ's Progressive Adaptive Rounding, see
``onnxsim/tesseraq.py``). Unlike every closed-form port in this codebase,
this is an iterative Adam optimization: floating-point summation-order
differences between this port's own scalar dense-matmul kernels and
numpy's own can compound across iterations (see
``onnxsim/tesseraq_entry.h``'s own accepted numerical scope note).
Measured empirically here rather than assumed: most configurations below
come back bit-for-bit identical to the pure-Python reference; longer/
multi-round runs occasionally see a handful of elements land on the
opposite side of PAR's own 0.5 soft-decision boundary (always its
immediate grid neighbor, never further), tolerated with a small, measured
fraction rather than asserted away. The jointly-optimized scale itself
always carries a little float32 rounding noise, checked with a loose
tolerance, not exact equality.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_tesseraq_cpp
from onnxsim.tesseraq import apply_tesseraq

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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _matmul_int4_models(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    float_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    return float_model, quant_model


def _correlated_calibration(K=64, num_samples=32, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _int4_codes(model):
    dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq.input[0])
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int16)
    hi = ((raw >> 4) & 0x0F).astype(np.int16)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int16)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    return codes.reshape([d for d in wq.dims])


def _int4_scale(model):
    dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    ws = next(t for t in model.graph.initializer if t.name == dq.input[1])
    return onnx.numpy_helper.to_array(ws)


def _assert_agrees(
    float_model, quant_model, calibration_data, max_mismatch_frac=0.0, **kwargs
):
    py = apply_tesseraq(float_model, quant_model, calibration_data, **kwargs)
    cpp = apply_tesseraq_cpp(float_model, quant_model, calibration_data, **kwargs)
    py_codes, cpp_codes = _int4_codes(py), _int4_codes(cpp)
    mismatch = int(np.sum(py_codes != cpp_codes))
    # Exact bit-for-bit agreement is the common case (see this module's own
    # docstring), but not guaranteed: enough Adam iterations can let
    # floating-point summation-order noise between this port's own scalar
    # matmul kernels and numpy's compound into a genuinely different
    # rounding decision for a handful of elements sitting almost exactly on
    # PAR's own 0.5 soft-decision boundary. `max_mismatch_frac` is the
    # measured tolerance for that -- always tiny, never assumed away.
    assert mismatch <= max_mismatch_frac * py_codes.size, (
        f"{mismatch}/{py_codes.size} code mismatches exceeds tolerance"
    )
    # A mismatched code still has to be its immediate neighbor (PAR only
    # ever disagrees about which side of 0.5 h landed on, not the far grid
    # point), and the jointly-optimized scale itself always carries a
    # little float32 rounding noise -- checked with a loose tolerance, not
    # exact equality.
    np.testing.assert_array_equal(
        np.abs(py_codes.astype(np.int32) - cpp_codes.astype(np.int32)) <= 1, True
    )
    np.testing.assert_allclose(_int4_scale(py), _int4_scale(cpp), rtol=1e-2, atol=1e-5)
    return cpp


def test_tesseraq_cpp_matches_python_exactly():
    float_model, quant_model = _matmul_int4_models()
    _assert_agrees(float_model, quant_model, _correlated_calibration())


def test_tesseraq_cpp_matches_python_across_shapes_and_rounds():
    for K, N, seed, iters, rounds in [
        (128, 32, 5, 40, 1),
        (64, 16, 11, 100, 3),
        (96, 24, 21, 60, 2),
        (32, 8, 23, 400, 4),
    ]:
        float_model, quant_model = _matmul_int4_models(K=K, N=N, seed=seed)
        calib = _correlated_calibration(K=K, seed=seed + 100)
        # A small, measured tolerance for longer/multi-round runs -- see
        # _assert_agrees's own comment on why a handful of elements can
        # land on the opposite side of PAR's 0.5 boundary.
        _assert_agrees(
            float_model,
            quant_model,
            calib,
            max_mismatch_frac=0.01,
            num_iterations=iters,
            par_rounds=rounds,
        )


def test_tesseraq_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 64, 12
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    float_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    _assert_agrees(float_model, quant_model, _correlated_calibration(K=K, seed=9))


def test_tesseraq_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(10)
    K, N = 64, 12
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    float_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    _assert_agrees(float_model, quant_model, _correlated_calibration(K=K, seed=11))


@pytest.mark.parametrize("num_bits", [2, 3, 4])
def test_tesseraq_cpp_matches_python_across_bit_widths(num_bits):
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=13)
    calib = _correlated_calibration(K=64, seed=14)
    _assert_agrees(
        float_model, quant_model, calib, num_bits=num_bits, num_iterations=60
    )


def test_tesseraq_cpp_matches_python_with_custom_hyperparameters():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=15)
    calib = _correlated_calibration(K=64, seed=16)
    _assert_agrees(
        float_model,
        quant_model,
        calib,
        num_iterations=80,
        par_rounds=3,
        learning_rate=0.2,
        scale_learning_rate=0.02,
        reg_param=0.02,
        warm_start=0.1,
        beta_range=(10.0, 1.5),
    )


def test_tesseraq_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_tesseraq_cpp(
        model, quant, calibration_data=[{"X": np.zeros((1, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == quant.SerializeToString()


def test_tesseraq_cpp_rejects_invalid_num_bits():
    float_model, quant_model = _matmul_int4_models()
    calib = _correlated_calibration()
    with pytest.raises(Exception):
        apply_tesseraq_cpp(float_model, quant_model, calib, num_bits=5)
    with pytest.raises(Exception):
        apply_tesseraq_cpp(float_model, quant_model, calib, num_bits=1)
