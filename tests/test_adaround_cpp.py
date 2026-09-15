"""Tests for ``onnxsim.apply_adaround_cpp`` -- the C++-backed port of
``onnxsim.apply_adaround`` (AIMET's Adaptive Rounding, see
``onnxsim/adaround.py``). Like ``test_tesseraq_cpp.py``, this is an
iterative Adam optimization, not a closed-form computation: floating-point
summation-order differences between this port's own scalar dense-matmul
kernels and numpy's own can compound across iterations (see
``onnxsim/adaround_entry.h``'s own accepted numerical scope note).
Measured empirically here rather than assumed: most configurations below
come back bit-for-bit identical to the pure-Python reference (AdaRound
only ever hardens once, at the very end, unlike TesseraQ's own progressive
coarse-to-fine hardening that locks in a decision mid-optimization, so
this port agrees exactly far more often in practice) -- but not always;
some configurations see a handful of elements land on the opposite side of
the rectified sigmoid's own 0.5 soft-decision boundary (always its
immediate grid neighbor, never further), tolerated with a small, measured
fraction rather than asserted away. AdaRound never touches the scale, so
(unlike TesseraQ) there's nothing else to check once the codes agree.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.adaround import apply_adaround
from onnxsim.onnx_simplifier import apply_adaround_cpp

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


def _assert_agrees(
    float_model, quant_model, calibration_data, max_mismatch_frac=0.0, **kwargs
):
    py = apply_adaround(float_model, quant_model, calibration_data, **kwargs)
    cpp = apply_adaround_cpp(float_model, quant_model, calibration_data, **kwargs)
    onnx.checker.check_model(cpp)
    py_codes, cpp_codes = _int4_codes(py), _int4_codes(cpp)
    mismatch = int(np.sum(py_codes != cpp_codes))
    # Exact bit-for-bit agreement is the common case (see this module's
    # own docstring), but not guaranteed -- `max_mismatch_frac` is the
    # measured tolerance for the rare cases it isn't, always tiny and
    # never assumed away (see test_tesseraq_cpp.py's own identical
    # pattern and comment for why this class of port can disagree at
    # all).
    assert mismatch <= max_mismatch_frac * py_codes.size, (
        f"{mismatch}/{py_codes.size} code mismatches exceeds tolerance"
    )
    # A mismatched code still has to be its immediate neighbor (only ever
    # disagrees about which side of 0.5 the relaxation landed on, not the
    # far grid point).
    np.testing.assert_array_equal(
        np.abs(py_codes.astype(np.int32) - cpp_codes.astype(np.int32)) <= 1, True
    )
    if mismatch == 0:
        # AdaRound never touches the scale (or anything besides Wq): when
        # the codes agree exactly, so should the whole model, byte for
        # byte.
        py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
        cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
        assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
        for a, b in zip(py_inits, cpp_inits):
            assert a.data_type == b.data_type, a.name
            assert list(a.dims) == list(b.dims), a.name
            assert a.raw_data == b.raw_data, a.name
    return cpp


def test_adaround_cpp_matches_python_exactly():
    float_model, quant_model = _matmul_int4_models()
    _assert_agrees(float_model, quant_model, _correlated_calibration())


def test_adaround_cpp_matches_python_across_shapes_and_iterations():
    for K, N, seed, iters in [
        (128, 32, 5, 40),
        (64, 16, 11, 100),
        (96, 24, 21, 800),
        (32, 8, 23, 1),
    ]:
        float_model, quant_model = _matmul_int4_models(K=K, N=N, seed=seed)
        calib = _correlated_calibration(K=K, seed=seed + 100)
        # A small, measured tolerance -- see _assert_agrees's own comment.
        _assert_agrees(
            float_model,
            quant_model,
            calib,
            max_mismatch_frac=0.01,
            num_iterations=iters,
        )


def test_adaround_cpp_gemm_transb():
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


def test_adaround_cpp_gemm_transb_with_bias():
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


def test_adaround_cpp_ill_conditioned_calibration():
    # Dead channel plus near-duplicate channel: exercises the same
    # regularization-sensitive paths test_gptq_cpp.py's own analogous
    # test does -- agreement must still be exact.
    K, N = 64, 16
    float_model, quant_model = _matmul_int4_models(K=K, N=N, seed=10)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, 4] = 0.0
    x[:, 8] = x[:, 0] * 1.0000001
    _assert_agrees(float_model, quant_model, [{"X": x}], num_iterations=300)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "num_iterations": 800,
            "learning_rate": 0.3,
            "reg_param": 0.05,
            "warm_start": 0.1,
            "beta_range": (10.0, 1.2),
        },
        {"num_iterations": 50, "warm_start": 0.0},
        {"num_iterations": 50, "warm_start": 1.0},
    ],
)
def test_adaround_cpp_matches_python_with_custom_hyperparameters(kwargs):
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=15)
    calib = _correlated_calibration(K=64, seed=16)
    _assert_agrees(float_model, quant_model, calib, **kwargs)


def test_adaround_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_adaround_cpp(
        model, quant, calibration_data=[{"X": np.zeros((1, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == quant.SerializeToString()
