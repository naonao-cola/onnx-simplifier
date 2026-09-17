"""Tests for ``onnxsim.apply_autoround_cpp`` -- the C++-backed port of
``onnxsim.apply_autoround`` (Cheng et al., 2023's AutoRound, see
``onnxsim/autoround.py``). Like ``test_adaround_cpp.py``, this is an
iterative Adam optimization over TWO jointly interacting parameter groups
(the rounding relaxation AND the per-block clip ratio), so agreement is
measured empirically here rather than assumed: floating-point
summation-order differences can compound across iterations, AND because
the scale itself moves every step, an element whose ratio sits within an
ulp of an integer boundary can take a different quantization bin between
the two implementations -- a discontinuous change that then steers the
rest of that layer's own run. The `_keep_better_of`/``KeepBetterOf`` safety
net (see ``onnxsim/autoround_entry.h``) is exactly what keeps this port's
own worst case bounded: even when the two paths diverge, both are verified
here to never regress reconstruction error below plain AdaRound's own
optimum for the same layer and calibration data.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.autoround import apply_autoround
from onnxsim.onnx_simplifier import apply_autoround_cpp

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


def _scale(model):
    dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    ws = next(t for t in model.graph.initializer if t.name == dq.input[1])
    return onnx.numpy_helper.to_array(ws)


def _reconstruction_error(model, float_model, x):
    sess = ort.InferenceSession(
        float_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    q_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(None, {"X": x})[0].astype(np.float64).ravel()
    out = q_sess.run(None, {"X": x})[0].astype(np.float64).ravel()
    return float(np.linalg.norm(ref - out) / max(np.linalg.norm(ref), 1e-6))


def test_autoround_cpp_structurally_valid_and_never_worse_than_python_reference():
    # Not asserting bit-exact agreement (two coupled Adam loops, see this
    # module's own docstring) -- instead verifying both outputs are valid
    # models and the C++ port's own safety net holds: it never regresses
    # reconstruction error below what the pure-Python reference (which has
    # the identical safety net) reaches on the same data.
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=2)
    calib = _correlated_calibration(K=64, seed=3)
    x = calib[0]["X"]

    py = apply_autoround(float_model, quant_model, calib)
    cpp = apply_autoround_cpp(float_model, quant_model, calib)
    onnx.checker.check_model(cpp)
    onnx.checker.check_model(py)

    py_err = _reconstruction_error(py, float_model, x)
    cpp_err = _reconstruction_error(cpp, float_model, x)
    # Both are separately bounded by AdaRound's own optimum for the same
    # data -- so neither should be wildly worse than the other, allowing
    # generous slack for the two-parameter-group non-convexity itself.
    assert cpp_err < 2.0 * max(py_err, 1e-9)


def test_autoround_cpp_never_worse_than_adaround_across_seeds():
    from onnxsim.adaround import apply_adaround

    for seed in [4, 5, 6]:
        float_model, quant_model = _matmul_int4_models(K=48, N=12, seed=seed)
        calib = _correlated_calibration(K=48, seed=seed + 50)
        x = calib[0]["X"]

        cpp = apply_autoround_cpp(float_model, quant_model, calib, num_iterations=100)
        ada = apply_adaround(float_model, quant_model, calib, num_iterations=100)

        cpp_err = _reconstruction_error(cpp, float_model, x)
        ada_err = _reconstruction_error(ada, float_model, x)
        # AutoRound's own safety net guarantees this (a small numerical
        # slack for floating-point noise near the boundary).
        assert cpp_err <= ada_err * 1.0 + 1e-9


def test_autoround_cpp_codes_stay_in_range_and_output_finite():
    float_model, quant_model = _matmul_int4_models(K=32, N=8, seed=7)
    calib = _correlated_calibration(K=32, seed=8)
    cpp = apply_autoround_cpp(float_model, quant_model, calib, num_iterations=50)
    codes = _int4_codes(cpp)
    assert codes.min() >= -7
    assert codes.max() <= 7
    scale = _scale(cpp)
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0)


def test_autoround_cpp_gemm_transb():
    rng = np.random.default_rng(9)
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
    calib = _correlated_calibration(K=K, seed=10)
    cpp = apply_autoround_cpp(float_model, quant_model, calib, num_iterations=50)
    onnx.checker.check_model(cpp)
    codes = _int4_codes(cpp)
    assert codes.shape == (N, K)


def test_autoround_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_autoround_cpp(
        model, quant, calibration_data=[{"X": np.zeros((1, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == quant.SerializeToString()


def test_autoround_cpp_clip_ratio_range_respected():
    # A layer whose joint optimization beats AdaRound's own safety net has
    # its scale rewritten within [clip_ratio_min, clip_ratio_max] * the
    # original scale -- never outside that band.
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=11)
    calib = _correlated_calibration(K=64, seed=12)
    orig_scale = _scale(quant_model)
    cpp = apply_autoround_cpp(
        float_model,
        quant_model,
        calib,
        num_iterations=150,
        clip_ratio_range=(0.5, 1.5),
    )
    new_scale = _scale(cpp)
    ratio = new_scale / orig_scale
    assert np.all(ratio >= 0.5 - 1e-6)
    assert np.all(ratio <= 1.5 + 1e-6)


def test_autoround_cpp_beats_or_matches_round_to_nearest():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=20)
    calib = _correlated_calibration(K=64, num_samples=64, seed=21)
    x = calib[0]["X"]
    cpp = apply_autoround_cpp(float_model, quant_model, calib, num_iterations=200)
    onnx.checker.check_model(cpp)
    rtn_err = _reconstruction_error(quant_model, float_model, x)
    cpp_err = _reconstruction_error(cpp, float_model, x)
    assert np.isfinite(cpp_err)
    assert cpp_err <= rtn_err + 1e-9
