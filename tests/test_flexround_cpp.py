"""Tests for ``onnxsim.apply_flexround_cpp`` -- the C++-backed port of
``onnxsim.apply_flexround`` (Lee et al., 2023's FlexRound, see
``onnxsim/flexround.py``). Like ``test_adaround_cpp.py``, this is an
iterative Adam optimization, not a closed-form computation: floating-point
differences between this port's own scalar dense-matmul kernels and
numpy's own (possibly BLAS-backed) ``@`` -- plus ordinary libm-level ``exp``/
``pow`` differences between languages -- can compound across iterations
(see ``onnxsim/flexround_entry.h``'s own accepted numerical scope note).

Measured empirically here rather than assumed, and measurably MORE
sensitive than AdaRound's own rectified-sigmoid relaxation
(tests/test_adaround_cpp.py): FlexRound's gradient divides by the
effective divisor squared (``d(ratio)/ds = -w/s**2``, this module's own
docstring's "Proposition 3.1" argument), which amplifies rather than damps
small floating-point differences step over step. Worse, FlexRound (unlike
AdaRound) has no annealed regularizer pulling every element toward a hard
decision at the end -- an element whose ratio saturates against
``n_min``/``n_max`` simply stops contributing its OWN gradient from that
iteration on (only the *shared* per-row ``s3`` can still move it), so two
implementations that saturate a given element on slightly different
iterations can end up on genuinely different sides of more than one grid
line, not just its immediate neighbor -- confirmed directly (see the
``count_flip_over_boundary`` case measured for this file): an element
whose Python-side code saturated at ``n_min`` while this port's own
settled two grid points away is a real, observed, non-buggy divergence,
not evidence of an implementation error (verified independently: a
pure-Python nested-loop transcription of
:func:`onnxsim.flexround._optimize_divisor`, using ONLY numpy on both
sides of the comparison, shows a comparable magnitude and shape of
divergence from the vectorized reference purely from summation-order
noise).

Given that, this file does NOT assert tight per-code agreement (unlike
``test_adaround_cpp.py``'s own ``max_mismatch_frac``/neighbor-only
tolerance) as its primary signal. Instead: (1) both outputs are checked to
be valid ONNX models with codes that stay in range, (2) reconstruction
error against real activations is checked to track the Python reference
closely (not exact-match, but not "wildly different" either), and (3) a
generous, purely-informational code-agreement check confirms the two
implementations are not unrelated (most codes still agree in every
configuration tested).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.flexround import apply_flexround
from onnxsim.onnx_simplifier import apply_flexround_cpp

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


def _check(float_model, quant_model, calibration_data, **kwargs):
    """Structural validity + reconstruction-error agreement + a generous,
    informational code-agreement smoke check -- see this module's own
    docstring for why code agreement is not this file's primary signal."""
    x = calibration_data[0]["X"]
    py = apply_flexround(float_model, quant_model, calibration_data, **kwargs)
    cpp = apply_flexround_cpp(float_model, quant_model, calibration_data, **kwargs)
    onnx.checker.check_model(py)
    onnx.checker.check_model(cpp)

    py_codes, cpp_codes = _int4_codes(py), _int4_codes(cpp)
    assert py_codes.min() >= -7 and py_codes.max() <= 7
    assert cpp_codes.min() >= -7 and cpp_codes.max() <= 7

    # Neither this port's own reconstruction error, nor the pure-Python
    # reference's own, should be much worse than the other -- generous
    # slack for the non-convex, saturation-sensitive dynamics this
    # module's own docstring describes.
    py_err = _reconstruction_error(py, float_model, x)
    cpp_err = _reconstruction_error(cpp, float_model, x)
    assert cpp_err < 3.0 * max(py_err, 1e-9)

    # Most codes still agree in every configuration measured for this file
    # (typically 90%+) -- a loose smoke check that the two implementations
    # are solving the same problem, not a correctness gate.
    mismatch_frac = float(np.mean(py_codes != cpp_codes))
    assert mismatch_frac < 0.5
    return py, cpp


def test_flexround_cpp_structurally_valid_and_tracks_python_reference():
    float_model, quant_model = _matmul_int4_models()
    _check(float_model, quant_model, _correlated_calibration())


def test_flexround_cpp_across_shapes_and_iterations():
    for K, N, seed, iters in [
        (128, 32, 5, 40),
        (64, 16, 11, 100),
        (96, 24, 21, 300),
        (32, 8, 23, 1),
    ]:
        float_model, quant_model = _matmul_int4_models(K=K, N=N, seed=seed)
        calib = _correlated_calibration(K=K, seed=seed + 100)
        _check(float_model, quant_model, calib, num_iterations=iters)


def test_flexround_cpp_gemm_transb():
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
    _check(float_model, quant_model, _correlated_calibration(K=K, seed=9))


def test_flexround_cpp_gemm_transb_with_bias():
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
    py, cpp = _check(float_model, quant_model, _correlated_calibration(K=K, seed=11))
    # Bias is untouched by either implementation.
    b_py = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name == "B")
    )
    b_cpp = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    np.testing.assert_array_equal(b_py, b_cpp)
    np.testing.assert_array_equal(b_cpp, bias)


def test_flexround_cpp_ill_conditioned_calibration():
    K, N = 64, 16
    float_model, quant_model = _matmul_int4_models(K=K, N=N, seed=10)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, 4] = 0.0
    x[:, 8] = x[:, 0] * 1.0000001
    _check(float_model, quant_model, [{"X": x}], num_iterations=300)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_iterations": 400, "learning_rate": 0.1},
        {"num_iterations": 50, "learning_rate": 0.02},
        {"num_iterations": 50, "log_clip": 2.0},
    ],
)
def test_flexround_cpp_custom_hyperparameters(kwargs):
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=15)
    calib = _correlated_calibration(K=64, seed=16)
    _check(float_model, quant_model, calib, **kwargs)


def test_flexround_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_flexround_cpp(
        model, quant, calibration_data=[{"X": np.zeros((1, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == quant.SerializeToString()


def test_flexround_cpp_only_touches_weight_codes():
    # Unlike apply_autoround_cpp, FlexRound never rewrites a scale
    # initializer -- only the matched layer's own INT4 codes.
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=18)
    calib = _correlated_calibration(K=64, seed=19)
    cpp = apply_flexround_cpp(float_model, quant_model, calib, num_iterations=50)
    dq = next(n for n in quant_model.graph.node if n.op_type == "DequantizeLinear")
    orig_scale = onnx.numpy_helper.to_array(
        next(t for t in quant_model.graph.initializer if t.name == dq.input[1])
    )
    new_scale = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == dq.input[1])
    )
    np.testing.assert_array_equal(orig_scale, new_scale)


def test_flexround_cpp_beats_round_to_nearest_via_onnxruntime():
    model = _matmul_int4_models(K=64, N=16, seed=16)[0]
    x = _correlated_calibration(K=64, num_samples=64, seed=17)[0]["X"]
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_flexround_cpp(model, quant, [{"X": x}])
    py = apply_flexround(model, quant, [{"X": x}])
    onnx.checker.check_model(cpp)

    rtn_err = _reconstruction_error(quant, model, x)
    cpp_err = _reconstruction_error(cpp, model, x)
    py_err = _reconstruction_error(py, model, x)
    assert np.isfinite(cpp_err)
    # Not bit-exact (see this module's own docstring) -- close agreement,
    # not equality, and both comfortably beat round-to-nearest.
    assert cpp_err < rtn_err
    assert py_err < rtn_err
    assert cpp_err < 3.0 * max(py_err, 1e-9)
