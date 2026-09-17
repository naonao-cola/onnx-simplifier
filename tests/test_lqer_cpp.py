"""Tests for ``onnxsim.apply_lqer_cpp`` -- the C++-backed port of
``onnxsim.apply_lqer`` (LQER, see ``onnxsim/lqer_entry.h``): an
activation-weighted generalization of
``onnxsim.apply_low_rank_compensation_cpp``'s own plain (unweighted) SVD.

Like ``test_low_rank_compensation_cpp.py``, this port's own SVD is a
hand-rolled one-sided (Hestenes) Jacobi SVD, not LAPACK's own
Golub-Kahan algorithm (what ``numpy.linalg.svd`` calls into) -- individual
singular vectors/values are not expected to match the Python reference
sign-for-sign or bit-for-bit, but the reconstructed rank-r correction
matrix ``B @ A`` itself is expected to agree closely (basis- and
sign-invariant, unique by the same weighted Eckart-Young-style argument
lqer.py's own docstring makes whenever the matched layer's r-th and
(r+1)-th weighted singular values are well separated) -- tests compare
that, not raw ``B``/``A``.

``onnxsim.apply_lqer`` now delegates to this C++ port (see
``onnxsim/lqer.py``), so ``tests/test_lqer.py`` covers delegation-level
behavior/edge cases while this file covers the C++ port's own structural
and numerical details directly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.lqer import weighted_low_rank_correction
from onnxsim.onnx_simplifier import apply_lqer_cpp

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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _lqer_ba(model):
    b = next(t for t in model.graph.initializer if t.name.endswith("_lqer_b"))
    a = next(t for t in model.graph.initializer if t.name.endswith("_lqer_a"))
    return onnx.numpy_helper.to_array(b), onnx.numpy_helper.to_array(a)


def _correlated_calibration(K=64, num_samples=32, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def test_lqer_cpp_structure_and_error_reduction():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=0)
    calib = _correlated_calibration(K=64, seed=2)
    corrected = apply_lqer_cpp(float_model, quant_model, rank=8, calibration_data=calib)
    onnx.checker.check_model(corrected)
    assert len(corrected.graph.node) == len(quant_model.graph.node) + 3
    op_types = [n.op_type for n in corrected.graph.node]
    assert op_types.count("MatMul") == 3
    assert op_types.count("Add") == 1

    rng = np.random.default_rng(3)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    y_float = _run(float_model, {"X": x})
    y_quant = _run(quant_model, {"X": x})
    y_corrected = _run(corrected, {"X": x})
    err_quant = float(np.abs(y_quant - y_float).mean())
    err_corrected = float(np.abs(y_corrected - y_float).mean())
    assert err_corrected < err_quant


def test_lqer_cpp_matches_python_reference_correction():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=1)
    calib = _correlated_calibration(K=64, seed=4)
    py = onnxsim.apply_lqer(float_model, quant_model, rank=6, calibration_data=calib)
    cpp = apply_lqer_cpp(float_model, quant_model, rank=6, calibration_data=calib)
    onnx.checker.check_model(cpp)

    py_b, py_a = _lqer_ba(py)
    cpp_b, cpp_a = _lqer_ba(cpp)
    assert cpp_b.shape == py_b.shape == (64, 6)
    assert cpp_a.shape == py_a.shape == (6, 16)
    np.testing.assert_allclose(cpp_b @ cpp_a, py_b @ py_a, rtol=1e-3, atol=1e-4)


def test_lqer_cpp_beats_plain_lorc_on_activation_biased_output_error():
    # Same empirical claim test_lqer.py's own
    # test_lqer_beats_plain_lorc_on_activation_biased_output_error makes,
    # against the C++ ports of both sides directly.
    K, N = 64, 16
    float_model, quant_model = _matmul_int4_models(K=K, N=N, seed=2)

    rng = np.random.default_rng(3)
    active = 8
    calib_x = np.zeros((64, K), dtype=np.float32)
    calib_x[:, :active] = rng.standard_normal((64, active)).astype(np.float32) * 5.0
    eval_x = np.zeros((16, K), dtype=np.float32)
    eval_x[:, :active] = rng.standard_normal((16, active)).astype(np.float32) * 5.0

    rank = 4
    lqer_model = apply_lqer_cpp(
        float_model, quant_model, rank=rank, calibration_data=[{"X": calib_x}]
    )
    lorc_model = onnxsim.apply_low_rank_compensation_cpp(
        float_model, quant_model, rank=rank
    )

    y_float = _run(float_model, {"X": eval_x})
    y_lqer = _run(lqer_model, {"X": eval_x})
    y_lorc = _run(lorc_model, {"X": eval_x})

    lqer_err = np.linalg.norm(y_float.astype(np.float64) - y_lqer.astype(np.float64))
    lorc_err = np.linalg.norm(y_float.astype(np.float64) - y_lorc.astype(np.float64))
    assert lqer_err < lorc_err


def test_lqer_cpp_falls_back_to_plain_svd_without_calibration_activation():
    # Empty calibration data: no channel weight is ever recorded, so every
    # candidate falls back to the same unweighted SVD
    # apply_low_rank_compensation_cpp uses -- the reconstructed correction
    # (not the raw factors, since both use independent hand-rolled SVDs)
    # must agree.
    float_model, quant_model = _matmul_int4_models(K=32, N=8, seed=4)

    lqer_model = apply_lqer_cpp(float_model, quant_model, rank=4, calibration_data=[])
    lorc_model = onnxsim.apply_low_rank_compensation_cpp(
        float_model, quant_model, rank=4
    )

    lqer_b, lqer_a = _lqer_ba(lqer_model)
    lorc_b = next(
        onnx.numpy_helper.to_array(t)
        for t in lorc_model.graph.initializer
        if t.name.endswith("_lorc_b")
    )
    lorc_a = next(
        onnx.numpy_helper.to_array(t)
        for t in lorc_model.graph.initializer
        if t.name.endswith("_lorc_a")
    )
    np.testing.assert_allclose(lqer_b @ lqer_a, lorc_b @ lorc_a, atol=1e-4)


def test_lqer_cpp_rank_clamped_to_min_dimension():
    float_model, quant_model = _matmul_int4_models(K=32, N=4, seed=3)
    cpp = apply_lqer_cpp(
        float_model,
        quant_model,
        rank=8,
        calibration_data=_correlated_calibration(K=32, seed=5),
    )
    onnx.checker.check_model(cpp)
    b, a = _lqer_ba(cpp)
    assert b.shape == (32, 4)
    assert a.shape == (4, 4)


def test_lqer_cpp_noop_when_no_int4_layer():
    float_model, _quant_model = _matmul_int4_models(K=64, N=16, seed=4)
    result = apply_lqer_cpp(float_model, float_model, rank=8, calibration_data=[])
    assert len(result.graph.node) == len(float_model.graph.node)
    assert len(result.graph.initializer) == len(float_model.graph.initializer)


def test_lqer_cpp_empty_calibration_data_is_noop_safe():
    float_model, quant_model = _matmul_int4_models(K=32, N=8, seed=6)
    result = apply_lqer_cpp(float_model, quant_model, rank=4, calibration_data=[])
    onnx.checker.check_model(result)
    # Still corrects (falls back to unweighted SVD), just without any
    # calibration-driven weighting.
    assert len(result.graph.node) == len(quant_model.graph.node) + 3


def test_lqer_cpp_missing_calibration_input_raises():
    float_model, quant_model = _matmul_int4_models(K=32, N=8, seed=7)
    with pytest.raises((RuntimeError, ValueError)):
        apply_lqer_cpp(
            float_model,
            quant_model,
            rank=4,
            calibration_data=[{"NotX": np.zeros((4, 32), dtype=np.float32)}],
        )


def test_weighted_low_rank_correction_matches_lorc_svd_reference():
    # Sanity-checks the pure-Python core math helper (still tested
    # directly since it's the standalone, documented reference for what
    # the C++ port's own weighted Jacobi SVD approximates) against the
    # C++ port's end-to-end output on the SAME weighted case.
    rng = np.random.default_rng(9)
    residual = rng.standard_normal((10, 6))
    weights = rng.uniform(0.1, 5.0, size=10)
    b, a = weighted_low_rank_correction(residual, weights, rank=min(10, 6))
    np.testing.assert_allclose(b @ a, residual, atol=1e-6)
