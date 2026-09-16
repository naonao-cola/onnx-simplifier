"""Tests for ``onnxsim.apply_low_rank_compensation_cpp`` -- the C++-backed
port of ``onnxsim.apply_low_rank_compensation`` (Low-Rank Compensation,
ZeroQuant-V2's LoRC; see ``onnxsim/low_rank_compensation_entry.h``). Like
``test_daq_cpp.py``, this technique is data-free (no ModelExecutor/
calibration data), but unlike every other data-free ``*_cpp`` port's own
fold-to-a-single-initializer shape, LoRC's own correction is *additive to
a matched layer's output*: it adds two new ``MatMul`` nodes and an
``Add`` node to the graph, so a test here has to check real graph
topology, not just a replacement weight.

This port's own SVD is a hand-rolled one-sided (Hestenes) Jacobi SVD, not
LAPACK's own Golub-Kahan algorithm (what ``numpy.linalg.svd`` calls into)
-- see ``low_rank_compensation_entry.cpp``'s own top-of-file comment for
why (no linear-algebra library is linked into this codebase) and for the
numerical caveat: individual singular vectors/values are not expected to
match the Python reference sign-for-sign or bit-for-bit. What *is*
expected to match closely is the reconstructed rank-r correction matrix
``B @ A`` itself (unique, by the Eckart-Young theorem, whenever the
matched layer's r-th and (r+1)-th singular values are well separated) --
``test_matches_python_reference_correction`` below compares that
reconstruction, not the raw ``B``/``A`` factors.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.low_rank_compensation import apply_low_rank_compensation
from onnxsim.onnx_simplifier import apply_low_rank_compensation_cpp

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
    """A plain MatMul float model, and its ``quantize_weight_only_int4``
    counterpart (block_size=32, one INT4 code + block-wise scale per 32
    elements of K, matching the axis/block_size this port's own candidate
    matching requires -- see quantize_weight_only_int4's own docstring).
    """
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


def _lorc_ba(model):
    """Finds the single correction's B/A initializers by their own
    ``<output_name>_lorc_b``/``_lorc_a`` naming convention (shared,
    verbatim, between low_rank_compensation.py and this C++ port)."""
    b = next(t for t in model.graph.initializer if t.name.endswith("_lorc_b"))
    a = next(t for t in model.graph.initializer if t.name.endswith("_lorc_a"))
    return onnx.numpy_helper.to_array(b), onnx.numpy_helper.to_array(a)


def test_correction_reduces_error_toward_float():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=0)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 64)).astype(np.float32)

    y_float = _run(float_model, {"X": x})
    y_quant = _run(quant_model, {"X": x})

    corrected = apply_low_rank_compensation_cpp(float_model, quant_model, rank=8)
    onnx.checker.check_model(corrected)
    # Three new nodes (2 MatMul + 1 Add) beyond the original single MatMul.
    assert len(corrected.graph.node) == len(quant_model.graph.node) + 3
    y_corrected = _run(corrected, {"X": x})

    err_quant = float(np.abs(y_quant - y_float).mean())
    err_corrected = float(np.abs(y_corrected - y_float).mean())
    # A rank-8 correction out of min(K, N) = 16 possible ranks captures
    # half of the quantization error's own singular directions (by the
    # Eckart-Young theorem, the best possible for that rank) -- this is
    # not a close call for random data, so a strict improvement is
    # expected, not just tolerated.
    assert err_corrected < err_quant


def test_matches_python_reference_correction():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=1)
    py = apply_low_rank_compensation(float_model, quant_model, rank=6)
    cpp = apply_low_rank_compensation_cpp(float_model, quant_model, rank=6)
    onnx.checker.check_model(cpp)

    py_b, py_a = _lorc_ba(py)
    cpp_b, cpp_a = _lorc_ba(cpp)
    assert cpp_b.shape == py_b.shape == (64, 6)
    assert cpp_a.shape == py_a.shape == (6, 16)

    # Not a raw-factor comparison (see this module's own docstring): B @ A
    # is the actual best rank-6 approximation of the same residual matrix
    # on both sides, unique (by Eckart-Young) up to ordinary floating-point
    # rounding whenever the 6th and 7th singular values are well
    # separated, which random data of this shape practically always gives.
    np.testing.assert_allclose(cpp_b @ cpp_a, py_b @ py_a, rtol=1e-3, atol=1e-4)


def test_rank_clamped_to_min_dimension():
    # K=32 (one full block_size=32 block), N=4: r = min(rank=8, K=32, N=4)
    # clamps to 4, not 8.
    float_model, quant_model = _matmul_int4_models(K=32, N=4, seed=3)
    cpp = apply_low_rank_compensation_cpp(float_model, quant_model, rank=8)
    onnx.checker.check_model(cpp)
    b, a = _lorc_ba(cpp)
    assert b.shape == (32, 4)
    assert a.shape == (4, 4)


def test_noop_when_no_int4_layer():
    float_model, _quant_model = _matmul_int4_models(K=64, N=16, seed=4)
    # Passing the float model itself as "quantized_model": its own MatMul
    # weight is a plain FLOAT initializer, not a DequantizeLinear(INT4,
    # ...)-fed one, so no candidate can ever match.
    result = apply_low_rank_compensation_cpp(float_model, float_model, rank=8)
    assert len(result.graph.node) == len(float_model.graph.node)
    assert len(result.graph.initializer) == len(float_model.graph.initializer)


def test_noop_when_shape_mismatch():
    _matched_float_model, quant_model = _matmul_int4_models(K=64, N=16, seed=5)
    mismatched_float_model, _ = _matmul_int4_models(K=64, N=32, seed=6)
    # Both models' single MatMul node outputs "Y", so the two would match
    # by output name -- but the float side's own W is [64, 32] while
    # quant_model's own INT4 Wq is [64, 16]: a genuine shape mismatch, not
    # just a missing counterpart.
    result = apply_low_rank_compensation_cpp(
        mismatched_float_model, quant_model, rank=8
    )
    assert len(result.graph.node) == len(quant_model.graph.node)
    assert len(result.graph.initializer) == len(quant_model.graph.initializer)
