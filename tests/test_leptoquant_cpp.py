"""Tests for ``onnxsim.apply_leptoquant_cpp`` -- the C++-backed port of
``onnxsim.apply_leptoquant`` (AngelSlim's LeptoQuant outlier-aware block
FP8 weight scale search, see ``onnxsim/passes/leptoquant.h``). Like the
other simple block-quant formats in this repo, this scheme has no
accumulation or iterative-refinement step at all (see that header's own
"ACCEPTED, PERMANENT DIVERGENCE" note), so this port is expected to track
the pure-Python port unusually closely -- but these tests still check
structural/algebraic properties and comparable (not required to be
bit-for-bit identical) reconstruction error, matching this repo's own
established contract for a ``*_cpp`` port (``tests/test_gguf_q2_k_cpp.py``,
``tests/test_gguf_q6_k_cpp.py``).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.deepseek_fp8 import quantize_dequantize_block_fp8

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 128


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


def _matmul_model(w, K, N, batch="batch"):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _current_weight(model, weight_input_index=1):
    # The C++ pass (like every sibling *_cpp port) rewires the matched
    # node's weight input to a freshly created initializer, leaving the
    # original one dangling unused in the graph -- so the node's own
    # current input name is the only reliable way to find the actual
    # (post-quantization) weight, not initializer list position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def _outlier_weight(rows=128, cols=128, seed=0, num_outliers=30):
    # A Laplacian-peaked bulk plus a small planted outlier population, one
    # to two orders of magnitude above the bulk -- see test_leptoquant.py's
    # own docstring for why this is the shape LeptoQuant's grid search is
    # meant to beat plain absmax scaling on.
    rng = np.random.default_rng(seed)
    w = rng.laplace(0.0, 0.05, size=(rows, cols))
    idx = rng.choice(w.size, size=num_outliers, replace=False)
    w.flat[idx] = rng.uniform(0.5, 2.0, num_outliers) * rng.choice(
        [-1.0, 1.0], num_outliers
    )
    return w.astype(np.float32)


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((_BLOCK_SIZE * 2, 8)).astype(np.float32)
    model = _matmul_model(w, K=_BLOCK_SIZE * 2, N=8)

    q = onnxsim.apply_leptoquant_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # No new graph nodes -- weight-only, folded straight into a new
    # initializer, matching leptoquant.py's own documented behavior.
    assert [n.op_type for n in q.graph.node] == [n.op_type for n in model.graph.node]
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_beats_absmax_block_fp8_on_outlier_weight():
    # The core empirical claim (matching test_leptoquant.py's own Python
    # test of the same name): on a weight with real outlier structure,
    # LeptoQuant's per-tile grid search must reconstruct at least as well
    # as plain absmax-scaled block FP8 (deepseek_fp8's own scheme).
    w = _outlier_weight(seed=4)
    model = _matmul_model(w, K=w.shape[0], N=w.shape[1])

    q = onnxsim.apply_leptoquant_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    deepseek = quantize_dequantize_block_fp8(w64, block_size=_BLOCK_SIZE)
    lepto_mse = float(np.mean(np.square(w64 - new_w)))
    deepseek_mse = float(np.mean(np.square(w64 - deepseek)))
    assert lepto_mse <= deepseek_mse


def test_cpp_behaves_similarly_to_python_port():
    # Not required to be bit-for-bit identical (see passes/leptoquant.h's
    # own documented divergence note), but should reach a very similar
    # reconstruction error on the same input -- this scheme has no
    # accumulation/iteration-order dependence at all, so the two ports
    # are expected to track each other unusually closely among this
    # repo's *_cpp pairs.
    w = _outlier_weight(rows=192, cols=160, seed=11)
    model = _matmul_model(w, K=w.shape[0], N=w.shape[1])

    py_q = onnxsim.apply_leptoquant(model)
    cpp_q = onnxsim.apply_leptoquant_cpp(model)
    py_w = onnx.numpy_helper.to_array(py_q.graph.initializer[-1]).astype(np.float64)
    cpp_w = _current_weight(cpp_q).astype(np.float64)

    w64 = w.astype(np.float64)
    py_err = np.linalg.norm(py_w - w64)
    cpp_err = np.linalg.norm(cpp_w - w64)
    assert cpp_err < np.linalg.norm(w64) * 0.2
    assert cpp_err < py_err * 1.5 and py_err < cpp_err * 1.5


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE * 2, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_leptoquant_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = _BLOCK_SIZE, 64
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.apply_leptoquant_cpp(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == weight.shape

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_ragged_last_tile_matches_python_zero_padding():
    # Neither dimension a multiple of 128 -- passes/leptoquant.h's own
    # scope note claims this ragged-last-tile approach (quantizing only
    # each tile's own real elements) reproduces the same values as
    # leptoquant.py's own zero-pad-then-discard approach, since a
    # padding zero can never change a tile's own max/quantile
    # statistics unless the whole tile is already all-zero. Checked
    # directly here.
    rng = np.random.default_rng(6)
    K, N = _BLOCK_SIZE + 20, 40
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)

    py_q = onnxsim.apply_leptoquant(model)
    cpp_q = onnxsim.apply_leptoquant_cpp(model)
    py_w = onnx.numpy_helper.to_array(py_q.graph.initializer[-1]).astype(np.float64)
    cpp_w = _current_weight(cpp_q).astype(np.float64)

    np.testing.assert_allclose(py_w, cpp_w, atol=1e-3)


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_leptoquant_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.apply_leptoquant_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_weight_is_not_constant():
    model = _model(
        """
        g (float[4,8] X, float[8,8] W) => (float[4,8] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result = onnxsim.apply_leptoquant_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
