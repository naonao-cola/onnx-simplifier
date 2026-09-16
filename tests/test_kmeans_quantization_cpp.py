"""Tests for ``onnxsim.apply_kmeans_quantization_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_kmeans`` (see
``onnxsim/passes/kmeans_quantization.h``). Unlike every fixed-codebook
scheme in this repo, k-means has no closed form, so this port's own
"ACCEPTED, PERMANENT DIVERGENCE" note documents a genuine (though narrow
and rare) case where the two ports' codebooks are not expected to match
at all: a weight tensor with fewer than 16 distinct percentile-derived
initial centroids. Outside that edge case (any realistic float32 weight
tensor, including every one used below), initialization is fully
deterministic and identical between the two ports, so these tests check
comparable (not required to be bit-for-bit identical) reconstruction
error and structural properties, matching this repo's own established
contract for a ``*_cpp`` port (``tests/test_gguf_q2_k_cpp.py``).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_NUM_CODES = 16  # 2**bits, bits=4 (this port's own hardcoded default)


def _model(body, initializer=(), opset=13, ir_version=8):
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
    # The C++ pass (like every other data-free *_cpp port in this repo)
    # rewires the matched node's weight input to a freshly created
    # initializer, leaving the original one dangling unused in the graph
    # -- so the *node's own current input name* is the only reliable way
    # to find the actual (post-quantization) weight, not initializer list
    # position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 8)).astype(np.float32)
    model = _matmul_model(w, K=64, N=8)

    q = onnxsim.apply_kmeans_quantization_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # every other data-free *_cpp port's own established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_at_most_16_distinct_values_in_the_whole_tensor():
    # Unlike every block-quant format in this repo, k-means fits ONE
    # codebook to the WHOLE weight tensor, not per-block -- so the
    # distinct-value cap applies globally, not per 16/32/256-element
    # chunk.
    rng = np.random.default_rng(1)
    w = rng.standard_normal((128, 16)).astype(np.float32) * 0.5
    model = _matmul_model(w, K=128, N=16)

    q = onnxsim.apply_kmeans_quantization_cpp(model)
    new_w = _current_weight(q)
    assert len(np.unique(new_w)) <= _NUM_CODES


def test_cpp_beats_naive_single_scale_int4_on_clustered_weight():
    # A weight tensor with a handful of tight, well-separated clusters
    # is exactly what k-means is good at, and exactly what a naive
    # single-scale symmetric int4 quantizer (16 evenly-spaced levels
    # across the tensor's own min/max) is bad at when the clusters are
    # not themselves evenly spaced.
    rng = np.random.default_rng(2)
    centers = np.array([-10.0, -0.1, 0.05, 3.0, 50.0])
    assignments = rng.integers(0, len(centers), size=64 * 8)
    w = (
        (centers[assignments] + rng.standard_normal(64 * 8) * 0.01)
        .reshape(64, 8)
        .astype(np.float32)
    )
    model = _matmul_model(w, K=64, N=8)

    q = onnxsim.apply_kmeans_quantization_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    lo, hi = w64.min(), w64.max()
    naive_scale = max(abs(lo), abs(hi)) / 8.0
    naive_codes = np.clip(np.round(w64 / naive_scale), -8, 7)
    naive_out = naive_codes * naive_scale

    kmeans_mse = float(np.mean((w64 - new_w) ** 2))
    naive_mse = float(np.mean((w64 - naive_out) ** 2))
    assert kmeans_mse < naive_mse


def test_cpp_behaves_similarly_to_python_port():
    # Not required to be bit-for-bit identical (see
    # passes/kmeans_quantization.h's own documented divergence note), but
    # for a weight tensor with plenty of distinct values (so the
    # too-few-distinct-percentiles fallback never triggers on either
    # side), initialization is fully deterministic and identical between
    # the two ports, so Lloyd's algorithm is expected to converge to
    # matching codebooks up to ordinary floating-point summation-order
    # noise.
    rng = np.random.default_rng(4)
    w = rng.standard_normal((64, 8)).astype(np.float32) * 0.5
    model = _matmul_model(w, K=64, N=8)

    py_q = onnxsim.quantize_weight_only_kmeans(model)
    cpp_q = onnxsim.apply_kmeans_quantization_cpp(model)

    # The Python port keeps the codebook/codes split visible in the graph
    # (Gather(Codebook, Cast(Codes, INT64))); the C++ port folds straight
    # to a replacement float32 initializer (see this header's own scope
    # note). Reconstruct the Python side's own dequantized weight
    # directly from its codebook/codes initializers instead.
    py_codebook = onnx.numpy_helper.to_array(
        next(t for t in py_q.graph.initializer if t.name.endswith("_codebook"))
    )
    py_codes = onnx.numpy_helper.to_array(
        next(t for t in py_q.graph.initializer if t.name.endswith("_codes"))
    )
    py_w = py_codebook[py_codes.astype(np.int64)].astype(np.float64)

    cpp_w = _current_weight(cpp_q).astype(np.float64)
    w64 = w.astype(np.float64)

    py_err = np.linalg.norm(py_w - w64)
    cpp_err = np.linalg.norm(cpp_w - w64)
    assert cpp_err < np.linalg.norm(w64) * 0.5
    assert cpp_err < py_err * 1.5 and py_err < cpp_err * 1.5


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = 64, 8
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
    q = onnxsim.apply_kmeans_quantization_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_kmeans_quantization_cpp(model)
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
    result = onnxsim.apply_kmeans_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
