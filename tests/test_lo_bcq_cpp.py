"""Tests for ``onnxsim.apply_lo_bcq_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_lo_bcq`` (LO-BCQ, see
``onnxsim/passes/lo_bcq.h``). Like ``tests/test_aqlm_cpp.py``'s own AQLM
port, LO-BCQ's own block-clustering step (grouping weight blocks by their
own [mean, std] feature vector via multi-dimensional k-means) has no
closed form and this port's own initialization is a genuinely different
(deterministic, feature-norm-sorted) scheme from lo_bcq.py's own
seeded-random-sample one -- see passes/lo_bcq.h's own "ACCEPTED, PERMANENT
DIVERGENCE" note. So these tests check structural/algebraic properties
(the block-clustered-codebook reconstruction quality against a naive
single-codebook baseline, never a tight bit-for-bit or even a tight
numeric cross-check against the pure-Python port), matching this repo's
own established contract for a ``*_cpp`` port whose underlying scheme is
non-deterministic (``tests/test_aqlm_cpp.py``, ``tests/test_kmeans_quantization_cpp.py``).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 32  # this port's own hardcoded default (lo_bcq.py's own block_size=32)
_NUM_CLUSTERS = 4  # ditto (num_clusters=4)
_NUM_CODES = 16  # ditto (2**bits, bits=4)


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
    # The C++ pass (like every other data-free *_cpp weight-only port
    # here) rewires the matched node's weight input to a freshly created
    # initializer, leaving the original one dangling unused in the graph.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_replaces_weight_with_same_shape_float():
    # num_blocks = N * (K // block_size) must comfortably exceed
    # num_clusters (4) -- otherwise lo_bcq.py's own num_blocks <=
    # num_clusters branch (a trivial `i % num_clusters` assignment, no
    # k-means at all) kicks in on both sides, which is a real but
    # degenerate case this test deliberately avoids. It must also give
    # each cluster's own flattened block-value pool (>> block_size
    # elements per block) comfortably more than num_codes (16) distinct
    # values, so a per-cluster codebook fit doesn't degenerate into
    # "every value claims its own exact centroid" the way
    # test_aqlm_cpp.py's own test had to be sized around.
    rng = np.random.default_rng(0)
    K, N = _BLOCK_SIZE * 10, 20  # num_blocks = 200 >> num_clusters (4)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_lo_bcq_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # No new graph nodes -- weight-only, folded straight into a new
    # initializer (unlike lo_bcq.py's own Gather/GatherElements/Reshape
    # graph rewrite).
    assert [n.op_type for n in q.graph.node] == [n.op_type for n in model.graph.node]
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_beats_naive_single_codebook_baseline_on_multiscale_weight():
    # LO-BCQ's whole point (versus kmeans_quantization.h's own single
    # whole-tensor codebook): blocks with wildly different magnitudes get
    # their own, separately-fit small codebook instead of sharing one. So
    # give each block one of four very different scales (cyclically, by
    # block index) -- a block's own [mean, std] feature vector separates
    # cleanly by scale, so block-clustering should group same-scale
    # blocks together and each cluster's own codebook should fit its own
    # scale well. A naive *single* 16-level codebook fit directly on the
    # whole (0.01x .. 100x dynamic range) flattened tensor cannot
    # represent the small-scale blocks well at all, since its centroids
    # are dominated by the large-scale values.
    rng = np.random.default_rng(1)
    K, N = _BLOCK_SIZE * 8, 4  # num_blocks = 32
    num_blocks_per_row = K // _BLOCK_SIZE
    scales = np.array([100.0, 10.0, 1.0, 0.01])
    w_nk = np.empty((N, K), dtype=np.float64)
    for n_idx in range(N):
        for b in range(num_blocks_per_row):
            block_idx = n_idx * num_blocks_per_row + b
            scale = scales[block_idx % len(scales)]
            w_nk[n_idx, b * _BLOCK_SIZE : (b + 1) * _BLOCK_SIZE] = (
                rng.standard_normal(_BLOCK_SIZE) * scale
            )
    w = w_nk.T.astype(np.float32)  # [K, N], MatMul layout
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_lo_bcq_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    # Naive single global 16-level codebook, fit via the same kind of
    # ordinary 1-D Lloyd's k-means this port's own per-cluster codebooks
    # use, but over the WHOLE flattened tensor at once (matching
    # kmeans_quantization.h's own single-codebook scope exactly).
    flat = w64.ravel()
    sorted_vals = np.sort(flat)
    percentiles = np.linspace(0, 100, _NUM_CODES)
    centroids = np.unique(np.percentile(sorted_vals, percentiles))
    if centroids.shape[0] < _NUM_CODES:
        extra_rng = np.random.default_rng(123)
        extra = extra_rng.choice(
            flat, size=_NUM_CODES - centroids.shape[0], replace=True
        )
        centroids = np.concatenate([centroids, extra])
    for _ in range(20):
        dist = np.abs(flat[:, None] - centroids[None, :])
        assign = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for c in range(_NUM_CODES):
            mask = assign == c
            if mask.any():
                new_centroids[c] = flat[mask].mean()
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    dist = np.abs(flat[:, None] - centroids[None, :])
    assign = np.argmin(dist, axis=1)
    naive_recon = centroids[assign].reshape(w64.shape)

    lo_bcq_err = float(np.mean((w64 - new_w) ** 2))
    naive_err = float(np.mean((w64 - naive_recon) ** 2))
    assert lo_bcq_err < naive_err


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE * 4, 8
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
    q = onnxsim.apply_lo_bcq_cpp(model)
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
    result = onnxsim.apply_lo_bcq_cpp(model)
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
    result = onnxsim.apply_lo_bcq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_block_size_divisible_k():
    # K not a multiple of block_size (32) -- lo_bcq.py's own encoder
    # skips this layer entirely (no ragged-last-block handling), and this
    # port matches that exactly.
    rng = np.random.default_rng(9)
    K, N = _BLOCK_SIZE + 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)
    result = onnxsim.apply_lo_bcq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
