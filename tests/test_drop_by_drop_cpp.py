"""Tests for ``onnxsim.apply_drop_by_drop_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_drop_by_drop`` (Drop-by-Drop, see
``onnxsim/passes/drop_by_drop.h``). Like AQLM's own greedy residual
k-means fit, Drop-by-Drop's own *weighted* Lloyd's-algorithm fit has no
closed form, and this port's own initialization is a genuinely different
(deterministic, magnitude-sorted) scheme from drop_by_drop.py's own
seeded-random-sample one -- see passes/drop_by_drop.h's own "ACCEPTED,
PERMANENT DIVERGENCE" note. So these tests check structural/algebraic
properties (the additive-codebook reconstruction quality, never a tight
bit-for-bit or even a tight numeric cross-check against the pure-Python
port), matching this repo's own established contract for a *_cpp port
whose underlying scheme is non-deterministic
(``tests/test_aqlm_cpp.py``, ``tests/test_kmeans_quantization_cpp.py``).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_GROUP_DIM = 8


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
    # The C++ pass (like every other *_cpp weight-only port here) rewires
    # the matched node's weight input to a freshly created initializer,
    # leaving the original one dangling unused in the graph.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_replaces_weight_with_same_shape_float():
    # num_groups = N * (K // group_dim) must exceed codebook_size (256)
    # for the "must differ from input" assertion below to be meaningful:
    # when every group can claim its own exact centroid (num_groups <=
    # codebook_size), the weighted greedy residual k-means fit
    # reconstructs every group exactly regardless of initialization
    # scheme -- the same degenerate case tests/test_aqlm_cpp.py's own
    # comment documents for AQLM's own (unweighted) fit; the weighting
    # only changes which points a codebook slot is pulled toward, not
    # whether a codebook can have as many distinct slots as there are
    # groups.
    rng = np.random.default_rng(0)
    K, N = _GROUP_DIM * 40, 10  # num_groups = 400 > codebook_size (256)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_drop_by_drop_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # No new graph nodes -- weight-only, folded straight into a new
    # initializer (unlike drop_by_drop.py's own Gather/Add graph rewrite
    # and its named partial{k} prefix outputs).
    assert [n.op_type for n in q.graph.node] == [n.op_type for n in model.graph.node]
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_beats_single_codebook_baseline():
    # Drop-by-Drop's own whole point: each additional additive codebook
    # stage targets exactly the previous stages' own leftover residual
    # (fit with the same fixed, per-group importance weight throughout),
    # so it can only reduce (never increase) reconstruction error versus
    # stopping after one codebook. This port hardcodes num_codebooks=4
    # (see passes/drop_by_drop.h's own scope note), so this test
    # reconstructs a plausible single-codebook-only baseline directly (an
    # importance-weighted k-means fit, the same shape as this port's own
    # first stage) rather than varying a knob the C++ port doesn't
    # expose.
    rng = np.random.default_rng(1)
    K, N = _GROUP_DIM * 8, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_drop_by_drop_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    # A single-codebook (16-centroid) baseline fit directly on the raw
    # per-group vectors, via importance-weighted Lloyd's k-means (the
    # same fixed per-group RMS-magnitude weight this port's own first
    # stage uses) -- deliberately a much smaller codebook than
    # drop-by-drop's own 256-entry one per stage, so the four-stage
    # additive reconstruction has a clear quality edge to demonstrate on
    # this small a weight.
    w_nk = w64.T  # [N, K]
    groups = w_nk.reshape(N * (K // _GROUP_DIM), _GROUP_DIM)
    importance = np.sqrt(np.mean(groups**2, axis=1))
    importance = np.maximum(importance, importance.max() * 1e-6 + 1e-12)
    rng2 = np.random.default_rng(2)
    k = 16
    centroids = groups[rng2.choice(groups.shape[0], size=k, replace=False)].copy()
    for _ in range(10):
        dist = np.sum((groups[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dist, axis=1)
        for c in range(k):
            mask = assign == c
            if np.any(mask):
                w_mask = importance[mask]
                centroids[c] = (groups[mask] * w_mask[:, None]).sum(
                    axis=0
                ) / w_mask.sum()
    dist = np.sum((groups[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    assign = np.argmin(dist, axis=1)
    naive_recon = centroids[assign].reshape(N, K).T

    dbd_err = float(np.mean((w64 - new_w) ** 2))
    naive_err = float(np.mean((w64 - naive_recon) ** 2))
    assert dbd_err < naive_err


def test_cpp_at_most_256_distinct_group_reconstructions_per_stage_budget():
    # Structural sanity check: with codebook_size=256 (this port's own
    # hardcoded default) and num_codebooks=4, far more than 256 distinct
    # group reconstructions are representable -- far more than the
    # handful of groups in this small test weight, so every group should
    # be able to reconstruct closely (this is not a tight numeric bound,
    # just a sanity check that the port is doing real work, not returning
    # the identity or a constant).
    rng = np.random.default_rng(3)
    K, N = _GROUP_DIM * 2, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_drop_by_drop_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)
    assert np.linalg.norm(w64 - new_w) < np.linalg.norm(w64) * 0.5


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _GROUP_DIM * 4, 8
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
    q = onnxsim.apply_drop_by_drop_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = _GROUP_DIM * 2, 6
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
    q = onnxsim.apply_drop_by_drop_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == weight.shape


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_drop_by_drop_cpp(model)
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
    result = onnxsim.apply_drop_by_drop_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_group_divisible_k():
    # K not a multiple of 8 -- drop_by_drop.py's own encoder skips this
    # layer entirely (no ragged-last-group handling), and this port
    # matches that exactly.
    rng = np.random.default_rng(9)
    K, N = _GROUP_DIM + 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)
    result = onnxsim.apply_drop_by_drop_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
