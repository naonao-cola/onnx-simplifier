"""Tests for ``onnxsim.apply_quip_sharp`` (QuIP#, see
``onnxsim/quip_sharp.py``) -- conjugates each matched layer's weight by a
pair of random orthogonal matrices (incoherence processing) before
quantizing 8-element groups onto the E8 lattice.

``apply_quip_sharp`` now delegates to ``onnxsim.apply_quip_sharp_cpp``
(see ``onnxsim/quip_sharp.py``'s own "Delegates to" note): the returned
graph folds the entire rotate/quantize/rotate-back sandwich into a single
replacement float32 initializer rather than keeping ``U``/``V``/the
packed INT4 lattice codes as explicit initializers and new ``MatMul``
nodes, and it no longer accepts a non-default ``seed``/``epsilon`` (both
hardcoded by the delegated C++ port) -- see ``tests/test_quip_sharp_cpp.py``
for the full C++ port test suite this one now exercises indirectly. The
E8-lattice/random-orthogonal-matrix helper functions below are unaffected
(still pure Python, reused directly by ``tests/test_quip_sharp_cpp.py``
for its own independent baseline).
"""

import itertools

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.quip_sharp import (
    _closest_point_d8,
    _closest_point_e8,
    _random_orthogonal_matrix,
)

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _matmul_model(K=32, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _brute_force_closest_e8(v):
    # Independent O(2^9)-per-point reference: try every combination of
    # floor/ceil per coordinate, for both the integer and half-integer
    # coset, keep only correct-parity candidates, return the true minimum.
    best, best_d = None, np.inf
    for offset in (0.0, 0.5):
        shifted = v - offset
        lo = np.floor(shifted)
        for bits in itertools.product([0, 1], repeat=8):
            cand = lo + np.array(bits, dtype=np.float64)
            if int(round(cand.sum())) % 2 != 0:
                continue
            cand_full = cand + offset
            d = np.sum((v - cand_full) ** 2)
            if d < best_d:
                best_d, best = d, cand_full
    return best, best_d


def test_random_orthogonal_matrix_is_orthogonal():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 8, 17):
        r = _random_orthogonal_matrix(n, rng)
        assert np.allclose(r @ r.T, np.eye(n), atol=1e-8)
        assert np.allclose(r.T @ r, np.eye(n), atol=1e-8)


def test_closest_point_e8_exact_on_lattice_points():
    rng = np.random.default_rng(1)
    pts = rng.integers(-3, 4, size=(20, 8)).astype(np.float64)
    for i in range(len(pts)):
        if pts[i].sum() % 2 != 0:
            pts[i, 0] += 1  # land exactly on D8
    assert np.array_equal(_closest_point_e8(pts), pts)

    coset_pts = pts + 0.5  # land exactly on the D8 + 1/2 coset
    assert np.array_equal(_closest_point_e8(coset_pts), coset_pts)


def test_closest_point_e8_matches_brute_force_search():
    rng = np.random.default_rng(2)
    v = rng.uniform(-2, 2, size=(8, 8))
    found = _closest_point_e8(v)
    for i in range(len(v)):
        _, brute_d = _brute_force_closest_e8(v[i])
        found_d = np.sum((v[i] - found[i]) ** 2)
        assert found_d <= brute_d + 1e-9


def test_closest_point_d8_has_even_coordinate_sum():
    rng = np.random.default_rng(3)
    v = rng.uniform(-3, 3, size=(30, 8))
    d = _closest_point_d8(v)
    sums = np.sum(d, axis=-1)
    assert np.all(np.round(sums) == sums)  # all-integer
    assert np.all(sums.astype(np.int64) % 2 == 0)  # even


def _current_weight(model, weight_input_index=1):
    # apply_quip_sharp_cpp (which apply_quip_sharp now delegates to) folds
    # the entire rotate/quantize/rotate-back sandwich into a replacement
    # float32 initializer, rewiring the matched node's own weight input --
    # see onnxsim/quip_sharp.py's own "Delegates to" note.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_quip_sharp_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=16, seed=4)
    q = onnxsim.apply_quip_sharp(model)
    onnx.checker.check_model(q)

    # No new graph nodes -- folded straight into a replacement initializer.
    assert [n.op_type for n in q.graph.node] == [n.op_type for n in model.graph.node]

    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_quip_sharp_replaces_weight_with_same_shape_float():
    K, N = 32, 16
    model = _matmul_model(K=K, N=N, seed=4)
    q = onnxsim.apply_quip_sharp(model)
    new_w = _current_weight(q)
    orig_w = onnx.numpy_helper.to_array(model.graph.initializer[0])
    assert new_w.shape == orig_w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, orig_w)


def test_quip_sharp_unaffected_by_ort_graph_optimization_level():
    # Unlike DequantizeLinear-based weight-only quantization (see
    # onnxsim/ort_matmul_nbits_workaround.py), this module's own delegated
    # C++ implementation folds to a plain float32 initializer with no new
    # nodes at all, so it shouldn't trip ONNX Runtime's MatMulNBitsFusion
    # transformer.
    model = _matmul_model(K=32, N=16, seed=8)
    q = onnxsim.apply_quip_sharp(model)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((8, 32)).astype(np.float32)

    def run(level):
        so = ort.SessionOptions()
        so.graph_optimization_level = level
        sess = ort.InferenceSession(
            q.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
        )
        return sess.run(None, {"X": x})[0]

    y_off = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    y_on = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    assert np.allclose(y_off, y_on, rtol=0, atol=1e-5)


def test_quip_sharp_seed_and_epsilon_must_be_default():
    # apply_quip_sharp_cpp (which this function now delegates to)
    # hardcodes seed=0/epsilon=1e-8 and cannot honor other values.
    model = _matmul_model(K=32, N=16, seed=8)
    with pytest.raises(ValueError):
        onnxsim.apply_quip_sharp(model, seed=4)
    with pytest.raises(ValueError):
        onnxsim.apply_quip_sharp(model, epsilon=1e-6)


def test_quip_sharp_gemm_transb_with_bias():
    rng = np.random.default_rng(10)
    K, N = 40, 10
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_quip_sharp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((8, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_quip_sharp_skips_non_group_divisible_k():
    model = _matmul_model(K=20, N=4, seed=11)  # 20 is not a multiple of 8
    q = onnxsim.apply_quip_sharp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_quip_sharp_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_quip_sharp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_quip_sharp_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_quip_sharp(model)
    assert result.SerializeToString() == model.SerializeToString()
