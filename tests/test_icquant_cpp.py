"""Tests for ``onnxsim.apply_icquant_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_icquant`` (see ``onnxsim/passes/icquant.h``).
Like the other outlier-aware *_cpp ports in this repo, this scheme has no
accumulation or iterative-refinement step at all (see that header's own
"ACCEPTED, PERMANENT DIVERGENCE" note), so this port is expected to track
the pure-Python port unusually closely -- but these tests still check
structural/algebraic properties and comparable (not required to be
bit-for-bit identical) reconstruction error, matching this repo's own
established contract for a ``*_cpp`` port (``tests/test_hqq_cpp.py``).

Unlike ``quantize_weight_only_icquant`` (which builds a real INT4/
DequantizeLinear/ScatterND/MatMul/Add graph rewrite needing opset 21),
this port folds the round trip directly into a replacement float32
initializer -- no new graph nodes, no opset gate -- so these tests build
models at opset 21 anyway only where a direct comparison against the
Python port is needed (the C++ side itself works at any opset).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_GROUP_SIZE = 32
_NUM_OUTLIERS = 1
_MAX_CODE = 7


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


def _matmul_model(w, K, N, batch="batch", opset=21):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
        opset=opset,
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


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    K, N = _GROUP_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_icquant_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_outlier_reconstructs_exactly():
    # MatMul's weight is [K, N]; ICQuant's own blocks group K per output
    # channel (column n), so a "group" here is a column-strided slice.
    # Plant one huge outlier per group -- with num_outliers=1, it must be
    # excluded from the group's own scale and reconstructed exactly.
    rng = np.random.default_rng(1)
    K, N = _GROUP_SIZE * 3, 4
    w = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    outlier_positions = []
    for n in range(N):
        for start in range(0, K, _GROUP_SIZE):
            pos = start + int(rng.integers(0, _GROUP_SIZE))
            w[pos, n] = 50.0 if (pos // _GROUP_SIZE + n) % 2 == 0 else -50.0
            outlier_positions.append((pos, n))
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_icquant_cpp(model)
    new_w = _current_weight(q)
    for pos, n in outlier_positions:
        assert new_w[pos, n] == pytest.approx(w[pos, n], rel=1e-6)


def test_cpp_at_most_2n_plus_1_distinct_levels_per_group_excluding_outlier():
    # 7-level-per-side symmetric grid -> at most 2*7+1 = 15 distinct
    # non-outlier values per group.
    rng = np.random.default_rng(2)
    K, N = _GROUP_SIZE * 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_icquant_cpp(model)
    new_w = _current_weight(q)

    for n in range(N):
        for start in range(0, K, _GROUP_SIZE):
            block = new_w[start : start + _GROUP_SIZE, n]
            # Drop the single most-different-from-median value (the
            # outlier reconstructs exactly, so it need not land on the
            # shared grid) before checking the grid-size bound.
            deviations = np.abs(block.astype(np.float64) - np.median(block))
            trimmed = np.delete(block, np.argmax(deviations))
            assert len(np.unique(trimmed)) <= 2 * _MAX_CODE + 1


def test_cpp_beats_naive_single_scale_int4_on_outlier_heavy_blocks():
    # ICQuant's whole premise: excluding a group's own outlier from the
    # scale computation should let the remaining elements use a much
    # tighter (and thus more accurate) grid than a naive fit that lets
    # the outlier inflate the scale.
    rng = np.random.default_rng(3)
    K, N = _GROUP_SIZE * 4, 4
    w = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
    for n in range(N):
        for start in range(0, K, _GROUP_SIZE):
            w[start, n] = 30.0 if (start // _GROUP_SIZE) % 2 == 0 else -30.0
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_icquant_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    naive = np.empty_like(w64)
    for n in range(N):
        for start in range(0, K, _GROUP_SIZE):
            block = w64[start : start + _GROUP_SIZE, n]
            scale = max(np.abs(block).max(), 1e-12) / _MAX_CODE
            codes = np.clip(np.round(block / scale), -_MAX_CODE, _MAX_CODE)
            naive[start : start + _GROUP_SIZE, n] = codes * scale

    # Exclude the planted outlier position itself (both schemes reproduce
    # it about as well, since ICQuant reconstructs it exactly and the
    # naive scheme's own scale is dominated by it) and compare error on
    # the remaining "normal" elements.
    mask = np.ones_like(w64, dtype=bool)
    for n in range(N):
        for start in range(0, K, _GROUP_SIZE):
            mask[start, n] = False

    icquant_err = float(np.mean((w64[mask] - new_w[mask]) ** 2))
    naive_err = float(np.mean((w64[mask] - naive[mask]) ** 2))
    assert icquant_err < naive_err


def test_cpp_behaves_similarly_to_python_port():
    # Not required to be bit-for-bit identical (see passes/icquant.h's own
    # documented divergence note), but should reach a very similar
    # reconstruction error on the same input.
    rng = np.random.default_rng(4)
    K, N = _GROUP_SIZE * 4, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N, opset=21)

    py_q = onnxsim.quantize_weight_only_icquant(model)
    cpp_q = onnxsim.apply_icquant_cpp(model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py_q, {"X": x})
    (cpp_y,) = _run(cpp_q, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    assert _rel_l2(py_y, cpp_y) < 0.1


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _GROUP_SIZE * 2, 8
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
    q = onnxsim.apply_icquant_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = _GROUP_SIZE * 2, 6
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
    q = onnxsim.apply_icquant_cpp(model)
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
    result = onnxsim.apply_icquant_cpp(model)
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
    result = onnxsim.apply_icquant_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_group_divisible_weight():
    # K not a multiple of 32 -- icquant.py's own encoder skips this layer
    # entirely (no ragged-last-group handling), and this port matches
    # that exactly.
    rng = np.random.default_rng(9)
    K, N = _GROUP_SIZE + 5, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)
    result = onnxsim.apply_icquant_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
