"""Tests for ``onnxsim.apply_adpq_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_adpq`` (see ``onnxsim/passes/adpq.h``).
Unlike the Python side, which builds a real
``DequantizeLinear(codes, scale)``+``ScatterND``(salient correction)+
``Add`` graph rewrite with packed INT4 codes (needing opset 21+), this
port folds the whole reconstruction directly into a replacement float32
initializer (see that header's own "ACCEPTED, PERMANENT DIVERGENCE"
note) and needs no opset gate at all -- so cross-checking against the
Python port is done end-to-end through onnxruntime rather than by
comparing initializers directly. This scheme has no accumulation/
iterative-refinement step, so this port is expected to track the Python
port's own float64 numpy implementation closely, up to floating-point
median-tie-breaking/summation-order differences -- comparable, not
required to be bit-for-bit identical, matching this repo's established
contract for every other data-free ``*_cpp`` port.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.adpq import quantize_weight_only_adpq

ort = pytest.importorskip("onnxruntime")

_GROUP_SIZE = 128


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
    # The C++ pass (like every sibling data-free *_cpp port) rewires the
    # matched node's weight input to a freshly created initializer,
    # leaving the original one dangling unused in the graph -- so the
    # *node's own current input name* is the only reliable way to find
    # the actual (post-quantization) weight, not initializer list
    # position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    K, N = _GROUP_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_adpq_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # every sibling data-free *_cpp port's established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_salient_outlier_round_trips_exactly():
    # A group of small Gaussian noise plus one huge planted outlier: the
    # outlier's magnitude clears the group's own adaptive threshold by a
    # wide margin (small noise -> small robust sigma -> small threshold),
    # so it must round-trip byte-exact (adpq.py's own ScatterND
    # correction reconstructs salient elements exactly; this port simply
    # leaves them untouched instead).
    rng = np.random.default_rng(1)
    K, N = _GROUP_SIZE, 2
    w = (rng.standard_normal((K, N)) * 0.01).astype(np.float32)
    w[5, 0] = 50.0
    w[10, 1] = -37.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_adpq_cpp(model)
    new_w = _current_weight(q)
    assert new_w[5, 0] == np.float32(50.0)
    assert new_w[10, 1] == np.float32(-37.5)
    # The bulk of the group (small noise, well under threshold) should
    # have actually been quantized, not merely copied.
    assert not np.array_equal(new_w[:, 0], w[:, 0])


def test_cpp_non_salient_at_most_15_distinct_levels_per_group():
    rng = np.random.default_rng(2)
    K, N = _GROUP_SIZE * 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_adpq_cpp(model)
    new_w = _current_weight(q)

    # MatMul's weight is [K, N]; AdpQ's own groups are along K per output
    # channel (column n), so a "group" here is a column-strided slice.
    # Ordinary standard-normal data at this scale should have no salient
    # outliers at all, so every group's own values should collapse onto
    # the signed 4-bit grid (at most 15 distinct levels).
    for n in range(N):
        col = new_w[:, n]
        for start in range(0, K, _GROUP_SIZE):
            group = col[start : start + _GROUP_SIZE]
            assert len(np.unique(group)) <= 15


def test_cpp_behaves_similarly_to_python_port_end_to_end():
    # The C++ port folds to a plain float32 initializer while the Python
    # port builds a real DequantizeLinear/ScatterND/Add subgraph --
    # compare them end-to-end through onnxruntime rather than by
    # inspecting initializers directly.
    rng = np.random.default_rng(3)
    K, N = _GROUP_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    py_q = quantize_weight_only_adpq(model)
    cpp_q = onnxsim.apply_adpq_cpp(model)
    onnx.checker.check_model(cpp_q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py_q, {"X": x})
    (cpp_y,) = _run(cpp_q, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    assert _rel_l2(py_y, cpp_y) < 0.05


def test_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(4)
    K, N = _GROUP_SIZE * 2, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5  # transB=1 layout
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_adpq_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_noop_when_k_not_divisible_by_group_size():
    rng = np.random.default_rng(5)
    K, N = _GROUP_SIZE + 5, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_adpq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_adpq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(6)
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
    result = onnxsim.apply_adpq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
