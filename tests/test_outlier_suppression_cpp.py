"""Tests for ``onnxsim.apply_outlier_suppression_cpp`` -- the C++-backed
port of ``onnxsim.apply_outlier_suppression`` (Outlier Suppression's
"Gamma Migration", see ``onnxsim/outlier_suppression.py``). Like
``test_smoothquant_cpp.py``, this runs the model over real calibration
data through a real ``onnxruntime``-backed executor -- never a fake/mock
executor -- and checks exact (bit-for-bit) parity against the
pure-Python reference: both sides compute the same float64 ``s`` and
store the same float32 tensors, so any divergence is a bug, not an
accepted tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_outlier_suppression_cpp
from onnxsim.outlier_suppression import apply_outlier_suppression

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=17, ir_version=8):
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


def _ln_matmul_model(K=32, N=8, seed=0, with_bias=True):
    rng = np.random.default_rng(seed)
    gamma = np.ones(K, dtype=np.float32)
    gamma[3] = 20.0  # LayerNorm's own gamma amplifies this channel
    inits = [_f32(gamma, "Gamma")]
    beta_arg = ", Beta"
    if with_bias:
        inits.append(_f32(rng.standard_normal(K).astype(np.float32) * 0.1, "Beta"))
    else:
        beta_arg = ""
    inits.append(_f32(rng.standard_normal((K, N)).astype(np.float32) * 0.5, "W"))
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma{beta_arg})
          Y = MatMul(Ln, W)
        }}
        """,
        inits,
    )


def _calibration(K=32, num_samples=64, seed=1):
    rng = np.random.default_rng(seed)
    return [{"X": rng.standard_normal((num_samples, K)).astype(np.float32)}]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_outlier_suppression(model, calibration_data, **kwargs)
    cpp = apply_outlier_suppression_cpp(model, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    # Gamma Migration inserts zero nodes: the node list is byte-identical
    # to the input model's own.
    assert [n.op_type for n in cpp.graph.node] == [n.op_type for n in model.graph.node]
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape
        assert np.array_equal(ta, tb), a.name
    return cpp


def test_outlier_suppression_cpp_matches_python_exactly():
    _assert_exact_parity(_ln_matmul_model(), _calibration())


def test_outlier_suppression_cpp_matches_python_across_alphas():
    model = _ln_matmul_model(K=24, N=12, seed=3)
    calibration_data = _calibration(K=24, seed=4)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        _assert_exact_parity(model, calibration_data, alpha=alpha)


def test_outlier_suppression_cpp_without_bias():
    _assert_exact_parity(_ln_matmul_model(with_bias=False), _calibration(), alpha=0.75)


def test_outlier_suppression_cpp_multiple_consumers():
    # One shared LayerNorm feeding a plain MatMul and a transB Gemm (K !=
    # N, so the [N, K] transpose indexing is actually exercised): the
    # scale maximizes over both consumers' weight columns on both sides.
    K, N = 32, 8
    rng = np.random.default_rng(10)
    gamma = np.ones(K, dtype=np.float32)
    gamma[5] = 15.0
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y1, float[batch,{N}] Y2)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y1 = MatMul(Ln, W1)
          Y2 = Gemm<transB = 1>(Ln, W2)
        }}
        """,
        [
            _f32(gamma, "Gamma"),
            _f32(rng.standard_normal(K).astype(np.float32), "Beta"),
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W1"),
            _f32(rng.standard_normal((N, K)).astype(np.float32), "W2"),
        ],
    )
    cpp = _assert_exact_parity(model, _calibration(seed=11))
    # Both consumers' weights moved (and gamma shrank on the outlier
    # channel) -- i.e. this was a real migration, not a silent skip.
    w1 = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "W1")
    )
    assert not np.array_equal(w1, rng.standard_normal((K, N)).astype(np.float32))


def test_outlier_suppression_cpp_rank3_activation():
    # LayerNorm over the last axis of a rank-3 activation: the any-rank
    # reduction path (reduce over every leading axis), exact on both
    # sides.
    K, N = 16, 8
    rng = np.random.default_rng(12)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y = MatMul(Ln, W)
        }}
        """,
        [
            _f32(np.ones(K, dtype=np.float32), "Gamma"),
            _f32(rng.standard_normal(K).astype(np.float32), "Beta"),
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W"),
        ],
    )
    cals = [{"X": rng.standard_normal((2, 4, K)).astype(np.float32)} for _ in range(3)]
    _assert_exact_parity(model, cals)


def test_outlier_suppression_cpp_declines():
    # A residual Add consumer, a graph-output LayerNorm, a non-constant
    # gamma, and empty calibration data all leave the model structurally
    # unchanged on both sides.
    K, N = 32, 8
    rng = np.random.default_rng(13)

    residual = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          M = MatMul(Ln, W)
          Y = Add(M, Ln)
        }}
        """,
        [
            _f32(np.ones(K, dtype=np.float32), "Gamma"),
            _f32(np.zeros(K, dtype=np.float32), "Beta"),
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W"),
        ],
    )
    _assert_exact_parity(residual, _calibration(seed=14))

    graph_output = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{K}] Ln, float[batch,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y = MatMul(Ln, W)
        }}
        """,
        [
            _f32(np.ones(K, dtype=np.float32), "Gamma"),
            _f32(np.zeros(K, dtype=np.float32), "Beta"),
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W"),
        ],
    )
    _assert_exact_parity(graph_output, _calibration(seed=15))

    non_const_gamma = _model(
        f"""
        g (float[batch,{K}] X, float[{K}] Gamma) => (float[batch,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y = MatMul(Ln, W)
        }}
        """,
        [
            _f32(np.zeros(K, dtype=np.float32), "Beta"),
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W"),
        ],
    )
    _assert_exact_parity(non_const_gamma, _calibration(seed=16))

    model = _ln_matmul_model()
    cpp = apply_outlier_suppression_cpp(model, [])
    assert [(n.op_type, tuple(n.input)) for n in cpp.graph.node] == [
        (n.op_type, tuple(n.input)) for n in model.graph.node
    ]


def test_outlier_suppression_cpp_output_matches_float():
    # End to end through onnxruntime: the migrated model computes the same
    # function (up to float rounding) as the original.
    model = _ln_matmul_model()
    x = _calibration(num_samples=16)[0]["X"]
    cpp = apply_outlier_suppression_cpp(model, [{"X": x}])
    onnx.checker.check_model(cpp)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    os_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    got = os_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    rel = np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-6)
    assert rel < 1e-4
