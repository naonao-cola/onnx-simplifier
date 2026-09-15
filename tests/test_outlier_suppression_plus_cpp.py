"""Tests for ``onnxsim.apply_outlier_suppression_plus_cpp`` -- the
C++-backed port of ``onnxsim.apply_outlier_suppression_plus`` (Outlier
Suppression+'s channel-wise shifting and scaling, see
``onnxsim/outlier_suppression_plus.py``). Like
``test_smoothquant_cpp.py``, this runs the model over real calibration
data through a real ``onnxruntime``-backed executor -- never a fake/mock
executor -- and checks exact (bit-for-bit) parity against the
pure-Python reference: both sides compute the same float64 ``z``/``s``
and store the same float32 tensors, so any divergence is a bug, not an
accepted tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_outlier_suppression_plus_cpp
from onnxsim.outlier_suppression_plus import apply_outlier_suppression_plus

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _lopsided_calibration(K=64, num_samples=32, seed=1):
    # Outlier Suppression+'s own motivating scenario: channels sitting
    # mostly on one side of zero, which a symmetric scale alone cannot fix.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, 3] = x[:, 3] * 5.0 + 18.0
    x[:, 7] = x[:, 7] * 4.0 - 12.0
    return [{"X": x}]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_outlier_suppression_plus(model, calibration_data, **kwargs)
    cpp = apply_outlier_suppression_plus_cpp(model, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    # Each migrated layer gains exactly one Sub, one Mul and one Add;
    # skipped layers gain nothing -- the count agrees by construction of
    # the exact node-list equality above, so just spell out the shape.
    for op in ("Sub", "Mul", "Add"):
        assert sum(1 for n in cpp.graph.node if n.op_type == op) == sum(
            1 for n in py.graph.node if n.op_type == op
        )
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape
        assert np.array_equal(ta, tb), a.name
    return cpp


def test_outlier_suppression_plus_cpp_matches_python_exactly():
    _assert_exact_parity(_matmul_model(), _lopsided_calibration())


def test_outlier_suppression_plus_cpp_matches_python_across_alphas():
    model = _matmul_model(K=32, N=24, seed=3)
    calibration_data = _lopsided_calibration(K=32, seed=4)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        _assert_exact_parity(model, calibration_data, alpha=alpha)


def test_outlier_suppression_plus_cpp_gemm_variants():
    # Plain Gemm, transB Gemm (K != N, so the [N, K] transpose indexing is
    # actually exercised), and biased Gemm -- all must match exactly.
    rng = np.random.default_rng(5)
    K, N = 48, 12
    for i, (shape, transb) in enumerate([((K, N), False), ((N, K), True)]):
        w = rng.standard_normal(shape).astype(np.float32)
        attrs = " <transB = 1>" if transb else ""
        model = _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{N}] Y)
            {{
              Y = Gemm{attrs}(X, W)
            }}
            """,
            initializer=[_f32(w, "W")],
        )
        _assert_exact_parity(model, _lopsided_calibration(K=K, seed=6 + i))
    b = rng.standard_normal((N,)).astype(np.float32)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
    )
    cpp = _assert_exact_parity(model, _lopsided_calibration(K=K, seed=9))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    assert np.array_equal(b_new, b)


def test_outlier_suppression_plus_cpp_shared_activation():
    # Two MatMuls sharing one activation: each gets its own Sub/Mul/Add
    # triple (the second with _1-suffixed names), exactly like Python.
    K, N = 32, 8
    rng = np.random.default_rng(10)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y1, float[batch,{N}] Y2)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(X, W2)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, N)), "W1"),
            _f32(rng.standard_normal((K, N)), "W2"),
        ],
    )
    _assert_exact_parity(model, _lopsided_calibration(K=K, seed=11))


def test_outlier_suppression_plus_cpp_all_negative_channel():
    # A channel entirely below zero: its maximum is negative, not zero --
    # seeding the running max against 0.0 would record the wrong range.
    K, N = 16, 8
    model = _matmul_model(K=K, N=N, seed=13)
    rng = np.random.default_rng(14)
    x = rng.standard_normal((4, K)).astype(np.float32)
    x[:, 0] = -5.0 - np.abs(x[:, 0])
    cpp = _assert_exact_parity(model, [{"X": x}])
    z = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "X_os_plus_shift")
    )
    assert z[0] < 0.0
    for t in cpp.graph.initializer:
        assert np.all(np.isfinite(onnx.numpy_helper.to_array(t)))


def test_outlier_suppression_plus_cpp_multi_batch_ranges():
    # Max/min accumulate across batches elementwise: the peak in each
    # direction may come from a different batch.
    K, N = 16, 8
    model = _matmul_model(K=K, N=N, seed=15)
    rng = np.random.default_rng(16)
    x1 = rng.standard_normal((4, K)).astype(np.float32)
    x2 = rng.standard_normal((4, K)).astype(np.float32)
    x1[:, 1] += 10.0
    x2[:, 1] -= 10.0
    _assert_exact_parity(model, [{"X": x1}, {"X": x2}])


def test_outlier_suppression_plus_cpp_skips():
    # Non-constant weight, rank-3 activation, and empty calibration data
    # all leave the model structurally unchanged on both sides.
    model = _matmul_model()
    assert len(apply_outlier_suppression_plus_cpp(model, []).graph.node) == len(
        model.graph.node
    )

    non_const = _model(
        """
        g (float[batch,64] X, float[64,16] W) => (float[batch,16] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    _assert_exact_parity(non_const, _lopsided_calibration())

    rng = np.random.default_rng(12)
    rank3 = _model(
        """
        g (float[batch,seq,64] X) => (float[batch,seq,16] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[_f32(rng.standard_normal((64, 16)), "W")],
    )
    rank3_cals = [{"X": rng.standard_normal((2, 4, 64)).astype(np.float32)}]
    _assert_exact_parity(rank3, rank3_cals)


def test_outlier_suppression_plus_cpp_output_matches_float():
    # End to end through onnxruntime: the migrated model computes the same
    # function (up to float rounding) as the original.
    model = _matmul_model()
    x = _lopsided_calibration(num_samples=16)[0]["X"]
    cpp = apply_outlier_suppression_plus_cpp(model, [{"X": x}])
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
