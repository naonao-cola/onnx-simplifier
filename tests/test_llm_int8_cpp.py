"""Tests for ``onnxsim.apply_llm_int8_cpp`` -- the C++-backed port of
``onnxsim.apply_llm_int8`` (LLM.int8()'s outlier/float32 + vector-wise
INT8 decomposition, see ``onnxsim/llm_int8.py``). Like
``test_outlier_suppression_plus_cpp.py``, this runs the model over real
calibration data through a real ``onnxruntime``-backed executor -- never
a fake/mock executor -- and checks exact (bit-for-bit) parity against
the pure-Python reference: both sides compute the same outlier sets,
scales and INT8 codes and emit the same graph, so any divergence is a
bug, not an accepted tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.llm_int8 import apply_llm_int8
from onnxsim.onnx_simplifier import apply_llm_int8_cpp

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=18, ir_version=10):
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


def _matmul_model(K=32, N=8, seed=0):
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


def _outlier_calibration(K=32, num_samples=16, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, 3] *= 20.0
    return [{"X": x}]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_llm_int8(model, calibration_data, **kwargs)
    cpp = apply_llm_int8_cpp(model, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape, a.name
        assert np.array_equal(ta, tb), a.name
    return cpp


def test_llm_int8_cpp_matches_python_exactly():
    cpp = _assert_exact_parity(_matmul_model(), _outlier_calibration())
    # The decomposition shape: outlier Gather/MatMul plus the INT8 chain.
    ops = sorted(n.op_type for n in cpp.graph.node)
    assert "MatMulInteger" in ops
    assert ops.count("Gather") == 2
    assert ops.count("Cast") == 2


def test_llm_int8_cpp_rounds_ties_to_even():
    # Exact .5 quotients must take the even neighbor (banker's rounding,
    # matching numpy and the ONNX Round op), not half-away-from-zero.
    K, N = 4, 2
    col = np.array([0.5, 1.5, 2.5, 127.0], dtype=np.float32)
    w = np.stack([col, col / 2], axis=1)
    model = _matmul_model(K=K, N=N, seed=0)
    model.graph.initializer.clear()
    model.graph.initializer.extend([_f32(w, "W")])
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, K)).astype(np.float32)
    x[:, 0] *= 20.0  # outlier channel, so the layer actually decomposes
    cals = [{"X": x}]
    cpp = _assert_exact_parity(model, cals)
    wq = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name.endswith("_wq_regular"))
    )
    # scale = 127/127 = 1 on output 0 (regular entries [1.5, 2.5, 127])
    # and 0.5 on output 1 ([0.75, 1.25, 63.5] -> [1.5, 2.5, 127]):
    # exact .5 quotients take the even neighbor on both sides.
    assert list(wq[:, 0]) == [2, 2, 127]
    assert list(wq[:, 1]) == [2, 2, 127]


def test_llm_int8_cpp_gemm_variants():
    # Plain Gemm, transB Gemm (K != N), and biased Gemm (bias rides into
    # the combining Add untouched) -- all must match exactly.
    rng = np.random.default_rng(5)
    K, N = 24, 6
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
        _assert_exact_parity(model, _outlier_calibration(K=K, seed=6 + i))
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
    cpp = _assert_exact_parity(model, _outlier_calibration(K=K, seed=9))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    assert np.array_equal(b_new, b)


def test_llm_int8_cpp_threshold_boundary():
    # A channel at exactly the threshold is regular (strict > marks
    # outliers); just above it is an outlier.
    K, N = 8, 4
    model = _matmul_model(K=K, N=N, seed=10)
    x = np.zeros((4, K), dtype=np.float32)
    x[:, 0] = 6.0
    x[:, 1] = 6.0001
    cpp = _assert_exact_parity(model, [{"X": x}])
    outlier_idx = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name.endswith("_outlier_idx"))
    )
    assert list(outlier_idx) == [1]


def test_llm_int8_cpp_skips():
    # No outliers, all outliers, non-constant weight, rank-3 activation,
    # pre-18 opset, and empty calibration data all leave the layer (or
    # the whole model) structurally unchanged on both sides.
    model = _matmul_model()
    quiet = [{"X": np.zeros((4, 32), dtype=np.float32)}]
    for cals in (quiet, []):
        for fn in (apply_llm_int8, apply_llm_int8_cpp):
            assert [n.op_type for n in fn(model, cals).graph.node] == ["MatMul"]

    loud = np.full((4, 32), 10.0, dtype=np.float32)
    for fn in (apply_llm_int8, apply_llm_int8_cpp):
        assert [n.op_type for n in fn(model, [{"X": loud}]).graph.node] == ["MatMul"]

    non_const = _model(
        """
        g (float[batch,32] X, float[32,8] W) => (float[batch,8] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        opset=18,
    )
    _assert_exact_parity(non_const, _outlier_calibration())

    rng = np.random.default_rng(12)
    rank3 = _model(
        """
        g (float[batch,seq,32] X) => (float[batch,seq,8] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[_f32(rng.standard_normal((32, 8)), "W")],
        opset=18,
    )
    rank3_cals = [{"X": rng.standard_normal((2, 4, 32)).astype(np.float32)}]
    _assert_exact_parity(rank3, rank3_cals)

    old = _matmul_model()
    old.opset_import.clear()
    oi = old.opset_import.add()
    oi.domain = ""
    oi.version = 17
    for fn in (apply_llm_int8, apply_llm_int8_cpp):
        assert [n.op_type for n in fn(old, _outlier_calibration()).graph.node] == [
            "MatMul"
        ]


def test_llm_int8_cpp_output_matches_float():
    # End to end through onnxruntime: the decomposed layer (outlier
    # float32 part + INT8 part, output name preserved) computes close to
    # the original -- and the MatMulInteger chain actually executes.
    model = _matmul_model()
    x = _outlier_calibration(num_samples=16)[0]["X"]
    cpp = apply_llm_int8_cpp(model, [{"X": x}])
    onnx.checker.check_model(cpp)
    assert [o.name for o in cpp.graph.output] == ["Y"]
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    q_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    got = q_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    assert np.all(np.isfinite(got))
    rel = np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-6)
    assert rel < 5e-2
