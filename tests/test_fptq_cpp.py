"""Tests for ``onnxsim.apply_fptq_cpp`` -- the C++-backed port of
``onnxsim.apply_fptq`` (FPTQ's logarithmic-equalization migration layered
on :mod:`onnxsim.smoothquant`'s own power-law core, see
``onnxsim/fptq.py``). Like ``test_smoothquant_cpp.py``, this runs the
model over real calibration data through a real ``onnxruntime``-backed
executor -- never a fake/mock executor.

``onnxsim.apply_fptq`` now delegates to this C++ port (see
``onnxsim/fptq.py``), so ``_assert_exact_parity`` below is a wiring/
reproducibility regression test, not an independent numerical cross-
check. The genuinely independent check is against
``onnxsim.apply_smoothquant_cpp``: FPTQ's own "tractable" branch is, by
construction, the identical power-law formula SmoothQuant already
computes, so on a layer whose calibration data never crosses the
outlier-ratio threshold, the two passes must migrate the SAME per-channel
scale (verified below by comparing the resulting weight tensors, up to
the different node/initializer naming each pass mints).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.fptq import apply_fptq
from onnxsim.onnx_simplifier import apply_fptq_cpp, apply_smoothquant_cpp

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


def _tractable_calibration(K=64, num_samples=8, seed=1):
    # Mild per-channel variation: outlier_ratio stays well under the
    # default threshold (10.0), so this takes the plain power-law branch.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, 3] *= 2.0
    x[:, 7] *= 1.5
    return [{"X": x}]


def _intractable_calibration(K=64, num_samples=8, seed=1):
    # One channel sits ~1000x above the rest -- forces outlier_ratio far
    # past the default 10.0 threshold, so this takes the logarithmic
    # branch.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32) * 0.1
    x[:, 5] = 100.0 + rng.standard_normal(num_samples).astype(np.float32)
    return [{"X": x}]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_fptq(model, calibration_data, **kwargs)
    cpp = apply_fptq_cpp(model, calibration_data, **kwargs)
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
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape
        assert np.array_equal(ta, tb), a.name
    return cpp


def test_fptq_cpp_matches_python_exactly_tractable_layer():
    model = _matmul_model()
    _assert_exact_parity(model, _tractable_calibration())


def test_fptq_cpp_matches_python_exactly_intractable_layer():
    model = _matmul_model()
    _assert_exact_parity(model, _intractable_calibration())


def test_fptq_cpp_matches_python_across_thresholds_and_alphas():
    model = _matmul_model(K=32, N=24, seed=3)
    calibration_data = _intractable_calibration(K=32, seed=4)
    for alpha, threshold in [(0.0, 10.0), (0.5, 5.0), (0.75, 50.0), (1.0, 2.0)]:
        _assert_exact_parity(
            model, calibration_data, alpha=alpha, outlier_ratio_threshold=threshold
        )


def test_fptq_cpp_gemm_variants():
    rng = np.random.default_rng(5)
    K, N = 48, 12
    cases = [((K, N), False), ((N, K), True)]
    for i, (shape, transb) in enumerate(cases):
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
        _assert_exact_parity(model, _intractable_calibration(K=K, seed=6 + i))
    # Biased Gemm: the bias rides along untouched on both sides.
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
    cpp = _assert_exact_parity(model, _intractable_calibration(K=K, seed=9))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    assert np.array_equal(b_new, b)


def test_fptq_cpp_shared_activation_gets_independent_muls():
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
    _assert_exact_parity(model, _intractable_calibration(K=K, seed=11))


def test_fptq_cpp_skips():
    model = _matmul_model()
    assert len(apply_fptq_cpp(model, []).graph.node) == len(model.graph.node)

    non_const = _model(
        """
        g (float[batch,64] X, float[64,16] W) => (float[batch,16] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    _assert_exact_parity(non_const, _intractable_calibration())

    rank3 = _model(
        """
        g (float[batch,seq,64] X) => (float[batch,seq,16] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[_f32(np.random.default_rng(12).standard_normal((64, 16)), "W")],
    )
    rank3_cals = [
        {"X": np.random.default_rng(12).standard_normal((2, 4, 64)).astype(np.float32)}
    ]
    _assert_exact_parity(rank3, rank3_cals)


def test_fptq_cpp_all_zero_channel_uses_epsilon():
    K, N = 16, 8
    model = _matmul_model(K=K, N=N, seed=13)
    rng = np.random.default_rng(14)
    x = rng.standard_normal((4, K)).astype(np.float32)
    x[:, 0] = 0.0
    cpp = _assert_exact_parity(model, [{"X": x}])
    for t in cpp.graph.initializer:
        assert np.all(np.isfinite(onnx.numpy_helper.to_array(t)))


def test_fptq_cpp_output_matches_float():
    model = _matmul_model()
    x = _intractable_calibration()[0]["X"]
    cpp = apply_fptq_cpp(model, [{"X": x}])
    onnx.checker.check_model(cpp)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    fq_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    got = fq_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    rel = np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-6)
    assert rel < 1e-4


def test_fptq_cpp_missing_calibration_input_raises():
    model = _matmul_model()
    with pytest.raises(Exception):
        apply_fptq_cpp(model, [{"NotX": np.zeros((1, 64), dtype=np.float32)}])


def test_fptq_cpp_tractable_layer_migrates_same_weight_as_smoothquant():
    # Genuinely independent check: onnxsim.apply_smoothquant_cpp is a
    # separate C++ port of a separate technique, not delegated from or to
    # apply_fptq_cpp. On a "tractable" layer (outlier_ratio below the
    # threshold), FPTQ's own scale IS SmoothQuant's own power-law scale --
    # so for the same alpha, the two passes must produce numerically
    # identical rescaled weight tensors (names differ; values must not).
    model = _matmul_model(K=32, N=8, seed=50)
    calibration_data = _tractable_calibration(K=32, seed=51)
    alpha = 0.5

    fptq_out = apply_fptq_cpp(model, calibration_data, alpha=alpha)
    sq_out = apply_smoothquant_cpp(model, calibration_data, alpha=alpha)

    fptq_w = onnx.numpy_helper.to_array(
        next(t for t in fptq_out.graph.initializer if t.name == "W")
    )
    sq_w = onnx.numpy_helper.to_array(
        next(t for t in sq_out.graph.initializer if t.name == "W")
    )
    np.testing.assert_array_equal(fptq_w, sq_w)
