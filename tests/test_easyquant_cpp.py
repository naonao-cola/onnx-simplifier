"""Tests for ``onnxsim.apply_easyquant_cpp`` -- the C++-backed port of
``onnxsim.apply_easyquant`` (EasyQuant's coordinate-descent W8A8 scale
search, see ``onnxsim/easyquant.py``). Like ``test_smoothquant_cpp.py``,
this runs the model over real calibration data through a real
``onnxruntime``-backed executor -- never a fake/mock executor.

``onnxsim.apply_easyquant`` now delegates to this C++ port (see
``onnxsim/easyquant.py``), so ``_assert_exact_parity`` below is a wiring/
reproducibility regression test, not an independent numerical cross-
check. The genuinely independent check
(``test_easyquant_cpp_search_beats_naive_max_abs_scale``) reuses
``test_easyquant.py``'s own approach: a plain, unsearched W8A8 round-trip
computed directly with numpy in this file (no onnxsim code at all) as the
baseline EasyQuant's own coordinate-descent search must beat.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.easyquant import apply_easyquant
from onnxsim.onnx_simplifier import apply_easyquant_cpp

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


def _outlier_calibration(K=32, num_samples=8, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, 3] *= 15.0
    return [{"X": x}]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_easyquant(model, calibration_data, **kwargs)
    cpp = apply_easyquant_cpp(model, calibration_data, **kwargs)
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


def test_easyquant_cpp_matches_python_exactly():
    model = _matmul_model()
    _assert_exact_parity(model, _outlier_calibration())


def test_easyquant_cpp_matches_python_across_search_params():
    model = _matmul_model(K=24, N=6, seed=2)
    calibration_data = _outlier_calibration(K=24, seed=3)
    for num_iterations, num_candidates, search_span in [
        (1, 5, 0.5),
        (2, 11, 0.3),
        (4, 21, 0.9),
    ]:
        _assert_exact_parity(
            model,
            calibration_data,
            num_iterations=num_iterations,
            num_candidates=num_candidates,
            search_span=search_span,
        )


def test_easyquant_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 20, 6
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    _assert_exact_parity(model, _outlier_calibration(K=K, seed=9))


def test_easyquant_cpp_biased_gemm():
    rng = np.random.default_rng(14)
    K, N = 16, 4
    b = rng.standard_normal((N,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, N)), "W"),
            _f32(b, "B"),
        ],
    )
    cpp = _assert_exact_parity(model, _outlier_calibration(K=K, seed=15))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    assert np.array_equal(b_new, b)


def test_easyquant_cpp_shared_activation_gets_independent_rewrites():
    K, N = 16, 4
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
    cpp = _assert_exact_parity(model, _outlier_calibration(K=K, seed=11))
    onnx.checker.check_model(cpp)


def test_easyquant_cpp_multi_batch_and_3d_activation():
    K, N = 16, 4
    model = _matmul_model(K=K, N=N, seed=12)
    rng = np.random.default_rng(13)
    cals = [
        {"X": rng.standard_normal((4, K)).astype(np.float32)},
        {"X": rng.standard_normal((6, K)).astype(np.float32)},
    ]
    _assert_exact_parity(model, cals)

    model3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(rng.standard_normal((K, N)).astype(np.float32), "W")],
    )
    cals3d = [{"X": rng.standard_normal((2, 3, K)).astype(np.float32)}]
    cpp = _assert_exact_parity(model3d, cals3d)
    onnx.checker.check_model(cpp)


def test_easyquant_cpp_skips():
    model = _matmul_model()
    assert len(apply_easyquant_cpp(model, []).graph.node) == len(model.graph.node)

    non_const = _model(
        """
        g (float[batch,16] X, float[16,4] W) => (float[batch,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    out = apply_easyquant_cpp(non_const, _outlier_calibration(K=16))
    assert out.SerializeToString() == non_const.SerializeToString()


def test_easyquant_cpp_output_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=16)
    x = _outlier_calibration(K=32, num_samples=32, seed=17)[0]["X"]
    cpp = apply_easyquant_cpp(model, [{"X": x}])
    onnx.checker.check_model(cpp)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    eq_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    got = eq_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    assert np.all(np.isfinite(got))
    rel = np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-6)
    assert rel < 0.2


def test_easyquant_cpp_missing_calibration_input_raises():
    model = _matmul_model()
    with pytest.raises(Exception):
        apply_easyquant_cpp(model, [{"NotX": np.zeros((1, 32), dtype=np.float32)}])


def _naive_w8a8_round_trip(model, x):
    # A plain, unsearched W8A8 baseline (one max-abs scale per output
    # channel for the weight, one max-abs scale for the whole activation
    # tensor -- exactly EasyQuant's own starting point before any search),
    # computed directly with numpy -- no onnxsim code, independent of both
    # apply_easyquant and apply_easyquant_cpp. Mirrors test_easyquant.py's
    # own identical helper.
    w = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_scale = np.maximum(np.max(np.abs(w), axis=0), 1e-12) / 127.0  # [N], W is [K,N]
    a_scale = max(float(np.max(np.abs(x))), 1e-12) / 127.0
    w_q = np.clip(np.round(w / w_scale), -127, 127) * w_scale
    x_q = np.clip(np.round(x / a_scale), -127, 127) * a_scale
    return x_q @ w_q


def _outlier_calibration_3d(K, num_samples, batch=32, seed=0):
    # Mirrors test_easyquant.py's own identical helper: a fixed (K-derived,
    # not seed-derived) subset of channels is a 15x-larger outlier, so the
    # naive single-shot max-abs scale (dominated by those channels) wastes
    # most of its resolution -- exactly the scenario a per-channel
    # searched scale has real room to improve on.
    outlier_rng = np.random.default_rng(1234 + K)
    outlier_channels = outlier_rng.choice(K, size=max(1, K // 8), replace=False)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, batch, K)).astype(np.float32)
    x[:, :, outlier_channels] *= 15.0
    return [{"X": x[i]} for i in range(num_samples)]


def test_easyquant_cpp_search_beats_naive_max_abs_scale():
    # Genuinely independent check, mirroring test_easyquant.py's own
    # identical test (now against the C++ port directly, since
    # apply_easyquant itself delegates to it): the coordinate-descent
    # search apply_easyquant_cpp actually runs must reconstruct the float
    # output more closely, on held-out data, than the naive one-shot
    # max-abs scale it starts from.
    rng = np.random.default_rng(2)
    K, N = 32, 12
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N)
    model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(w, "W"))

    fit_data = _outlier_calibration_3d(K, num_samples=16, batch=64, seed=3)
    cpp = apply_easyquant_cpp(model, fit_data, num_iterations=3, num_candidates=25)
    onnx.checker.check_model(cpp)

    held_out = _outlier_calibration_3d(K, num_samples=1, batch=512, seed=4)[0]["X"]
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    cpp_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    float_y = sess.run(["Y"], {"X": held_out})[0].astype(np.float64)
    easyquant_y = cpp_sess.run(["Y"], {"X": held_out})[0].astype(np.float64)
    naive_y = _naive_w8a8_round_trip(model, held_out.astype(np.float64))

    def _rel_l2(a, b):
        a, b = a.ravel(), b.ravel()
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)

    easyquant_err = _rel_l2(float_y, easyquant_y)
    naive_err = _rel_l2(float_y, naive_y)
    assert easyquant_err < naive_err
