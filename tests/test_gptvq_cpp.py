"""Tests for ``onnxsim.apply_gptvq_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_gptvq`` (see ``onnxsim/gptvq_entry.h`` and
``onnxsim/gptvq.py``). Like ``test_quarot_gptq_cpp.py``, this pass fits a
fresh k-means codebook per layer using its own independent RNG derivation
(not a numpy Generator sequenced across matches in graph node order), so
its output is expected to be *accurate*, not bit-identical to the Python
port -- most tests below check structure and numerical accuracy rather
than exact equality. The one exception is
``test_cpp_gptvq_matches_python_correction_math``, which plugs the C++
port's own fitted codebook and captured calibration activations into the
pure-Python GPTQ-style group correction to isolate that part of the
pipeline from the two ports' unrelated RNGs -- there, exact agreement is
expected and checked.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.gptvq import _gptvq_quantize_groups

ort = pytest.importorskip("onnxruntime")


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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _matmul_model(K=64, N=16, weight=None, seed=0):
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


def _correlated_calibration(K, num_samples=32, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_cpp_gptvq_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=32, N=8, seed=0)
    calib = _correlated_calibration(K=32, seed=2)
    q = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=0)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Gather", "Reshape", "Transpose"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_cpp_gptvq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=3)
    calib = _correlated_calibration(K=32, seed=4)
    q = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=3, num_centroids=64)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_gptvq_gemm_with_bias_leaves_bias_untouched():
    rng = np.random.default_rng(6)
    K, N = 32, 12
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
    calib = _correlated_calibration(K=K, seed=7)
    q = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=6)
    onnx.checker.check_model(q)

    gemm = next(n for n in q.graph.node if n.op_type == "Gemm")
    assert gemm.input[2] == "B"
    assert (
        onnx.numpy_helper.to_array(
            next(t for t in q.graph.initializer if t.name == "B")
        ).tolist()
        == bias.tolist()
    )

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_gptvq_skips_non_vector_dim_divisible_k():
    model = _matmul_model(K=33, N=8, seed=8)  # 33 is odd, not divisible by 2
    calib = _correlated_calibration(K=33, seed=9)
    q = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_gptvq_skips_layer_with_no_calibration_data():
    model = _matmul_model(K=32, N=8, seed=12)
    q = onnxsim.apply_gptvq_cpp(model, calibration_data=[], seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_gptvq_respects_skip_names():
    model = _matmul_model(K=32, N=8, seed=17)
    calib = _correlated_calibration(K=32, seed=18)
    q = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=0, skip_names=["W"])
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_gptvq_is_deterministic_for_a_given_seed():
    model = _matmul_model(K=32, N=8, seed=13)
    calib = _correlated_calibration(K=32, seed=14)
    q1 = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=42)
    q2 = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=42)
    assert q1.SerializeToString() == q2.SerializeToString()


def test_cpp_gptvq_different_seeds_give_different_codebooks():
    model = _matmul_model(K=32, N=8, seed=15)
    calib = _correlated_calibration(K=32, seed=16)
    q1 = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=1)
    q2 = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=2)
    assert q1.SerializeToString() != q2.SerializeToString()


def _codebook_and_codes(model, w_name):
    codebook = None
    codes = None
    for t in model.graph.initializer:
        if t.name.startswith(f"{w_name}_gptvq_codebook"):
            codebook = onnx.numpy_helper.to_array(t).astype(np.float64)
        elif t.name.startswith(f"{w_name}_gptvq_codes"):
            codes = onnx.numpy_helper.to_array(t).astype(np.int64)
    return codebook, codes


@pytest.mark.parametrize("vector_dim", [1, 2, 4])
def test_cpp_gptvq_matches_python_correction_math(vector_dim):
    # Cross-checks the C++ port's Hessian computation and group-correction
    # search against onnxsim.gptvq's own `_gptvq_quantize_groups` -- applied
    # to the *same* codebook and captured calibration activations the C++
    # port used -- this isolates the Hessian/correction math itself from
    # the two ports' unrelated, independently-seeded k-means RNGs (never a
    # cross-language parity goal for the codebook fit -- see this module's
    # own docstring).
    K, N = 32, 4
    rng = np.random.default_rng(300 + vector_dim)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)
    calib = _correlated_calibration(K=K, num_samples=40, rank=8, seed=400 + vector_dim)

    q = onnxsim.apply_gptvq_cpp(
        model, calibration_data=calib, seed=7, vector_dim=vector_dim, num_centroids=16
    )
    onnx.checker.check_model(q)

    codebook_cpp, codes_cpp = _codebook_and_codes(q, "W")
    assert codebook_cpp is not None
    assert codes_cpp is not None

    x = np.concatenate([batch["X"] for batch in calib], axis=0).astype(np.float64)
    w_nk = weight.astype(np.float64).T  # [N, K]
    h = x.T @ x

    codes_ref = _gptvq_quantize_groups(w_nk, codebook_cpp, h, 0.01, vector_dim)

    np.testing.assert_array_equal(codes_cpp, codes_ref)


def test_cpp_gptvq_codebook_fit_intentionally_diverges_from_python_for_same_seed():
    # Locks in the same documented, permanent divergence
    # test_quarot_gptq_cpp.py's own analogous test locks in: this port
    # fits its k-means codebook via a partial Fisher-Yates sample over
    # std::mt19937_64, while quantize_weight_only_gptvq (gptvq.py) uses
    # numpy.random.Generator.choice sequenced through a single Generator.
    # Not expected to ever alias for the same seed.
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=20)
    calib = _correlated_calibration(K=K, seed=21)

    py_q = onnxsim.quantize_weight_only_gptvq(model, calibration_data=calib, seed=123)
    cpp_q = onnxsim.apply_gptvq_cpp(model, calibration_data=calib, seed=123)

    codebook_py, _ = _codebook_and_codes(py_q, "W")
    codebook_cpp, _ = _codebook_and_codes(cpp_q, "W")
    assert codebook_py is not None
    assert codebook_cpp is not None
    assert codebook_py.shape == codebook_cpp.shape
    assert not np.allclose(codebook_py, codebook_cpp)
