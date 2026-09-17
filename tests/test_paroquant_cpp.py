"""Tests for ``onnxsim.apply_paroquant_cpp`` -- the C++-backed port of
ParoQuant's own SmoothQuant-scale-plus-pairwise-Givens-rotation
preprocessing (see ``onnxsim/paroquant_entry.h``). Like
``tests/test_spqr_cpp.py``, this runs the model over real calibration data
through a real ``onnxruntime``-backed executor -- never a fake/mock
executor.

``onnxsim.paroquant.apply_paroquant`` (the pure-Python entry point) is now
a thin wrapper around this same C++ function -- see
``onnxsim/paroquant.py`` and ``tests/test_paroquant.py`` (which exercises
the identical public contract through that wrapper). Unlike
``onnxsim.apply_spinquant_cpp``'s eigendecomposition, the per-pair Givens
angle search here has no RNG/LAPACK-equivalent algorithm choice to diverge
on (see ``paroquant_entry.h``'s own "ACCEPTED SCOPE" note), so this file
checks both the fitted rotation's own algebraic properties (orthogonal,
block-diagonal on adjacent pairs) and its numerical closeness to the float
reference.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_paroquant_cpp

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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


def _matmul_model(K=32, N=8, seed=0, opset=21):
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
        opset=opset,
    )


def _calibration(K=32, num_samples=16, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
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


def test_cpp_rotation_is_orthogonal_and_block_diagonal_pairwise():
    K, N, block_size = 16, 4, 4
    model = _matmul_model(K=K, N=N, seed=3)
    q = apply_paroquant_cpp(
        model, block_size=block_size, calibration_data=_calibration(K=K, seed=4)
    )
    onnx.checker.check_model(q)
    r_init = next(t for t in q.graph.initializer if t.name.endswith("_paroquant_r"))
    r = onnx.numpy_helper.to_array(r_init).astype(np.float64)
    assert r.shape == (K, K)
    np.testing.assert_allclose(r @ r.T, np.eye(K), atol=1e-6)

    for i in range(0, K, 2):
        j = i + 1
        assert np.isclose(r[i, i], r[j, j])
        assert np.isclose(r[i, j], -r[j, i])
        # A valid Givens rotation: cos^2 + sin^2 == 1.
        assert np.isclose(r[i, i] ** 2 + r[i, j] ** 2, 1.0)
        for other in range(K):
            if other in (i, j):
                continue
            assert r[i, other] == 0.0
            assert r[j, other] == 0.0
            assert r[other, i] == 0.0
            assert r[other, j] == 0.0


def test_cpp_weight_reconstruction_within_quantization_bound():
    K, N, block_size = 16, 4, 8
    model = _matmul_model(K=K, N=N, seed=8)
    q = apply_paroquant_cpp(
        model,
        block_size=block_size,
        alpha=0.5,
        num_angle_steps=9,
        calibration_data=_calibration(K=K, seed=9),
    )

    codes_init = next(
        t for t in q.graph.initializer if t.name.endswith("_paroquant_codes")
    )
    scale_init = next(
        t for t in q.graph.initializer if t.name.endswith("_paroquant_scale")
    )
    r_init = next(t for t in q.graph.initializer if t.name.endswith("_paroquant_r"))
    inv_s_init = next(
        t for t in q.graph.initializer if t.name.endswith("_paroquant_inv_scale")
    )

    codes_kn = onnx.numpy_helper.to_array(codes_init).astype(np.float64)
    scale_blocks_kn = onnx.numpy_helper.to_array(scale_init).astype(np.float64)
    r = onnx.numpy_helper.to_array(r_init).astype(np.float64)
    inv_s = onnx.numpy_helper.to_array(inv_s_init).astype(np.float64)

    scale_full_kn = np.repeat(scale_blocks_kn, block_size, axis=0)
    w_tilde_kn = codes_kn * scale_full_kn

    orig_w = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W")
    ).astype(np.float64)
    s = 1.0 / inv_s
    w_smooth_nk = orig_w.T * s[np.newaxis, :]
    w_tilde_expected_nk = w_smooth_nk @ r  # exact before quantization

    err = np.abs(w_tilde_kn.T - w_tilde_expected_nk)
    half_step = 0.5 * scale_full_kn.T
    assert np.all(err <= half_step * (1.0 + 1e-3) + 1e-6)


def test_cpp_output_stays_close_to_float():
    K, N, block_size = 32, 8, 8
    model = _matmul_model(K=K, N=N, seed=0)
    q = apply_paroquant_cpp(
        model, block_size=block_size, calibration_data=_calibration(K=K, seed=1)
    )
    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2
    assert "Mul" in op_types
    assert "DequantizeLinear" in op_types

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(5)
    K, N, block_size = 32, 8, 8
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = apply_paroquant_cpp(
        model, block_size=block_size, calibration_data=_calibration(K=K, seed=6)
    )
    onnx.checker.check_model(q)
    assert any(
        n.op_type == "Add" and n.name.endswith("_bias_add_node") for n in q.graph.node
    )

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)
    q = apply_paroquant_cpp(
        model, block_size=8, calibration_data=_calibration(K=20, seed=7)
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_declines_odd_block_size():
    model = _matmul_model(K=32, N=8, seed=10)
    q = apply_paroquant_cpp(
        model, block_size=7, calibration_data=_calibration(K=32, seed=10)
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = apply_paroquant_cpp(model, calibration_data=_calibration(K=32, seed=8))
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_paroquant_cpp(
        model, calibration_data=[{"X": np.zeros((2, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_below_opset21():
    model = _matmul_model(K=32, N=8, seed=9, opset=13)
    result = apply_paroquant_cpp(model, calibration_data=_calibration(K=32, seed=9))
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_missing_graph_input_raises_value_error():
    model = _matmul_model(K=16, N=4, seed=10)
    with pytest.raises(ValueError):
        apply_paroquant_cpp(
            model,
            block_size=8,
            calibration_data=[{"WrongName": np.zeros((2, 16), dtype=np.float32)}],
        )


def test_cpp_empty_calibration_data_leaves_model_unchanged():
    model = _matmul_model(K=16, N=4, seed=11)
    result = apply_paroquant_cpp(model, block_size=8, calibration_data=[])
    assert result.SerializeToString() == model.SerializeToString()
