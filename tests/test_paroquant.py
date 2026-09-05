"""Tests for ``onnxsim.apply_paroquant`` -- see ``onnxsim/paroquant.py`` for
the technique (a SmoothQuant-style channel scale plus a block-diagonal
*pairwise* (Givens) rotation -- many independent 2x2 rotations, unlike
``onnxsim/spinquant.py``'s single dense ``[K, K]`` rotation -- fit via a
small per-pair grid search, followed by block-wise INT4 quantization).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


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


def _matmul_model(K=32, N=8, seed=0, opset=21):
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
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_paroquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_paroquant(model, block_size=8, num_samples=16, seed=1)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2
    assert "Mul" in op_types
    assert "DequantizeLinear" in op_types

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_paroquant_rotation_matrix_is_orthogonal_and_block_diagonal_pairwise():
    model = _matmul_model(K=16, N=4, seed=3)
    q = onnxsim.apply_paroquant(model, block_size=4, num_samples=32, seed=4)
    r_init = next(t for t in q.graph.initializer if t.name.endswith("_paroquant_r"))
    r = onnx.numpy_helper.to_array(r_init).astype(np.float64)
    k = r.shape[0]
    assert np.allclose(r @ r.T, np.eye(k), atol=1e-4)

    # Every 2x2 block on adjacent-channel pairs (0,1), (2,3), ... is a
    # Givens rotation; every off-pair entry is exactly zero (unlike
    # onnxsim.apply_spinquant's dense rotation matrix).
    for i in range(0, k, 2):
        j = i + 1
        assert np.isclose(r[i, i], r[j, j])
        assert np.isclose(r[i, j], -r[j, i])
        for other in range(k):
            if other in (i, j):
                continue
            assert r[i, other] == 0.0
            assert r[j, other] == 0.0
            assert r[other, i] == 0.0
            assert r[other, j] == 0.0


def test_paroquant_weight_reconstruction_matches_numpy_within_tight_tolerance():
    model = _matmul_model(K=16, N=4, seed=8)
    q = onnxsim.apply_paroquant(
        model, block_size=8, alpha=0.5, num_angle_steps=9, num_samples=16, seed=9
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

    codes_kn = onnx.numpy_helper.to_array(codes_init).astype(np.float64)  # [K, N]
    scale_blocks_kn = onnx.numpy_helper.to_array(scale_init).astype(np.float64)
    r = onnx.numpy_helper.to_array(r_init).astype(np.float64)
    inv_s = onnx.numpy_helper.to_array(inv_s_init).astype(np.float64)

    k, n = codes_kn.shape
    block_size = 8
    scale_full_kn = np.repeat(scale_blocks_kn, block_size, axis=0)
    w_tilde_kn = codes_kn * scale_full_kn  # dequantized, [K, N]

    orig_w = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W")
    ).astype(np.float64)
    s = 1.0 / inv_s
    w_smooth_nk = orig_w.T * s[np.newaxis, :]  # [N, K]
    w_tilde_expected_nk = w_smooth_nk @ r  # exact before quantization

    # Round-to-nearest quantization error is bounded by half the per-block
    # scale; checking that bound (numpy, not an onnxruntime round-trip --
    # see this project's own platform-numerics note: onnxruntime's MatMul
    # reduction order isn't bit-exact across CPU architectures) verifies
    # both the "exact before quantization" scale+rotation algebra and that
    # the codes/scale initializers actually quantize it correctly.
    err = np.abs(w_tilde_kn.T - w_tilde_expected_nk)
    half_step = 0.5 * scale_full_kn.T
    assert np.all(err <= half_step * (1.0 + 1e-3) + 1e-6)


def test_paroquant_gemm_transb_with_bias():
    rng = np.random.default_rng(5)
    K, N = 32, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_paroquant(model, block_size=8, num_samples=16, seed=6)
    onnx.checker.check_model(q)
    assert "Add" in [n.op_type for n in q.graph.node]

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_paroquant_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)  # 20 is not a multiple of 8
    q = onnxsim.apply_paroquant(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_paroquant_declines_odd_block_size():
    model = _matmul_model(K=32, N=8, seed=10)
    q = onnxsim.apply_paroquant(model, block_size=7)
    assert q.SerializeToString() == model.SerializeToString()


def test_paroquant_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_paroquant(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_paroquant_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_paroquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_paroquant_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_paroquant(model)
    assert result.SerializeToString() == model.SerializeToString()
