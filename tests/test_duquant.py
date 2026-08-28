"""Tests for ``onnxsim.apply_duquant`` -- see ``onnxsim/duquant.py`` for
the technique (calibrated permutation redistributing outlier channels
across quantization blocks, plus a block-local random rotation, then INT4
round-to-nearest quantization of both the weight and the activation).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.duquant import _build_duquant_rotation

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


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
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


def test_build_duquant_rotation_is_orthogonal():
    rng = np.random.default_rng(0)
    absmax = rng.random(32)
    u = _build_duquant_rotation(absmax, block_size=8, outlier_fraction=0.1, rng=rng)
    assert np.allclose(u @ u.T, np.eye(32), atol=1e-8)


def test_build_duquant_rotation_spreads_outlier_channels_across_blocks():
    # 4 channels with massive magnitude, all originally sitting in what
    # would be the SAME block (indices 0-3, block_size=8 means indices
    # 0-7 are one block) -- the permutation must not leave them clustered.
    k, block_size = 32, 8
    absmax = np.full(k, 1.0)
    outlier_channels = [0, 1, 2, 3]
    for c in outlier_channels:
        absmax[c] = 1000.0

    rng = np.random.default_rng(1)
    # outlier_fraction picks exactly 4 channels (4/32 = 0.125).
    u = _build_duquant_rotation(absmax, block_size, outlier_fraction=0.125, rng=rng)

    # Recover the permutation from U's block structure: since U = P @ R
    # with R block-diagonal, U's non-zero COLUMN blocks per ROW block of P
    # tell us which original channel landed in which block. Simpler: probe
    # by rotating each original one-hot outlier channel through U and
    # checking which block of the output has non-trivial energy.
    num_blocks = k // block_size
    block_of_channel = {}
    for c in outlier_channels:
        onehot = np.zeros(k)
        onehot[c] = 1.0
        rotated = onehot @ u
        energy_per_block = [
            np.sum(rotated[b * block_size : (b + 1) * block_size] ** 2)
            for b in range(num_blocks)
        ]
        block_of_channel[c] = int(np.argmax(energy_per_block))

    # All 4 outlier channels must land in DISTINCT blocks (there are
    # exactly num_blocks=4 blocks and 4 outliers -- one each).
    assert len(set(block_of_channel.values())) == len(outlier_channels)


def test_duquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_duquant(model, block_size=8, num_samples=16, seed=1)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert "DequantizeLinear" in op_types
    assert "ReduceMax" in op_types  # data-free per-token activation scale

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_duquant_rotation_matrix_is_orthogonal():
    model = _matmul_model(K=16, N=4, seed=3)
    q = onnxsim.apply_duquant(model, block_size=4, num_samples=32, seed=4)
    u_init = next(t for t in q.graph.initializer if t.name.endswith("_duquant_u"))
    u = onnx.numpy_helper.to_array(u_init).astype(np.float64)
    assert np.allclose(u @ u.T, np.eye(u.shape[0]), atol=1e-4)


def test_duquant_gemm_transb_with_bias():
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
    q = onnxsim.apply_duquant(model, block_size=8, num_samples=16, seed=6)
    onnx.checker.check_model(q)
    assert "Add" in [n.op_type for n in q.graph.node]

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.5


def test_duquant_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)  # 20 is not a multiple of 8
    q = onnxsim.apply_duquant(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_duquant_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_duquant(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_duquant_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_duquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_duquant_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_duquant(model)
    assert result.SerializeToString() == model.SerializeToString()
