"""Tests for ``onnxsim.apply_duquant_cpp`` -- the C++-backed port of
DuQuant's own calibrated permutation + block-local random rotation (see
``onnxsim/duquant_entry.h``). Like ``tests/test_spinquant_cpp.py``, this
runs the model over real calibration data through a real
``onnxruntime``-backed executor -- never a fake/mock executor.

``onnxsim.duquant.apply_duquant`` (the pure-Python entry point) is now a
thin wrapper around this same C++ function -- see ``onnxsim/duquant.py`` and
``tests/test_duquant.py`` (which exercises the identical public contract
through that wrapper). There is no longer a separate pure-Python permutation/
rotation to compare against for exact parity -- ``duquant_entry.h``'s own
"ACCEPTED, PERMANENT DIVERGENCE" note explains why a per-node RNG plus a
Gram-Schmidt rotation need not agree column-for-column, or channel-for-
-channel, with the Python reference's own single sequentially-advancing
generator and sign-corrected-QR rotation. This file instead verifies what IS
guaranteed regardless of which valid permutation/rotation is found: the
fitted matrix ``U`` is orthogonal, the permute-rotate-then-quantize
composition stays numerically close to the unrotated float reference, and
documented edge cases behave as specified.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_duquant_cpp

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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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


def _calibration(K=32, num_samples=16, seed=1, outlier_channels=()):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in outlier_channels:
        x[:, c] *= 50.0
    return [{"X": x}]


def test_cpp_rotation_is_orthogonal():
    K, N, block_size = 24, 6, 8
    model = _matmul_model(K=K, N=N, seed=0)
    q = apply_duquant_cpp(
        model, block_size=block_size, calibration_data=_calibration(K=K)
    )
    onnx.checker.check_model(q)
    u_init = next(t for t in q.graph.initializer if t.name.endswith("_duquant_u"))
    u = onnx.numpy_helper.to_array(u_init).astype(np.float64)
    assert u.shape == (K, K)
    np.testing.assert_allclose(u @ u.T, np.eye(K), atol=1e-4)
    np.testing.assert_allclose(u.T @ u, np.eye(K), atol=1e-4)


def test_cpp_output_stays_close_to_float():
    K, N, block_size = 32, 8, 8
    model = _matmul_model(K=K, N=N, seed=2)
    q = apply_duquant_cpp(
        model, block_size=block_size, calibration_data=_calibration(K=K, seed=3)
    )
    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2
    assert "DequantizeLinear" in op_types
    assert "ReduceMax" in op_types  # data-free per-token activation scale

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_outlier_channels_spread_across_blocks():
    # Same probe idea as tests/test_duquant.py's own pure-Python
    # test_build_duquant_rotation_spreads_outlier_channels_across_blocks,
    # but through the full graph rewrite this port performs: a handful of
    # channels with massive calibration magnitude, all originally within
    # what would be a single block, must land in distinct blocks of U's
    # own row-block structure.
    K, block_size = 32, 8
    outlier_channels = [0, 1, 2, 3]
    model = _matmul_model(K=K, N=4, seed=1)
    q = apply_duquant_cpp(
        model,
        block_size=block_size,
        outlier_fraction=0.125,  # 4 / 32
        calibration_data=_calibration(K=K, seed=2, outlier_channels=outlier_channels),
    )
    u_init = next(t for t in q.graph.initializer if t.name.endswith("_duquant_u"))
    u = onnx.numpy_helper.to_array(u_init).astype(np.float64)

    num_blocks = K // block_size
    block_of_channel = {}
    for c in outlier_channels:
        onehot = np.zeros(K)
        onehot[c] = 1.0
        rotated = onehot @ u
        energy_per_block = [
            np.sum(rotated[b * block_size : (b + 1) * block_size] ** 2)
            for b in range(num_blocks)
        ]
        block_of_channel[c] = int(np.argmax(energy_per_block))
    assert len(set(block_of_channel.values())) == len(outlier_channels)


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
    q = apply_duquant_cpp(
        model, block_size=block_size, calibration_data=_calibration(K=K, seed=6)
    )
    onnx.checker.check_model(q)
    assert any(
        n.op_type == "Add" and n.name.endswith("_bias_add_node") for n in q.graph.node
    )

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)
    q = apply_duquant_cpp(
        model, block_size=8, calibration_data=_calibration(K=20, seed=7)
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
    q = apply_duquant_cpp(model, calibration_data=_calibration(K=32, seed=8))
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
    result = apply_duquant_cpp(
        model, calibration_data=[{"X": np.zeros((2, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_below_opset21():
    model = _matmul_model(K=32, N=8, seed=9, opset=13)
    result = apply_duquant_cpp(model, calibration_data=_calibration(K=32, seed=9))
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_missing_graph_input_raises_value_error():
    model = _matmul_model(K=16, N=4, seed=10)
    with pytest.raises(ValueError):
        apply_duquant_cpp(
            model,
            block_size=8,
            calibration_data=[{"WrongName": np.zeros((2, 16), dtype=np.float32)}],
        )


def test_cpp_empty_calibration_data_leaves_model_unchanged():
    model = _matmul_model(K=16, N=4, seed=11)
    result = apply_duquant_cpp(model, block_size=8, calibration_data=[])
    assert result.SerializeToString() == model.SerializeToString()
