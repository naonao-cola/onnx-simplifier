"""Tests for ``onnxsim.quantize_weight_only_drop_by_drop`` and
``onnxsim.select_drop_by_drop_prefix`` (Drop-by-Drop, see
``onnxsim/drop_by_drop.py``) -- successive-refinement additive codebook
quantization where every prefix of the ``K`` fitted codebook stages is
independently a usable, complete reconstruction, not just the full sum.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _stage_arrays(model, w_name, num_codebooks):
    """Independent reference decode: reads each stage's codebook/codes
    initializers directly and returns the list of per-stage
    ``[num_groups, group_dim]`` reconstructions -- without using any of
    this module's own internal functions or the ops it inserts, and
    without summing them, so callers can build any prefix sum by hand.
    """
    stages = []
    for m in range(num_codebooks):
        codebook = onnx.numpy_helper.to_array(
            next(
                t
                for t in model.graph.initializer
                if t.name == f"{w_name}_dbd_codebook{m}"
            )
        ).astype(np.float64)
        codes = onnx.numpy_helper.to_array(
            next(
                t for t in model.graph.initializer if t.name == f"{w_name}_dbd_codes{m}"
            )
        )
        stages.append(codebook[codes])
    return stages


def test_drop_by_drop_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_drop_by_drop(model, group_dim=8, seed=0)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert "Gather" in op_types
    assert "Add" in op_types

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_drop_by_drop_dequantized_values_match_hand_decoded_reference():
    K, N = 32, 8
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(K=K, N=N, weight=weight)
    q = onnxsim.quantize_weight_only_drop_by_drop(
        model, group_dim=8, num_codebooks=3, seed=1
    )

    combined = sum(_stage_arrays(q, "W", num_codebooks=3))
    w_hand_nk = combined.reshape(N, K)
    w_hand = w_hand_nk.T  # back to original [K, N] storage

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    weight_tensor_name = matmul_node.input[1]
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(q)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=weight_tensor_name))
    rng2 = np.random.default_rng(3)
    x = rng2.standard_normal((4, K)).astype(np.float32)
    (w_graph,) = _run(probe_model, {"X": x})[len(q.graph.output) :]

    assert np.allclose(w_hand, w_graph.astype(np.float64), rtol=0, atol=1e-5)


def test_drop_by_drop_prefix_reconstruction_improves_with_more_terms():
    # The whole point of Drop-by-Drop: reconstructing from only the first
    # k of K additive terms (summed directly from the initializers, no
    # graph surgery) is already a complete, valid reconstruction at every
    # k, and error should not increase as k grows -- distinct from merely
    # checking the final K-term sum is accurate, which any additive
    # residual scheme (including AQLM's) already satisfies.
    K, N = 40, 10
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, weight=weight)
    q = onnxsim.quantize_weight_only_drop_by_drop(
        model, group_dim=8, num_codebooks=4, codebook_size=16, seed=5
    )

    stages = _stage_arrays(q, "W", num_codebooks=4)
    w_nk = weight.astype(np.float64).T  # [N, K]

    errors = []
    running = np.zeros_like(stages[0])
    for stage in stages:
        running = running + stage
        recon_nk = running.reshape(N, K)
        errors.append(np.sum((recon_nk - w_nk) ** 2))

    assert len(errors) == 4
    assert all(errors[i] >= errors[i + 1] - 1e-6 for i in range(len(errors) - 1))
    # Strictly better with all 4 terms than with just the first 1.
    assert errors[3] < errors[0]


def test_select_drop_by_drop_prefix_matches_hand_truncated_reconstruction():
    K, N = 32, 8
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(K=K, N=N, weight=weight)
    q = onnxsim.quantize_weight_only_drop_by_drop(
        model, group_dim=8, num_codebooks=4, codebook_size=16, seed=7
    )

    for k in (1, 2, 3, 4):
        truncated = onnxsim.select_drop_by_drop_prefix(q, k)
        onnx.checker.check_model(truncated)

        stages = _stage_arrays(truncated, "W", num_codebooks=4)
        hand_combined = sum(stages[:k])
        w_hand_nk = hand_combined.reshape(N, K)
        w_hand = w_hand_nk.T

        matmul_node = next(n for n in truncated.graph.node if n.op_type == "MatMul")
        weight_tensor_name = matmul_node.input[1]
        probe_model = onnx.ModelProto()
        probe_model.CopyFrom(truncated)
        probe_model.graph.output.append(onnx.ValueInfoProto(name=weight_tensor_name))
        rng2 = np.random.default_rng(8)
        x = rng2.standard_normal((4, K)).astype(np.float32)
        (w_graph,) = _run(probe_model, {"X": x})[len(truncated.graph.output) :]

        assert np.allclose(w_hand, w_graph.astype(np.float64), rtol=0, atol=1e-5)


def test_select_drop_by_drop_prefix_rejects_k_out_of_range():
    model = _matmul_model(K=32, N=8, seed=9)
    q = onnxsim.quantize_weight_only_drop_by_drop(model, group_dim=8, num_codebooks=2)

    with pytest.raises(ValueError):
        onnxsim.select_drop_by_drop_prefix(q, 0)
    with pytest.raises(ValueError):
        onnxsim.select_drop_by_drop_prefix(q, 3)


def test_drop_by_drop_gemm_transb():
    rng = np.random.default_rng(10)
    K, N = 64, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.quantize_weight_only_drop_by_drop(model, group_dim=8, seed=2)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_drop_by_drop_codes_stay_in_codebook_range():
    model = _matmul_model(K=32, N=8, seed=11)
    q = onnxsim.quantize_weight_only_drop_by_drop(
        model, group_dim=8, num_codebooks=2, codebook_size=16, seed=3
    )
    for m in range(2):
        codes = onnx.numpy_helper.to_array(
            next(t for t in q.graph.initializer if t.name == f"W_dbd_codes{m}")
        )
        assert np.all(codes >= 0) and np.all(codes < 16)


def test_drop_by_drop_skips_non_group_divisible_k():
    model = _matmul_model(K=20, N=4, seed=12)  # 20 is not a multiple of 8
    q = onnxsim.quantize_weight_only_drop_by_drop(model, group_dim=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_drop_by_drop_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_drop_by_drop(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_drop_by_drop_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_drop_by_drop(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_select_drop_by_drop_prefix_noop_when_no_drop_by_drop_layers():
    model = _matmul_model(K=32, N=8, seed=13)
    result = onnxsim.select_drop_by_drop_prefix(model, 1)
    assert result.SerializeToString() == model.SerializeToString()
