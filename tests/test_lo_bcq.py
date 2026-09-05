"""Tests for ``onnxsim.quantize_weight_only_lo_bcq`` -- see
``onnxsim/lo_bcq.py`` for the technique (per-block-cluster, Lloyd-max-fitted
weight codebooks -- unlike ``kmeans_quantization.py``'s single shared
codebook, several small codebooks are fit, one per data-driven cluster of
blocks, via ``Gather``/``GatherElements``, no scale multiply needed).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _matmul_model(K=64, N=8, weight=None, seed=0):
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


def _dequantize_by_hand(model, w_name="W"):
    codebooks = onnx.numpy_helper.to_array(
        next(
            t for t in model.graph.initializer if t.name == f"{w_name}_lo_bcq_codebooks"
        )
    ).astype(np.float64)
    cluster_ids = onnx.numpy_helper.to_array(
        next(
            t
            for t in model.graph.initializer
            if t.name == f"{w_name}_lo_bcq_cluster_ids"
        )
    ).astype(np.int64)
    codes = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == f"{w_name}_lo_bcq_codes")
    ).astype(np.int64)
    selected = codebooks[cluster_ids]  # [num_blocks, num_codes]
    gathered = np.take_along_axis(selected, codes, axis=1)  # [num_blocks, block_size]
    return gathered


def test_lo_bcq_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=8, seed=0)
    q = onnxsim.quantize_weight_only_lo_bcq(model, block_size=32)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "GatherElements", "Reshape"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_lo_bcq_codebooks_have_num_clusters_by_2_pow_bits_shape():
    model = _matmul_model(K=64, N=8, seed=1)
    q = onnxsim.quantize_weight_only_lo_bcq(
        model, bits=4, block_size=32, num_clusters=3
    )
    codebooks = next(t for t in q.graph.initializer if t.name == "W_lo_bcq_codebooks")
    assert onnx.numpy_helper.to_array(codebooks).shape == (3, 16)


def test_lo_bcq_cluster_ids_stay_in_range():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_lo_bcq(model, num_clusters=4, block_size=32)
    cluster_ids = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W_lo_bcq_cluster_ids")
    )
    # weight is [K=64, N=8] (not transposed), reduction axis K -> w_nk is
    # [N=8, K=64], block_size=32 -> 2 blocks/row * 8 rows = 16 blocks.
    assert cluster_ids.shape == (16,)
    assert np.all(cluster_ids >= 0) and np.all(cluster_ids < 4)


def test_lo_bcq_codes_stay_in_range():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_lo_bcq(model, bits=4, block_size=32)
    codes = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W_lo_bcq_codes")
    )
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_lo_bcq_dequantized_values_are_close_to_original():
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_lo_bcq(
        model, bits=4, block_size=32, num_clusters=4, outer_iters=10
    )

    # weight is [K, N] (not transposed) -> w_nk = weight.T, [N, K].
    w_nk_hand = _dequantize_by_hand(q).reshape(8, 64)
    w_hand = w_nk_hand.T
    assert np.sqrt(np.mean((w_hand - weight.astype(np.float64)) ** 2)) < 0.05


def test_lo_bcq_two_statistically_different_regions_get_different_clusters():
    # LO-BCQ's whole point vs. a single shared codebook: blocks are
    # clustered by their OWN statistics, not by position -- so a
    # small-scale region and a large-scale region of the SAME tensor
    # should be assigned to differently-scaled per-cluster codebooks.
    rng = np.random.default_rng(5)
    small_rows = rng.standard_normal((32, 64)).astype(np.float64) * 0.05
    large_rows = rng.standard_normal((32, 64)).astype(np.float64) * 5.0
    w_nk = np.concatenate([small_rows, large_rows], axis=0)  # [64, 64]
    weight = w_nk.T.astype(np.float32)  # [K=64, N=64], transA-style layout
    model = _matmul_model(K=64, N=64, weight=weight)

    q = onnxsim.quantize_weight_only_lo_bcq(
        model, bits=4, block_size=64, num_clusters=2, outer_iters=10
    )
    cluster_ids = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W_lo_bcq_cluster_ids")
    )
    # One block per row (block_size == K == 64) -> 64 blocks, first 32
    # from the small-scale rows, last 32 from the large-scale rows.
    assert cluster_ids.shape == (64,)
    small_clusters = set(cluster_ids[:32].tolist())
    large_clusters = set(cluster_ids[32:].tolist())
    assert small_clusters.isdisjoint(large_clusters)

    codebooks = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W_lo_bcq_codebooks")
    )
    small_scale = np.abs(codebooks[cluster_ids[0]]).max()
    large_scale = np.abs(codebooks[cluster_ids[32]]).max()
    assert large_scale > small_scale * 5


def test_lo_bcq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=8, seed=6)
    q = onnxsim.quantize_weight_only_lo_bcq(model, block_size=32)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_lo_bcq_gemm_transb():
    rng = np.random.default_rng(8)
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
    q = onnxsim.quantize_weight_only_lo_bcq(model, block_size=32)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.2


def test_lo_bcq_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(9)
    w_base = rng.standard_normal((64, 8)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((64, 4)).astype(np.float32) * 0.1
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["Y"]),
        onnx.helper.make_node("MatMul", ["X", "W_other"], ["H"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 64])],
        [
            onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 8]),
            onnx.helper.make_tensor_value_info("H", onnx.TensorProto.FLOAT, [4, 4]),
        ],
        [_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    q = onnxsim.quantize_weight_only_lo_bcq(
        model, block_size=32, skip_names={"W_other"}
    )
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_lo_bcq_codebooks" in names
    assert "W_other_lo_bcq_codebooks" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_lo_bcq_skips_non_block_divisible_weight():
    rng = np.random.default_rng(10)
    weight = rng.standard_normal((48, 8)).astype(np.float32) * 0.5
    model = _matmul_model(K=48, N=8, weight=weight)
    q = onnxsim.quantize_weight_only_lo_bcq(model, block_size=32)
    assert q.SerializeToString() == model.SerializeToString()


def test_lo_bcq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_lo_bcq(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_lo_bcq_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 4])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 4])],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    result = onnxsim.quantize_weight_only_lo_bcq(model)
    assert result.SerializeToString() == model.SerializeToString()
