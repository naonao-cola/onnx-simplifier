"""Tests for ``onnxsim.quantize_weight_only_kmeans`` -- see
``onnxsim/kmeans_quantization.py`` for the technique (a per-layer,
k-means-fitted weight codebook -- unlike NF4/MXFP4's fixed, data-independent
codebooks -- represented via ordinary Gather/Cast, no scale multiply needed).
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


def _matmul_model(K=32, N=8, weight=None, seed=0):
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
    codebook = onnx.numpy_helper.to_array(
        next(
            t for t in model.graph.initializer if t.name == f"{w_name}_kmeans_codebook"
        )
    ).astype(np.float64)
    codes = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == f"{w_name}_kmeans_codes")
    ).astype(np.int64)
    return codebook[codes]


def test_kmeans_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.quantize_weight_only_kmeans(model)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_kmeans_codebook_has_2_pow_bits_entries():
    model = _matmul_model(K=32, N=8, seed=1)
    q = onnxsim.quantize_weight_only_kmeans(model, bits=4)
    codebook = next(t for t in q.graph.initializer if t.name == "W_kmeans_codebook")
    assert onnx.numpy_helper.to_array(codebook).shape == (16,)


def test_kmeans_dequantized_values_are_close_to_original():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_kmeans(model, bits=4, iters=30)

    w_hand = _dequantize_by_hand(q)
    # A well-converged 16-centroid k-means fit to ~256 Gaussian samples
    # should get most weights within a small fraction of the tensor's own
    # standard deviation -- a real fit-quality check, not just "it runs".
    assert np.sqrt(np.mean((w_hand - weight.astype(np.float64)) ** 2)) < 0.05


def test_kmeans_two_different_layers_get_different_codebooks():
    # The whole point vs. NF4/MXFP4: each layer's codebook is fit to
    # THAT layer's own data, not shared globally.
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((32, 8)).astype(np.float32) * 0.1
    w2 = rng.standard_normal((32, 8)).astype(np.float32) * 5.0
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["H"]),
        onnx.helper.make_node("MatMul", ["H", "W2"], ["Y"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 32])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 8])],
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    q = onnxsim.quantize_weight_only_kmeans(model)
    cb1 = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W1_kmeans_codebook")
    )
    cb2 = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W2_kmeans_codebook")
    )
    assert not np.allclose(np.sort(cb1), np.sort(cb2))
    # w2's own scale (~5.0) should be reflected in its own codebook's range.
    assert np.abs(cb2).max() > np.abs(cb1).max() * 5


def test_kmeans_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=4)
    q = onnxsim.quantize_weight_only_kmeans(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_kmeans_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 32, 12
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
    q = onnxsim.quantize_weight_only_kmeans(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.2


def test_kmeans_codes_stay_in_range():
    rng = np.random.default_rng(7)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_kmeans(model, bits=4)
    codes = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W_kmeans_codes")
    )
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_kmeans_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(8)
    w_base = rng.standard_normal((32, 8)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((32, 4)).astype(np.float32) * 0.1
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["Y"]),
        onnx.helper.make_node("MatMul", ["X", "W_other"], ["H"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 32])],
        [
            onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 8]),
            onnx.helper.make_tensor_value_info("H", onnx.TensorProto.FLOAT, [4, 4]),
        ],
        [_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    q = onnxsim.quantize_weight_only_kmeans(model, skip_names={"W_other"})
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_kmeans_codebook" in names
    assert "W_other_kmeans_codebook" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_kmeans_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_kmeans(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_kmeans_noop_when_no_matmul_present():
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
    result = onnxsim.quantize_weight_only_kmeans(model)
    assert result.SerializeToString() == model.SerializeToString()
