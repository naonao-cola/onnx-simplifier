"""Tests for ``onnxsim.quantize_weight_only_gptvq`` (GPTVQ, see
``onnxsim/gptvq.py``) -- combines ``onnxsim.gptq``'s Hessian-based
sequential error compensation with an ``onnxsim.aqlm``-style codebook fit
to the layer's own weight values, quantizing consecutive groups of
``vector_dim`` columns jointly instead of one scalar element at a time.
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


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # GPTQ's own motivating scenario, reused here: input channels that are
    # *correlated* (every channel a linear combination of a handful of
    # latent factors) -- independent per-group codebook assignment can't
    # compensate for one group's error using another's, but the Hessian's
    # off-diagonal terms (capturing exactly this correlation) can.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _dequantize_gptvq_by_hand(model, w_name="W"):
    """Independent reference decode: reads the codebook/codes initializers
    directly and reconstructs via numpy, without using any of this
    module's own internal functions or the ops it inserts. Returns the
    ``[num_groups, vector_dim]`` gathered codebook vectors -- reshaping
    back to the weight's own ``[N, K]``/original layout is the caller's
    job, since only the caller knows which.
    """
    codebook = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == f"{w_name}_gptvq_codebook")
    ).astype(np.float64)
    codes = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == f"{w_name}_gptvq_codes")
    )
    return codebook[codes]


def test_gptvq_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=32, N=8, seed=0)
    x = _correlated_calibration(K=32, num_samples=32, rank=3, seed=1)
    q = onnxsim.quantize_weight_only_gptvq(
        model, calibration_data=[{"X": x}], vector_dim=2, num_centroids=16
    )
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Gather", "Reshape", "Transpose"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_gptvq_dequantized_values_match_hand_decoded_reference():
    K, N = 32, 8
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(K=K, N=N, weight=weight)
    x = _correlated_calibration(K=K, num_samples=32, rank=3, seed=3)

    q = onnxsim.quantize_weight_only_gptvq(
        model,
        calibration_data=[{"X": x}],
        vector_dim=2,
        num_centroids=16,
        seed=1,
    )

    gathered = _dequantize_gptvq_by_hand(q)
    w_hand_nk = gathered.reshape(N, K)
    w_hand = w_hand_nk.T  # back to original [K, N] storage

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    weight_tensor_name = matmul_node.input[1]
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(q)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=weight_tensor_name))
    rng2 = np.random.default_rng(4)
    x2 = rng2.standard_normal((4, K)).astype(np.float32)
    (w_graph,) = _run(probe_model, {"X": x2})[len(q.graph.output) :]

    assert np.allclose(w_hand, w_graph.astype(np.float64), rtol=0, atol=1e-5)


def test_gptvq_reduces_reconstruction_error_vs_plain_kmeans():
    # The whole point vs. onnxsim.kmeans_quantization/onnxsim.aqlm: using
    # the calibration Hessian to compensate each group's error into the
    # columns not yet processed should reconstruct the *layer's actual
    # output* (weighted by real activation statistics) better than
    # assigning every group to its nearest codebook entry independently,
    # with no error feedback at all.
    K, N = 32, 8
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, weight=weight)
    x = _correlated_calibration(K=K, num_samples=64, rank=4, seed=6)
    w_float = weight.astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    q_gptvq = onnxsim.quantize_weight_only_gptvq(
        model,
        calibration_data=[{"X": x}],
        vector_dim=2,
        num_centroids=8,
        seed=7,
    )
    w_gptvq_nk = _dequantize_gptvq_by_hand(q_gptvq).reshape(N, K)
    y_gptvq = x.astype(np.float64) @ w_gptvq_nk.T
    gptvq_err = np.linalg.norm(y_float - y_gptvq)

    # No-compensation baseline: assign every group to its nearest entry of
    # the *same fitted codebook* GPTVQ itself would use, but never
    # propagate any group's residual to later groups.
    codebook = onnx.numpy_helper.to_array(
        next(t for t in q_gptvq.graph.initializer if t.name == "W_gptvq_codebook")
    ).astype(np.float64)
    w_nk = w_float.T
    num_groups = N * (K // 2)
    groups = w_nk.reshape(num_groups, 2)
    dist = np.sum((groups[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist, axis=1)
    w_rtn_nk = codebook[nearest].reshape(N, K)
    y_rtn = x.astype(np.float64) @ w_rtn_nk.T
    rtn_err = np.linalg.norm(y_float - y_rtn)

    assert gptvq_err < rtn_err


def test_gptvq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=8)
    x = _correlated_calibration(K=64, num_samples=32, rank=6, seed=9)
    q = onnxsim.quantize_weight_only_gptvq(
        model, calibration_data=[{"X": x}], vector_dim=4, num_centroids=32
    )
    onnx.checker.check_model(q)

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_gptvq_gemm_transb():
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
    x = _correlated_calibration(K=K, num_samples=32, rank=5, seed=11)
    q = onnxsim.quantize_weight_only_gptvq(
        model, calibration_data=[{"X": x}], vector_dim=2, num_centroids=32
    )
    onnx.checker.check_model(q)

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_gptvq_codes_stay_in_codebook_range():
    model = _matmul_model(K=32, N=8, seed=12)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=13)
    q = onnxsim.quantize_weight_only_gptvq(
        model, calibration_data=[{"X": x}], vector_dim=2, num_centroids=16
    )
    codes = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == "W_gptvq_codes")
    )
    assert np.all(codes >= 0) and np.all(codes < 16)


def test_gptvq_handles_dead_input_channel():
    # A channel with zero variance in the calibration data (H's diagonal
    # is exactly 0 there) must not blow up the Hessian inversion.
    model = _matmul_model(K=32, N=8, seed=14)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=15)
    x[:, 5] = 0.0  # dead channel
    q = onnxsim.quantize_weight_only_gptvq(
        model, calibration_data=[{"X": x}], vector_dim=2, num_centroids=16
    )
    onnx.checker.check_model(q)

    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))


def test_gptvq_skips_non_vector_dim_divisible_k():
    model = _matmul_model(K=20, N=4, seed=16)  # 20 is not a multiple of 8
    x = np.zeros((4, 20), dtype=np.float32)
    q = onnxsim.quantize_weight_only_gptvq(
        model, calibration_data=[{"X": x}], vector_dim=8
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_gptvq_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(17)
    w_base = rng.standard_normal((32, 8)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((32, 4)).astype(np.float32) * 0.1
    model = _model(
        """
        g (float[batch,32] X) => (float[batch,8] Y, float[batch,4] H)
        {
          Y = MatMul(X, W)
          H = MatMul(X, W_other)
        }
        """,
        initializer=[_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    x = rng.standard_normal((16, 32)).astype(np.float32)
    q = onnxsim.quantize_weight_only_gptvq(
        model,
        calibration_data=[{"X": x}],
        vector_dim=2,
        num_centroids=16,
        skip_names={"W_other"},
    )
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_gptvq_codebook" in names
    assert "W_other_gptvq_codebook" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_gptvq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_gptvq(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_gptvq_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_gptvq(model)
    assert result.SerializeToString() == model.SerializeToString()
