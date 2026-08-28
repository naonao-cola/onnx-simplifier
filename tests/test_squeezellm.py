"""Tests for ``onnxsim.quantize_weight_only_squeezellm`` (SqueezeLLM, see
``onnxsim/squeezellm.py``) -- sensitivity-weighted k-means fits a small
per-group codebook (arbitrary values, not a uniform grid) to each group's
own weight distribution, with a handful of outlier elements excluded from
the fit and corrected back to their exact original value via a dense
sparse-position correction.
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


def _matmul_model(K=64, N=16, weight=None, seed=0, opset=13):
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


def _dequantize_squeezellm_by_hand(model, w_name, block_size, weight_transposed):
    """Independent reference decode: reads the codebook/codes/sparse-diff
    initializers directly and reconstructs via numpy, without using any of
    this module's own internal functions or the ops it inserts -- checks
    actual values, not just that the graph runs.
    """
    codebook = onnx.numpy_helper.to_array(
        next(
            t
            for t in model.graph.initializer
            if t.name == f"{w_name}_squeezellm_codebook"
        )
    )
    codes = onnx.numpy_helper.to_array(
        next(
            t for t in model.graph.initializer if t.name == f"{w_name}_squeezellm_codes"
        )
    ).reshape(-1, block_size)
    sparse_diff = onnx.numpy_helper.to_array(
        next(
            t
            for t in model.graph.initializer
            if t.name == f"{w_name}_squeezellm_sparse_diff"
        )
    )
    n, k = sparse_diff.shape
    num_groups = codes.shape[0]
    dequant_nk = (
        codebook[np.arange(num_groups)[:, np.newaxis], codes].reshape(n, k)
        + sparse_diff
    )
    return dequant_nk if weight_transposed else dequant_nk.T


def _varied_calibration(K=64, num_samples=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def test_squeezellm_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _varied_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(q_model)

    op_types = {n.op_type for n in q_model.graph.node}
    assert "GatherND" in op_types

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q_model, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_squeezellm_dequantized_values_match_hand_decoded_reference():
    # Probes the graph's own reconstructed-weight tensor (the node feeding
    # the MatMul's weight input) via onnxruntime and checks it against an
    # independent numpy decode of the raw codebook/codes/sparse-diff
    # initializers -- a real per-element correctness check of both this
    # module's own graph construction and the hand-decode helper the other
    # tests below rely on, not just "the graph runs".
    K, N, block_size = 32, 8, 32
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(K=K, N=N, weight=weight)
    x = _varied_calibration(K=K, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=calibration_data, block_size=block_size
    )
    w_hand = _dequantize_squeezellm_by_hand(
        q_model, "W", block_size, weight_transposed=False
    )

    matmul_node = next(n for n in q_model.graph.node if n.op_type == "MatMul")
    weight_tensor_name = matmul_node.input[1]
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(q_model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=weight_tensor_name))
    (w_graph,) = _run(probe_model, {"X": x})[len(q_model.graph.output) :]

    assert np.allclose(w_hand, w_graph.astype(np.float64), rtol=0, atol=1e-5)


def test_squeezellm_sparse_correction_is_exact_at_outlier_positions():
    K, N, block_size = 32, 4, 32
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    # A handful of extreme values that should be captured as outliers and
    # corrected back exactly, whatever the k-means codebook alone would
    # have reconstructed them as.
    weight[0, 0] = 50.0
    weight[5, 2] = -37.0
    model = _matmul_model(K=K, N=N, weight=weight)
    x = _varied_calibration(K=K, num_samples=32, seed=5)
    calibration_data = [{"X": x}]

    q_model = onnxsim.quantize_weight_only_squeezellm(
        model,
        calibration_data=calibration_data,
        block_size=block_size,
        outlier_fraction=0.02,
    )
    w_hand = _dequantize_squeezellm_by_hand(
        q_model, "W", block_size, weight_transposed=False
    )
    assert np.isclose(w_hand[0, 0], 50.0, atol=1e-3)
    assert np.isclose(w_hand[5, 2], -37.0, atol=1e-3)


def test_squeezellm_codebook_beats_naive_uniform_on_multimodal_weights():
    # A single block whose values cluster tightly around two far-apart
    # modes: a naive uniform grid spanning the block's full min/max wastes
    # most of its levels on the empty gap between the modes, while a
    # sensitivity-weighted (here, uniform-sensitivity) codebook fit
    # directly to the data should concentrate levels at the two modes and
    # reconstruct far more accurately at the same bit width.
    K, N, block_size, bits = 32, 1, 32, 2  # 4 levels
    rng = np.random.default_rng(6)
    weight = np.concatenate(
        [
            rng.normal(-1.0, 0.02, K // 2),
            rng.normal(1.0, 0.02, K - K // 2),
        ]
    ).astype(np.float32)[:, np.newaxis]
    model = _matmul_model(K=K, N=N, weight=weight)
    x = _varied_calibration(K=K, num_samples=16, seed=7)
    calibration_data = [{"X": x}]

    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=calibration_data, block_size=block_size, bits=bits
    )
    w_hand = _dequantize_squeezellm_by_hand(
        q_model, "W", block_size, weight_transposed=False
    )
    squeezellm_mse = float(np.mean((w_hand - weight) ** 2))

    lo, hi = weight.min(), weight.max()
    scale = (hi - lo) / (2**bits - 1)
    naive_codes = np.round((weight - lo) / scale)
    naive_dequant = naive_codes * scale + lo
    naive_mse = float(np.mean((naive_dequant - weight) ** 2))

    assert squeezellm_mse < naive_mse


def test_squeezellm_codes_stay_in_range():
    K, N, bits = 32, 8, 3
    weight_rng = np.random.default_rng(8)
    weight = weight_rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)
    x = _varied_calibration(K=K, num_samples=16, seed=9)
    calibration_data = [{"X": x}]

    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=calibration_data, block_size=32, bits=bits
    )
    codes = onnx.numpy_helper.to_array(
        next(t for t in q_model.graph.initializer if t.name == "W_squeezellm_codes")
    )
    assert np.all(codes >= 0) and np.all(codes < 2**bits)


def test_squeezellm_gemm_transb():
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
    x = _varied_calibration(K=K, num_samples=32, seed=11)
    calibration_data = [{"X": x}]

    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=calibration_data, block_size=32
    )
    onnx.checker.check_model(q_model)

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q_model, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_squeezellm_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=12)  # 48 is not a multiple of 64
    x = _varied_calibration(K=48, num_samples=16, seed=13)
    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=[{"X": x}], block_size=64
    )
    assert q_model.SerializeToString() == model.SerializeToString()


def test_squeezellm_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q_model = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=[{"X": np.zeros((4, 64), dtype=np.float32)}]
    )
    assert q_model.SerializeToString() == model.SerializeToString()


def test_squeezellm_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_squeezellm(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_squeezellm_noop_on_old_opset():
    # GatherND's batch_dims needs opset >= 12.
    model = _matmul_model(K=32, N=8, seed=14, opset=11)
    x = _varied_calibration(K=32, num_samples=16, seed=15)
    result = onnxsim.quantize_weight_only_squeezellm(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()
