"""Tests for ``onnxsim.quantize_weight_only_pb_llm`` (PB-LLM, see
``onnxsim/pb_llm.py``) -- salience-driven per-column split between INT8
(salient columns) and ~1-bit binarization (everything else), for an
ordinary dense float32 MatMul/Gemm weight.
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model(K=64, N=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    ), weight


def _gemm_transb_model(K=48, N=8, seed=2):
    rng = np.random.default_rng(seed)
    # transB=1 -> weight stored [N, K]
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.3
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    ), weight


def _salient_calibration(K, num_samples=48, salient_channels=(2, 9), seed=1):
    # A handful of input channels with much larger activation magnitude
    # than the rest -- PB-LLM's own motivating case, since
    # salience_j = mean(|w_j|) * diag(H)_j depends on both the weight
    # column's own magnitude and how strongly the channel is actually
    # excited by real activations (via H = X^T X).
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in salient_channels:
        x[:, c] *= 25.0
    return x


def _outlier_weight(K=64, N=8, outlier_cols=(2, 9), seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float64) * 0.05
    for c in outlier_cols:
        weight[c, :] = rng.standard_normal(N) * 3.0
    return weight.astype(np.float32)


def _decode_pb_llm_weight(model, w_name, orig_shape):
    prefix = f"{w_name}_pb_llm"
    by_name = {t.name: t for t in model.graph.initializer}
    code = onnx.numpy_helper.to_array(by_name[f"{prefix}_code"]).astype(np.float64)
    scale = onnx.numpy_helper.to_array(by_name[f"{prefix}_scale"]).astype(np.float64)
    recon = code * scale
    assert recon.shape == orig_shape
    return recon, code


def test_reconstruction_error_improves_on_salient_outliers():
    K, N = 64, 8
    outlier_cols = (2, 9)
    model, weight = _matmul_model(K=K, N=N, seed=0)
    weight = _outlier_weight(K=K, N=N, outlier_cols=outlier_cols, seed=0)
    model.graph.initializer[0].CopyFrom(_f32(weight, "W"))

    calib = [{"X": _salient_calibration(K, salient_channels=outlier_cols)}]
    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=0.1
    )

    recon, code = _decode_pb_llm_weight(quantized, "W", weight.shape)
    pb_llm_err = np.linalg.norm(weight.astype(np.float64) - recon)

    # The outlier columns should have been selected salient (a full INT8
    # code range there), not just given a plain +-1 binary code.
    assert np.any(np.abs(code[outlier_cols[0], :]) > 1)
    assert np.any(np.abs(code[outlier_cols[1], :]) > 1)

    # A naive all-binary reconstruction (salient_ratio=0) should do
    # noticeably worse on the outlier columns than PB-LLM's own split.
    fully_binary = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=0.0
    )
    naive_recon, naive_code = _decode_pb_llm_weight(fully_binary, "W", weight.shape)
    assert set(np.unique(naive_code).tolist()) <= {-1, 1}
    naive_err = np.linalg.norm(weight.astype(np.float64) - naive_recon)

    assert pb_llm_err < naive_err


def test_codes_are_in_valid_discrete_set():
    K, N = 40, 6
    model, weight = _matmul_model(K=K, N=N, seed=3)
    calib = [{"X": _salient_calibration(K, salient_channels=(1, 5), seed=4)}]
    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=0.2
    )

    by_name = {t.name: t for t in quantized.graph.initializer}
    code = onnx.numpy_helper.to_array(by_name["W_pb_llm_code"])
    assert code.dtype == np.int8
    assert np.all(code >= -127) and np.all(code <= 127)

    # At least one column should have taken the full INT8 range (salient)
    # and at least one should be the plain +-1 binary code (non-salient).
    per_col_abs_max = np.abs(code).max(axis=1)
    assert np.any(per_col_abs_max > 1)
    assert np.any(per_col_abs_max == 1)


def test_salient_ratio_zero_is_fully_binary():
    K, N = 32, 4
    model, weight = _matmul_model(K=K, N=N, seed=6)
    calib = [{"X": _salient_calibration(K, salient_channels=(3, 10), seed=7)}]
    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=0.0
    )
    by_name = {t.name: t for t in quantized.graph.initializer}
    code = onnx.numpy_helper.to_array(by_name["W_pb_llm_code"])
    assert set(np.unique(code).tolist()) <= {-1, 1}


def test_salient_ratio_one_is_full_int8():
    K, N = 32, 4
    model, weight = _matmul_model(K=K, N=N, seed=8)
    calib = [{"X": _salient_calibration(K, salient_channels=(3, 10), seed=9)}]
    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=1.0
    )
    recon, code = _decode_pb_llm_weight(quantized, "W", weight.shape)

    # Full INT8 round-to-nearest should reconstruct closely everywhere.
    rel_err = np.linalg.norm(weight.astype(np.float64) - recon) / np.linalg.norm(
        weight.astype(np.float64)
    )
    assert rel_err < 0.05


def test_end_to_end_float_closeness():
    K, N = 64, 8
    model, weight = _matmul_model(K=K, N=N, seed=5)
    calib = [{"X": _salient_calibration(K, salient_channels=(3, 20), seed=6)}]
    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=0.15
    )

    x = np.random.default_rng(7).standard_normal((4, K)).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})

    # A mostly-binary quantizer is aggressively lossy by design -- not
    # remotely INT4-level closeness -- but should still be well within the
    # same order of magnitude as the float output, not noise.
    assert _rel_l2(float_out, quant_out) < 0.9


def test_gemm_transb_weight():
    # transB=1 stores W as [N, K] rather than MatMul's [K, N] -- exercises
    # the other branch of the scale-broadcast-shape logic (see
    # quantize_weight_only_pb_llm's own comment on weight_transposed).
    K, N = 48, 8
    model, weight = _gemm_transb_model(K=K, N=N, seed=8)
    calib = [{"X": _salient_calibration(K, salient_channels=(4, 15), seed=9)}]
    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, salient_ratio=0.15
    )

    recon, code = _decode_pb_llm_weight(quantized, "W", weight.shape)
    assert np.all(code >= -127) and np.all(code <= 127)

    x = _salient_calibration(K, num_samples=3, salient_channels=(4, 15), seed=10)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})
    assert _rel_l2(float_out, quant_out) < 0.9


def test_noop_on_non_matching_layer():
    K, N = 32, 4
    model, weight = _matmul_model(K=K, N=N, seed=11)
    calib = [{"X": _salient_calibration(K, salient_channels=(1, 6), seed=12)}]

    quantized = onnxsim.quantize_weight_only_pb_llm(
        model, calibration_data=calib, skip_names={"W"}
    )
    assert quantized.SerializeToString() == model.SerializeToString()

    # Also a layer whose weight isn't a plain constant 2-D float32 tensor
    # (a 1-D bias-shaped initializer here) shouldn't be touched at all.
    conv_like = _model(
        """
        g (float[1,4] X) => (float[1,4] Y)
        {
          Y = Add(X, Bias)
        }
        """,
        [_f32(np.zeros(4, dtype=np.float32), "Bias")],
    )
    out = onnxsim.quantize_weight_only_pb_llm(conv_like, calibration_data=calib)
    assert out.SerializeToString() == conv_like.SerializeToString()
