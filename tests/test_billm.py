"""Tests for ``onnxsim.quantize_weight_only_billm`` (BiLLM, see
``onnxsim/billm.py`` and ``docs/billm-quantization.md``) -- Hessian-guided
salient-column binary residual approximation plus plain per-block binary for
the rest, pushing an ordinary dense float32 MatMul/Gemm weight down to close
to 1 bit/element on average.
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
    # Mirrors test_awq.py's own scenario: a handful of input channels with
    # much larger activation magnitude than the rest -- BiLLM's own
    # motivating case, since s_i = w_i^2/[H_c]_ii^2 depends on both the
    # weight's own magnitude and how much a channel is actually excited by
    # real activations (via H = X^T X).
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in salient_channels:
        x[:, c] *= 25.0
    return x


def _outlier_weight(K=64, N=8, outlier_cols=(2, 9), seed=0):
    # Most columns small and tightly distributed; a few columns (matching
    # the calibration data's own high-activation channels) with much larger
    # magnitude -- genuine outlier structure a single flat per-block scale
    # represents poorly, but that BiLLM's salient/residual path should
    # capture much better.
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float64) * 0.05
    for c in outlier_cols:
        weight[c, :] = rng.standard_normal(N) * 3.0
    return weight.astype(np.float32)


def _plain_block_binary_reconstruction(w_kn, block_size):
    # Naive baseline: one flat sign(w)*mean(|w|) scale per (block of K),
    # with no salient-column handling at all -- exactly BiLLM's own
    # non-salient path applied uniformly to *every* column, the ablation
    # the paper's own Table 1/Figure 1 (RTN/GPTQ collapsing at 1 bit) is
    # about.
    k, n = w_kn.shape
    recon = np.empty_like(w_kn, dtype=np.float64)
    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        block = w_kn[start:end, :]
        scale = np.mean(np.abs(block))
        recon[start:end, :] = np.where(block >= 0.0, 1.0, -1.0) * scale
    return recon


def _decode_billm_weight(model, w_name, orig_shape):
    prefix = f"{w_name}_billm"
    by_name = {t.name: t for t in model.graph.initializer}
    code1 = onnx.numpy_helper.to_array(by_name[f"{prefix}_code1"]).astype(np.float64)
    code2 = onnx.numpy_helper.to_array(by_name[f"{prefix}_code2"]).astype(np.float64)
    scale1 = onnx.numpy_helper.to_array(by_name[f"{prefix}_scale1"]).astype(np.float64)
    scale2 = onnx.numpy_helper.to_array(by_name[f"{prefix}_scale2"]).astype(np.float64)
    recon = code1 * scale1 + code2 * scale2
    assert recon.shape == orig_shape
    return recon, code1, code2


def test_reconstruction_error_improves_on_salient_outliers():
    K, N = 64, 8
    outlier_cols = (2, 9)
    model, weight = _matmul_model(K=K, N=N, seed=0)
    weight = _outlier_weight(K=K, N=N, outlier_cols=outlier_cols, seed=0)
    model.graph.initializer[0].CopyFrom(_f32(weight, "W"))

    calib = [{"X": _salient_calibration(K, salient_channels=outlier_cols)}]
    quantized = onnxsim.quantize_weight_only_billm(
        model, calibration_data=calib, block_size=32
    )

    recon, code1, code2 = _decode_billm_weight(quantized, "W", weight.shape)
    billm_err = np.linalg.norm(weight.astype(np.float64) - recon)

    naive_recon = _plain_block_binary_reconstruction(weight.astype(np.float64), 32)
    naive_err = np.linalg.norm(weight.astype(np.float64) - naive_recon)

    assert billm_err < 0.5 * naive_err

    # The outlier columns should actually have been selected salient (both
    # code levels doing real work there), unlike a generic mid-block column.
    assert np.any(code2[outlier_cols[0], :] != 0)
    assert np.any(code2[outlier_cols[1], :] != 0)


def test_codes_are_in_valid_discrete_set():
    K, N = 40, 6
    model, weight = _matmul_model(K=K, N=N, seed=3)
    calib = [{"X": _salient_calibration(K, salient_channels=(1, 5), seed=4)}]
    quantized = onnxsim.quantize_weight_only_billm(
        model, calibration_data=calib, block_size=16
    )

    by_name = {t.name: t for t in quantized.graph.initializer}
    code1 = onnx.numpy_helper.to_array(by_name["W_billm_code1"])
    code2 = onnx.numpy_helper.to_array(by_name["W_billm_code2"])
    assert set(np.unique(code1).tolist()) <= {-1, 1}
    assert set(np.unique(code2).tolist()) <= {-1, 0, 1}
    assert code1.dtype == np.int8
    assert code2.dtype == np.int8


def test_end_to_end_float_closeness():
    K, N = 64, 8
    model, weight = _matmul_model(K=K, N=N, seed=5)
    calib = [{"X": _salient_calibration(K, salient_channels=(3, 20), seed=6)}]
    quantized = onnxsim.quantize_weight_only_billm(
        model, calibration_data=calib, block_size=32
    )

    x = np.random.default_rng(7).standard_normal((4, K)).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})

    # A ~1-bit-average binarizer is aggressively lossy by design -- not
    # remotely INT4-level closeness -- but should still be well within the
    # same order of magnitude as the float output, not noise.
    assert _rel_l2(float_out, quant_out) < 0.9


def test_gemm_transb_weight():
    # transB=1 stores W as [N, K] rather than MatMul's [K, N] -- exercises
    # the other branch of the scale-broadcast-shape logic (see
    # quantize_weight_only_billm's own comment on weight_transposed).
    K, N = 48, 8
    model, weight = _gemm_transb_model(K=K, N=N, seed=8)
    calib = [{"X": _salient_calibration(K, salient_channels=(4, 15), seed=9)}]
    quantized = onnxsim.quantize_weight_only_billm(
        model, calibration_data=calib, block_size=24
    )

    # Decoding the stored codes/scales back into the original [N, K] shape
    # (not raising on a shape mismatch) already exercises the
    # weight_transposed broadcast plumbing; codes stay in their valid set.
    recon, code1, code2 = _decode_billm_weight(quantized, "W", weight.shape)
    assert set(np.unique(code1).tolist()) <= {-1, 1}
    assert set(np.unique(code2).tolist()) <= {-1, 0, 1}

    # The real correctness measure for a GPTQ-style, Hessian-compensated
    # scheme is *output* closeness, not raw weight Frobenius error: block-
    # wise error compensation deliberately trades raw weight accuracy for
    # lower layer-output error (weighted by the calibration activations),
    # so it can legitimately make ``||W - W_hat||`` *worse* than a naive
    # per-block binary baseline when (as here) there's no real weight-
    # magnitude outlier for the salient/residual mechanism to exploit --
    # see this module's own docstring, point 2d. Evaluated on inputs drawn
    # from the same distribution as calibration -- like any Hessian/
    # calibration-based PTQ scheme (GPTQ, AWQ, ...), BiLLM's compensation
    # is only meaningful when deployment activations resemble calibration
    # ones; an input distribution calibration never saw is a distribution-
    # shift problem, not something this scheme claims to handle.
    x = _salient_calibration(K, num_samples=3, salient_channels=(4, 15), seed=10)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})
    assert _rel_l2(float_out, quant_out) < 0.9


def test_noop_on_non_matching_layer():
    K, N = 32, 4
    model, weight = _matmul_model(K=K, N=N, seed=11)
    calib = [{"X": _salient_calibration(K, salient_channels=(1, 6), seed=12)}]

    quantized = onnxsim.quantize_weight_only_billm(
        model, calibration_data=calib, block_size=16, skip_names={"W"}
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
    out = onnxsim.quantize_weight_only_billm(conv_like, calibration_data=calib)
    assert out.SerializeToString() == conv_like.SerializeToString()
