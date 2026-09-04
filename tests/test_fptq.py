"""Tests for ``onnxsim.apply_fptq`` (FPTQ, see ``onnxsim/fptq.py``) --
:mod:`onnxsim.smoothquant`'s own power-law migration scale for most layers,
switched to FPTQ's logarithmic equalization scale
(``s_j = ref * log2(1 + max(|X_j|) / ref)``, ``ref`` the layer's per-channel
geometric-mean activation max) on layers whose activation has a channel at
least ``outlier_ratio_threshold`` times that geometric mean -- a lossless
pre-conditioning transform meant to run ahead of separate W4 weight-only and
W8A8 activation quantizers.
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


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _with_extra_output(model, name):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    existing = {o.name for o in out.graph.output}
    if name not in existing:
        out.graph.output.append(onnx.ValueInfoProto(name=name))
    return out


def _mild_calibration(K=64, num_samples=64, outlier_channels=(3, 7), seed=1):
    # Mild outliers -- well under the default outlier_ratio_threshold -- so
    # the layer stays "tractable" and keeps the power-law scale.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for i, c in enumerate(outlier_channels):
        x[:, c] *= 3.0 + 0.5 * i
    return x


def _severe_outlier_calibration(K=64, num_samples=64, outlier_channel=3, seed=1):
    # One channel ~1000x the rest -- FPTQ's own motivating "intractable
    # layer" scenario.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, outlier_channel] *= 1000.0
    return x


def test_fptq_output_matches_float_almost_exactly():
    # Like SmoothQuant, FPTQ's migration is exact real-number math regardless
    # of which scale formula (power-law or logarithmic) is used.
    model = _matmul_model(K=64, N=16, seed=0)
    x = _severe_outlier_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    fptq_model = onnxsim.apply_fptq(model, calibration_data=calibration_data)
    onnx.checker.check_model(fptq_model)
    assert any(n.op_type == "Mul" for n in fptq_model.graph.node)

    float_out = _run(model, {"X": x})
    fptq_out = _run(fptq_model, {"X": x}, output_names=["Y"])
    assert np.all(np.isfinite(fptq_out["Y"]))
    assert _rel_l2(float_out["Y"], fptq_out["Y"]) < 1e-4


def test_fptq_intractable_layer_tempers_weight_growth_vs_full_linear():
    # On a severe single-channel outlier, full linear equalization
    # (SmoothQuant alpha=1, s_j = max(|X_j|)) forces the matching weight row
    # to blow up by the same ~1000x factor. FPTQ's logarithmic scale should
    # compensate far more gently (log2(1 + 1000) ~= 10x) while still
    # meaningfully shrinking the outlier channel's own migrated activation
    # range compared to doing nothing at all.
    K = 64
    model = _matmul_model(K=K, N=16, seed=2)
    x = _severe_outlier_calibration(K=K, num_samples=64, outlier_channel=3, seed=3)
    calibration_data = [{"X": x}]

    fptq_model = onnxsim.apply_fptq(model, calibration_data=calibration_data)
    linear_model = onnxsim.apply_smoothquant(
        model, calibration_data=calibration_data, alpha=1.0
    )

    fptq_w = onnx.numpy_helper.to_array(fptq_model.graph.initializer[0])
    linear_w = onnx.numpy_helper.to_array(linear_model.graph.initializer[0])
    float_w = onnx.numpy_helper.to_array(model.graph.initializer[0])

    outlier_row_growth_fptq = np.abs(fptq_w[3]).max() / max(
        np.abs(float_w[3]).max(), 1e-9
    )
    outlier_row_growth_linear = np.abs(linear_w[3]).max() / max(
        np.abs(float_w[3]).max(), 1e-9
    )
    assert outlier_row_growth_fptq < outlier_row_growth_linear

    mul_node = next(n for n in fptq_model.graph.node if n.op_type == "Mul")
    probe_model = _with_extra_output(fptq_model, mul_node.output[0])
    result = _run(probe_model, {"X": x}, output_names=[mul_node.output[0]])
    migrated = result[mul_node.output[0]]
    migrated_range = np.abs(migrated).max(axis=0)
    float_range = np.abs(x.astype(np.float64)).max(axis=0)
    # Still meaningfully reduces the outlier channel's own dominance over
    # the rest of the layer, just not all the way to a uniform 1.0 the way
    # alpha=1 would.
    assert (migrated_range.max() / migrated_range.min()) < (
        float_range.max() / float_range.min()
    )


def test_fptq_tractable_layer_matches_smoothquant_power_law():
    K = 64
    model = _matmul_model(K=K, N=16, seed=4)
    x = _mild_calibration(K=K, num_samples=64, seed=5)
    calibration_data = [{"X": x}]

    fptq_model = onnxsim.apply_fptq(model, calibration_data=calibration_data, alpha=0.5)
    sq_model = onnxsim.apply_smoothquant(
        model, calibration_data=calibration_data, alpha=0.5
    )
    fptq_w = onnx.numpy_helper.to_array(fptq_model.graph.initializer[0])
    sq_w = onnx.numpy_helper.to_array(sq_model.graph.initializer[0])
    np.testing.assert_allclose(fptq_w, sq_w, rtol=1e-5)


def test_fptq_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 96, 12
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
    x = _severe_outlier_calibration(K=K, num_samples=32, outlier_channel=10, seed=7)
    calibration_data = [{"X": x}]

    fptq_model = onnxsim.apply_fptq(model, calibration_data=calibration_data)
    onnx.checker.check_model(fptq_model)

    float_out = _run(model, {"X": x})
    fptq_out = _run(fptq_model, {"X": x}, output_names=["Y"])
    assert _rel_l2(float_out["Y"], fptq_out["Y"]) < 1e-4


def test_fptq_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_fptq(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_fptq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result = onnxsim.apply_fptq(
        model, calibration_data=[{"X": np.zeros((4, 64), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
