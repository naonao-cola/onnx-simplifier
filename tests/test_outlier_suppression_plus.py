"""Tests for ``onnxsim.apply_outlier_suppression_plus`` (Outlier
Suppression+, see ``onnxsim/outlier_suppression_plus.py``) -- shifts each
MatMul/Gemm activation channel to re-center it around zero
(``z_j = (max(X_j) + min(X_j)) / 2``) before applying
:mod:`onnxsim.smoothquant`'s own scale migration, folding the shift's
constant contribution back into the layer's output via a new ``Add`` so the
whole transformation stays exact pre-quantization.
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


def _lopsided_calibration(K=64, num_samples=64, lopsided_channels=(3, 7), seed=1):
    # Outlier Suppression+'s own motivating scenario: a few channels sitting
    # mostly on one side of zero (a large mean offset), which a symmetric
    # per-channel scale (SmoothQuant's own migration) cannot shrink at all.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for i, c in enumerate(lopsided_channels):
        x[:, c] = x[:, c] * 2.0 + (20.0 + 10.0 * i)  # shifted well off zero
    return x


def test_outlier_suppression_plus_output_matches_float_almost_exactly():
    # Like SmoothQuant, this is exact real-number math (a shift folded back
    # in via Add, a scale folded via reciprocal Mul/Mul) -- no quantization
    # happens here, so the output should match the float model far more
    # tightly than any lossy INT4/INT8 scheme's tolerance.
    model = _matmul_model(K=64, N=16, seed=0)
    x = _lopsided_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    osp_model = onnxsim.apply_outlier_suppression_plus(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(osp_model)
    assert any(n.op_type == "Sub" for n in osp_model.graph.node)
    assert any(n.op_type == "Mul" for n in osp_model.graph.node)
    assert any(n.op_type == "Add" for n in osp_model.graph.node)

    float_out = _run(model, {"X": x})
    osp_out = _run(osp_model, {"X": x}, output_names=["Y"])
    assert np.all(np.isfinite(osp_out["Y"]))
    assert _rel_l2(float_out["Y"], osp_out["Y"]) < 1e-4


def test_outlier_suppression_plus_shift_recenters_lopsided_channels():
    K = 64
    model = _matmul_model(K=K, N=16, seed=2)
    lopsided = (3, 7)
    x = _lopsided_calibration(K=K, num_samples=64, lopsided_channels=lopsided, seed=3)
    calibration_data = [{"X": x}]

    osp_model = onnxsim.apply_outlier_suppression_plus(
        model, calibration_data=calibration_data
    )
    sub_node = next(n for n in osp_model.graph.node if n.op_type == "Sub")
    probe_model = _with_extra_output(osp_model, sub_node.output[0])
    result = _run(probe_model, {"X": x}, output_names=[sub_node.output[0]])
    shifted = result[sub_node.output[0]]

    # After shifting, every channel's own (max + min) / 2 should collapse to
    # ~0 -- including the originally lopsided ones, which the shift alone
    # (with no scaling at all) is exactly designed to fix.
    midpoints = (shifted.max(axis=0) + shifted.min(axis=0)) / 2.0
    assert np.allclose(midpoints, 0.0, atol=1e-3)

    original_midpoints = (x.max(axis=0) + x.min(axis=0)) / 2.0
    for c in lopsided:
        assert abs(original_midpoints[c]) > 5.0  # confirms the scenario itself


def test_outlier_suppression_plus_reduces_range_more_than_smoothquant_alone():
    # The shift should make a lopsided channel's post-migration range no
    # larger than plain SmoothQuant would leave it (SmoothQuant, having no
    # shift, cannot narrow a lopsided channel's max(|X|) at all -- it can
    # only move that difficulty into the weight, never shrink it).
    K = 64
    model = _matmul_model(K=K, N=16, seed=4)
    x = _lopsided_calibration(K=K, num_samples=64, lopsided_channels=(3, 7), seed=5)
    calibration_data = [{"X": x}]

    osp_model = onnxsim.apply_outlier_suppression_plus(
        model, calibration_data=calibration_data
    )
    mul_node = next(n for n in osp_model.graph.node if n.op_type == "Mul")
    probe_model = _with_extra_output(osp_model, mul_node.output[0])
    result = _run(probe_model, {"X": x}, output_names=[mul_node.output[0]])
    osp_scaled = result[mul_node.output[0]]

    sq_model = onnxsim.apply_smoothquant(model, calibration_data=calibration_data)
    sq_mul = next(n for n in sq_model.graph.node if n.op_type == "Mul")
    sq_probe = _with_extra_output(sq_model, sq_mul.output[0])
    sq_result = _run(sq_probe, {"X": x}, output_names=[sq_mul.output[0]])
    sq_scaled = sq_result[sq_mul.output[0]]

    osp_spread = np.abs(osp_scaled).max(axis=0)
    sq_spread = np.abs(sq_scaled).max(axis=0)
    assert osp_spread.max() / osp_spread.min() < sq_spread.max() / sq_spread.min()


def test_outlier_suppression_plus_gemm_transb():
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
    x = _lopsided_calibration(K=K, num_samples=32, lopsided_channels=(10, 50), seed=7)
    calibration_data = [{"X": x}]

    osp_model = onnxsim.apply_outlier_suppression_plus(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(osp_model)

    float_out = _run(model, {"X": x})
    osp_out = _run(osp_model, {"X": x}, output_names=["Y"])
    assert _rel_l2(float_out["Y"], osp_out["Y"]) < 1e-4


def test_outlier_suppression_plus_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_outlier_suppression_plus(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_outlier_suppression_plus_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result = onnxsim.apply_outlier_suppression_plus(
        model, calibration_data=[{"X": np.zeros((4, 64), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_outlier_suppression_plus_skips_non_2d_activation():
    # A 3-D activation (e.g. [batch, seq, hidden], typical of an
    # ONNX-exported transformer) isn't a plain 2-D tensor -- matches this
    # module's own documented scope, same as onnxsim.apply_smoothquant.
    K, N = 16, 8
    rng = np.random.default_rng(8)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    x = rng.standard_normal((2, 3, K)).astype(np.float32)
    result = onnxsim.apply_outlier_suppression_plus(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()
