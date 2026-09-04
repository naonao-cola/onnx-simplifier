"""Tests for ``onnxsim.apply_affinequant`` (AffineQuant, see
``onnxsim/affinequant.py``) -- grid-searches a per-block Learnable Weight
Clipping ratio, then chooses among a no-transform, a diagonal (OmniQuant
-equivalent), or a block-diagonal-affine Learnable Equivalent
Transformation, on top of an existing ``quantize_weight_only_int4``
-quantized model.
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
        [_f32(weight, "W")],
    )


def _correlated_outlier_calibration(K=64, num_samples=64, outlier_dims=(3, 7), seed=1):
    # A calibration set with a nonzero per-channel mean (the shift should
    # help), a couple of channels with much larger magnitude (LWC's
    # clipping and the diagonal scale should help), and cross-channel
    # correlation baked in (adjacent channels are linear combinations of a
    # shared latent factor) -- a case a *diagonal* transform structurally
    # cannot exploit (it can only rescale each channel on its own) but a
    # block-affine transform can, by rotating into the block's own
    # covariance eigenbasis.
    rng = np.random.default_rng(seed)
    num_blocks = K // 8
    latent = rng.standard_normal((num_samples, num_blocks)).astype(np.float32)
    mixing = rng.standard_normal((num_blocks, 8)).astype(np.float32)
    x = (latent[:, :, None] * mixing[None, :, :]).reshape(num_samples, K)
    x = x + rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    x = x.astype(np.float32) + 2.0
    for c in outlier_dims:
        x[:, c] *= 20.0
    return x


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    codes = onnx.numpy_helper.to_array(wq).astype(np.float64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def test_affinequant_reduces_reconstruction_error_with_correlated_outliers():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_outlier_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=8
    )
    onnx.checker.check_model(aq_model)

    (float_y,) = _run(model, {"X": x})
    (aq_y,) = _run(aq_model, {"X": x})
    aq_err = np.linalg.norm(y_float - aq_y.astype(np.float64))
    assert aq_err < rtn_err


def test_affinequant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _correlated_outlier_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    aq_model = onnxsim.apply_affinequant(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
        affine_block_size=8,
    )
    onnx.checker.check_model(aq_model)

    (float_y,) = _run(model, {"X": x})
    (aq_y,) = _run(aq_model, {"X": x})
    assert np.all(np.isfinite(aq_y))
    assert _rel_l2(float_y, aq_y) < 0.25


def test_affinequant_never_worse_than_diagonal_omniquant():
    # AffineQuant's own block-affine candidate is a strict superset of
    # OmniQuant's diagonal LET (R=I is always reachable as a degenerate
    # block-affine choice, and is exactly what candidate 2 already
    # searches on its own), so AffineQuant's chosen reconstruction error
    # can never exceed OmniQuant's, on the same data/model/search grid.
    model = _matmul_model(K=64, N=16, seed=4)
    x = _correlated_outlier_calibration(K=64, num_samples=48, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    oq_model = onnxsim.apply_omniquant(model, quant, calibration_data=calibration_data)
    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=8
    )

    (oq_y,) = _run(oq_model, {"X": x})
    (aq_y,) = _run(aq_model, {"X": x})
    oq_err = np.linalg.norm(y_float - oq_y.astype(np.float64))
    aq_err = np.linalg.norm(y_float - aq_y.astype(np.float64))
    assert aq_err <= oq_err + 1e-6


def test_affinequant_never_worse_than_plain_rtn():
    # Every candidate this module searches (LWC-only, diagonal LET,
    # block-affine LET) always includes "no change" / "no transform" as a
    # reachable choice (clip_ratio=1.0 for LWC, alpha=0 for both LET
    # variants), so the chosen reconstruction error can never be worse
    # than plain RTN's, only equal or better -- mirroring
    # ``onnxsim.apply_omniquant``'s own analogous guarantee.
    model = _matmul_model(K=32, N=8, seed=6)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 32)).astype(np.float32)  # zero-mean, no outliers
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=8
    )

    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float
    w_rtn = _dequantize_int4(quant)
    rtn_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_rtn)

    (aq_y,) = _run(aq_model, {"X": x})
    aq_err = np.linalg.norm(y_float - aq_y.astype(np.float64))
    assert aq_err <= rtn_err + 1e-6


def test_affinequant_rotation_matrix_is_block_diagonal_and_orthogonal():
    # Directly verifies the numerical contract this module's docstring
    # promises: the inserted rotation is exactly block-diagonal (zero
    # outside affine_block_size x affine_block_size blocks along the
    # diagonal) and exactly orthogonal (R @ R.T == I), checked against the
    # ONNX initializers themselves with a tight relative tolerance -- not
    # via an onnxruntime round trip, since onnxruntime's own MatMul kernel
    # reduction order is not bit-exact across CPU architectures.
    K, N, block = 32, 8, 8
    model = _matmul_model(K=K, N=N, seed=8)
    x = _correlated_outlier_calibration(
        K=K, num_samples=64, outlier_dims=(2, 9), seed=9
    )
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=block
    )
    onnx.checker.check_model(aq_model)

    matmul_nodes = [n for n in aq_model.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) == 1
    rotation_name = matmul_nodes[0].input[1]
    rotation = onnx.numpy_helper.to_array(
        next(t for t in aq_model.graph.initializer if t.name == rotation_name)
    ).astype(np.float64)

    assert rotation.shape == (K, K)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(K), atol=1e-4, rtol=1e-4)
    off_block = rotation.copy()
    for start in range(0, K, block):
        stop = start + block
        off_block[start:stop, start:stop] = 0.0
    np.testing.assert_allclose(off_block, np.zeros((K, K)), atol=1e-6)


def test_affinequant_odd_k_skips_block_affine_candidate():
    # K=48 is not evenly divisible by affine_block_size=32, so the
    # block-affine candidate must be skipped for that layer (falling back
    # to the diagonal or LWC-only candidate) rather than erroring.
    model = _matmul_model(K=48, N=8, seed=10)
    x = _correlated_outlier_calibration(
        K=48, num_samples=32, outlier_dims=(4,), seed=11
    )
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=32
    )
    onnx.checker.check_model(aq_model)
    assert not any(n.op_type == "MatMul" for n in aq_model.graph.node)


def test_affinequant_gemm_transb():
    rng = np.random.default_rng(12)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _correlated_outlier_calibration(
        K=K, num_samples=32, outlier_dims=(10, 50), seed=13
    )
    calibration_data = [{"X": x}]

    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=8
    )
    onnx.checker.check_model(aq_model)

    (float_y,) = _run(model, {"X": x})
    (aq_y,) = _run(aq_model, {"X": x})
    assert _rel_l2(float_y, aq_y) < 0.25


def test_affinequant_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=14)
    x = _correlated_outlier_calibration(
        K=32, num_samples=16, outlier_dims=(1,), seed=15
    )
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    aq_model = onnxsim.apply_affinequant(
        model, quant, calibration_data=calibration_data, affine_block_size=8
    )
    wq = next(
        t for t in aq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    assert np.all(codes >= -7) and np.all(codes <= 7)


def test_affinequant_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_affinequant(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
