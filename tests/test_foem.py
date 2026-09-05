"""Tests for ``onnxsim.apply_foem`` (FOEM, see ``onnxsim/foem.py``) --
GPTQ's own sequential Hessian-compensated rounding, with an added
first-order correction for the accumulated deviation between a column's
current pre-quantization value and its own untouched original value (a
drift GPTQ's own second-order-only correction doesn't account for).
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


def _matmul_model(K=64, N=16, seed=0):
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


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # Same motivating scenario onnxsim.apply_gptq's own tests use: input
    # channels correlated via a handful of latent factors, so off-diagonal
    # Hessian terms (and, here, accumulated cross-column drift) actually
    # matter.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def test_foem_beta_zero_matches_plain_gptq_exactly():
    # foem_beta=0 drops the added first-order term entirely, so FOEM's own
    # codes must exactly reproduce onnxsim.apply_gptq's.
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    foem_model = onnxsim.apply_foem(
        model, quant, calibration_data=calibration_data, foem_beta=0.0
    )

    w_gptq = _dequantize_int4(gptq_model)
    w_foem = _dequantize_int4(foem_model)
    np.testing.assert_array_equal(w_gptq, w_foem)


def test_foem_reduces_reconstruction_error_vs_plain_gptq():
    # A long reduction dimension (many columns) so accumulated first-order
    # drift actually has room to build up across the pass -- FOEM's own
    # motivating scenario. Verified empirically (see onnxsim/foem.py's own
    # docstring): this module's default foem_beta gives a small, repeatable
    # improvement on this scenario -- a larger beta was swept and found to
    # *increase* error instead, so this is not a "bigger beta is always
    # better" claim, just the honest, verified behavior at the default.
    model = _matmul_model(K=256, N=16, seed=2)
    x = _correlated_calibration(K=256, num_samples=128, rank=12, seed=3)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    gptq_model = onnxsim.apply_gptq(
        model, quant, calibration_data=calibration_data, proc_block_size=16
    )
    w_gptq = _dequantize_int4(gptq_model)
    gptq_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_gptq)

    foem_model = onnxsim.apply_foem(
        model,
        quant,
        calibration_data=calibration_data,
        proc_block_size=16,
    )
    w_foem = _dequantize_int4(foem_model)
    foem_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_foem)

    assert foem_err < gptq_err


def test_foem_reduces_reconstruction_error_vs_rtn():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    rtn_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_rtn)

    foem_model = onnxsim.apply_foem(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(foem_model)
    w_foem = _dequantize_int4(foem_model)
    foem_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_foem)

    assert foem_err < rtn_err


def test_foem_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _correlated_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    foem_model = onnxsim.apply_foem(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(foem_model)

    (float_y,) = _run(model, {"X": x})
    (foem_y,) = _run(foem_model, {"X": x})
    assert np.all(np.isfinite(foem_y))
    assert _rel_l2(float_y, foem_y) < 0.25


def test_foem_preserves_scale_and_shape():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    quant_dq = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    before_scale = onnx.numpy_helper.to_array(
        next(t for t in quant.graph.initializer if t.name == quant_dq.input[1])
    )
    foem_model = onnxsim.apply_foem(model, quant, calibration_data=calibration_data)
    foem_dq = next(n for n in foem_model.graph.node if n.op_type == "DequantizeLinear")
    after_scale = onnx.numpy_helper.to_array(
        next(t for t in foem_model.graph.initializer if t.name == foem_dq.input[1])
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = next(
        t for t in foem_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    assert list(wq.dims) == [32, 8]


def test_foem_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=6)
    x = _correlated_calibration(K=32, num_samples=16, rank=2, seed=7) * 3
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    foem_model = onnxsim.apply_foem(model, quant, calibration_data=calibration_data)
    wq = next(
        t for t in foem_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
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


def test_foem_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_foem(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
