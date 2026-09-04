"""Tests for ``onnxsim.apply_tesseraq`` (TesseraQ's Progressive Adaptive
Rounding + joint scale optimization, see ``onnxsim/tesseraq.py``) --
progressively hardens each INT4-quantized MatMul/Gemm layer's per-element
rounding decision in coarse-to-fine rounds, jointly optimizing each weight
block's own dequantization scale, to minimize that layer's real
reconstruction error.
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


def _matmul_int4_models(K=64, N=16, batch=4, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    float_model = _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    return float_model, quant_model


def _dequantize_int4(quant_model):
    """Decodes the DequantizeLinear(Wq, Ws)-fed MatMul/Gemm's weight in
    ``quant_model`` back to a dense float array, using plain numpy against
    the initializers directly (independent of tesseraq.py's own internal
    math)."""
    dq_node = next(n for n in quant_model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in quant_model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in quant_model.graph.initializer if t.name == dq_node.input[1])
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
    scale_full = scale_full[tuple(slicer)]
    return codes, codes * scale_full


def test_tesseraq_reduces_reconstruction_error_vs_round_to_nearest():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, batch=32, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    w_float = onnx.numpy_helper.to_array(float_model.graph.initializer[0]).astype(
        np.float64
    )
    _, w_rtn = _dequantize_int4(quant_model)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    tesseraq_model = onnxsim.apply_tesseraq(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_iterations=200,
        par_rounds=4,
    )
    _, w_par = _dequantize_int4(tesseraq_model)
    y_par = x.astype(np.float64) @ w_par
    par_err = np.linalg.norm(y_float - y_par)

    assert par_err < rtn_err


def test_tesseraq_low_bit_width_beats_naive_clipping_to_same_range():
    # Naively narrowing to 3-bit by clipping the existing INT4-range-
    # calibrated RTN codes into [-3, 3] (same scale, unadjusted) wastes most
    # of the narrower range's own resolution -- exactly the failure mode
    # this module's docstring documents its jointly-optimized scale as
    # solving. TesseraQ at num_bits=3 should reconstruct meaningfully
    # better than that naive same-range baseline.
    float_model, quant_model = _matmul_int4_models(K=64, N=16, batch=32, seed=11)
    rng = np.random.default_rng(12)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    w_float = onnx.numpy_helper.to_array(float_model.graph.initializer[0]).astype(
        np.float64
    )
    y_float = x.astype(np.float64) @ w_float

    codes_rtn, w_rtn = _dequantize_int4(quant_model)
    dq_node = next(n for n in quant_model.graph.node if n.op_type == "DequantizeLinear")
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)
    ws = next(t for t in quant_model.graph.initializer if t.name == dq_node.input[1])
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)[:, : codes_rtn.shape[1]]
    codes_naive_3bit = np.clip(codes_rtn.astype(np.float64), -3, 3)
    y_naive = x.astype(np.float64) @ (codes_naive_3bit * scale_full)
    naive_err = np.linalg.norm(y_float - y_naive)

    tesseraq_model = onnxsim.apply_tesseraq(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_bits=3,
        num_iterations=300,
        par_rounds=4,
    )
    codes_tq, w_tq = _dequantize_int4(tesseraq_model)
    assert np.all(codes_tq >= -3) and np.all(codes_tq <= 3)
    y_tq = x.astype(np.float64) @ w_tq
    tq_err = np.linalg.norm(y_float - y_tq)

    assert tq_err < naive_err


def test_tesseraq_output_stays_close_to_float_via_onnxruntime():
    float_model, quant_model = _matmul_int4_models(K=64, N=16, batch=16, seed=3)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    tesseraq_model = onnxsim.apply_tesseraq(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=200
    )
    onnx.checker.check_model(tesseraq_model)

    (float_y,) = _run(float_model, {"X": x})
    (tq_y,) = _run(tesseraq_model, {"X": x})
    assert np.all(np.isfinite(tq_y))
    assert _rel_l2(float_y, tq_y) < 0.25


def test_tesseraq_all_codes_hardened_and_in_range():
    float_model, quant_model = _matmul_int4_models(K=32, N=8, seed=7)
    rng = np.random.default_rng(8)
    calibration_data = [{"X": rng.standard_normal((4, 32)).astype(np.float32) * 3}]

    tesseraq_model = onnxsim.apply_tesseraq(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_bits=3,
        num_iterations=120,
        par_rounds=3,
    )
    codes, _ = _dequantize_int4(tesseraq_model)
    assert np.all(codes == np.round(codes))  # every element is a hard integer
    assert np.all(codes >= -3) and np.all(codes <= 3)


def test_tesseraq_rejects_out_of_range_num_bits():
    float_model, quant_model = _matmul_int4_models(K=32, N=8, seed=9)
    calibration_data = [{"X": np.zeros((4, 32), dtype=np.float32)}]
    with pytest.raises(ValueError):
        onnxsim.apply_tesseraq(
            float_model, quant_model, calibration_data=calibration_data, num_bits=5
        )
    with pytest.raises(ValueError):
        onnxsim.apply_tesseraq(
            float_model, quant_model, calibration_data=calibration_data, num_bits=1
        )


def test_tesseraq_gemm_transb_with_bias():
    rng = np.random.default_rng(9)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _model(
        f"""
        g (float[8,{K}] X) => (float[8,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    onnx.checker.check_model(quant_model)

    x = rng.standard_normal((8, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    tesseraq_model = onnxsim.apply_tesseraq(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=150
    )
    onnx.checker.check_model(tesseraq_model)

    (float_y,) = _run(float_model, {"X": x})
    (tq_y,) = _run(tesseraq_model, {"X": x})
    assert _rel_l2(float_y, tq_y) < 0.25


def test_tesseraq_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_tesseraq(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
