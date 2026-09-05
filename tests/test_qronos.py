"""Tests for ``onnxsim.apply_qronos`` (Qronos, see ``onnxsim/qronos.py``) --
generalizes ``onnxsim.apply_gptq`` by processing layers in forward-execution
order and correcting each layer for the error already baked into its input
by previously-quantized upstream layers, in addition to its own weight-
rounding error.
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


def _two_layer_model(K1=64, N1=32, N2=16, seed=0):
    # quantize_weight_only_int4 only quantizes a MatMul whose activation
    # input has a statically-known float32 type -- true automatically for a
    # graph input/output, but an intermediate tensor like Y1 needs its own
    # explicit value_info entry (the parser text form alone doesn't add one).
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((N1, N2)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K1}] X) => (float[batch,{N2}] Y2)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(Y1, W2)
        }}
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, ["batch", N1])
    )
    return model


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # Same motivating scenario as test_gptq.py's own helper: correlated
    # input channels give the Hessian-based correction something to
    # exploit that independent per-element rounding can't.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _dequantize_int4(model, matmul_output_name):
    matmul = next(n for n in model.graph.node if n.output[0] == matmul_output_name)
    dq_node = next(
        n
        for n in model.graph.node
        if n.op_type == "DequantizeLinear" and n.output[0] == matmul.input[1]
    )
    # Fetch Wq/Ws by the DequantizeLinear node's own input names, not by
    # scanning for "some tensor of this dtype": quantize_weight_only_int4
    # never prunes the original (now-dead) float32 weight initializer, so a
    # dtype-only scan can silently grab that instead of the real scale.
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


def test_qronos_reduces_two_layer_reconstruction_error_on_average():
    # The paper's headline claim: correcting a downstream layer for the
    # error already baked into its input by an upstream layer's own
    # quantization (Qronos) should compound less than treating each layer
    # independently from clean float activations (plain GPTQ). Averaged
    # over several random weight/calibration seeds, since any single toy
    # seed can go either way.
    gptq_errs = []
    qronos_errs = []
    for seed in range(6):
        model = _two_layer_model(K1=64, N1=32, N2=16, seed=seed)
        x = _correlated_calibration(K=64, num_samples=64, rank=6, seed=seed + 100)
        calibration_data = [{"X": x}]

        quant = onnxsim.quantize_weight_only_int4(model)
        gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
        qronos_model = onnxsim.apply_qronos(
            model, quant, calibration_data=calibration_data
        )
        onnx.checker.check_model(gptq_model)
        onnx.checker.check_model(qronos_model)

        w1_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(
            np.float64
        )
        w2_float = onnx.numpy_helper.to_array(model.graph.initializer[1]).astype(
            np.float64
        )
        y_float = x.astype(np.float64) @ w1_float @ w2_float

        w1_gptq = _dequantize_int4(gptq_model, "Y1")
        w2_gptq = _dequantize_int4(gptq_model, "Y2")
        y_gptq = x.astype(np.float64) @ w1_gptq @ w2_gptq

        w1_qronos = _dequantize_int4(qronos_model, "Y1")
        w2_qronos = _dequantize_int4(qronos_model, "Y2")
        y_qronos = x.astype(np.float64) @ w1_qronos @ w2_qronos

        gptq_errs.append(np.linalg.norm(y_float - y_gptq))
        qronos_errs.append(np.linalg.norm(y_float - y_qronos))

    assert np.mean(qronos_errs) < np.mean(gptq_errs)


def test_qronos_first_layer_matches_plain_gptq():
    # Qronos's own correction is exactly a no-op for a layer with no
    # already-quantized upstream layer feeding it (X_quant == X_float,
    # so W_opt == W_float exactly) -- this module's docstring's own claim
    # that Qronos strictly generalizes, rather than replaces, GPTQ.
    model = _two_layer_model(K1=32, N1=16, N2=8, seed=1)
    x = _correlated_calibration(K=32, num_samples=32, rank=4, seed=2)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    qronos_model = onnxsim.apply_qronos(model, quant, calibration_data=calibration_data)

    w1_gptq = _dequantize_int4(gptq_model, "Y1")
    w1_qronos = _dequantize_int4(qronos_model, "Y1")
    np.testing.assert_allclose(w1_gptq, w1_qronos, rtol=1e-8, atol=1e-10)


def test_qronos_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _correlated_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    qronos_model = onnxsim.apply_qronos(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(qronos_model)

    (float_y,) = _run(model, {"X": x})
    (qronos_y,) = _run(qronos_model, {"X": x})
    assert np.all(np.isfinite(qronos_y))
    assert _rel_l2(float_y, qronos_y) < 0.25


def test_qronos_preserves_scale_and_shape():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    quant_dq = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    before_scale = onnx.numpy_helper.to_array(
        next(t for t in quant.graph.initializer if t.name == quant_dq.input[1])
    )
    qronos_model = onnxsim.apply_qronos(model, quant, calibration_data=calibration_data)
    qronos_dq = next(
        n for n in qronos_model.graph.node if n.op_type == "DequantizeLinear"
    )
    after_scale = onnx.numpy_helper.to_array(
        next(t for t in qronos_model.graph.initializer if t.name == qronos_dq.input[1])
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = next(
        t
        for t in qronos_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
    )
    assert list(wq.dims) == [32, 8]


def test_qronos_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=6)
    x = _correlated_calibration(K=32, num_samples=16, rank=2, seed=7) * 3
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    qronos_model = onnxsim.apply_qronos(model, quant, calibration_data=calibration_data)
    wq = next(
        t
        for t in qronos_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
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


def test_qronos_handles_dead_input_channel():
    # A channel with zero variance in the calibration data (H's diagonal is
    # exactly 0 there) must not blow up the Hessian inversion.
    model = _matmul_model(K=32, N=8, seed=10)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=12)
    x[:, 5] = 0.0  # dead channel
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    qronos_model = onnxsim.apply_qronos(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(qronos_model)

    (qronos_y,) = _run(qronos_model, {"X": x})
    assert np.all(np.isfinite(qronos_y))


def test_qronos_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_qronos(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
