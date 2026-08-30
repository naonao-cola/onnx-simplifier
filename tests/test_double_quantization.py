"""Tests for ``onnxsim.apply_double_quantization`` -- see
``onnxsim/double_quantization.py`` for the technique (QLoRA-style
second-level UINT8 quantization of an already-quantized model's own
per-block/per-channel scale tensors).
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _int4_codes_tensor(codes, name="codes"):
    # onnx.helper (not the text parser) because these need to be byte-equal
    # to onnxsim.adaround._pack_int4's own packed-nibble encoding: the
    # parser has no INT4 tensor literal syntax that lets us hand it
    # pre-packed raw_data directly.
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = onnx.TensorProto.INT4
    tensor.dims.extend(codes.shape)

    from onnxsim.adaround import _pack_int4

    tensor.raw_data = _pack_int4(codes.astype(np.int64))
    return tensor


def _blockwise_int4_model(k=32, n=8, block_size=8, seed=0):
    # Mirrors quantize_weight_only_int4's own output shape: INT4 codes
    # [K, N], scale [K/block_size, N], DequantizeLinear(axis=0, block_size).
    rng = np.random.default_rng(seed)
    codes = rng.integers(-7, 8, size=(k, n)).astype(np.int8)
    scale = (rng.random((k // block_size, n)).astype(np.float32) + 0.1) * 0.05

    return _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,{n}] Y)
        {{
          w_hat = DequantizeLinear<axis=0, block_size={block_size}>(codes, scale)
          Y = MatMul(X, w_hat)
        }}
        """,
        initializer=[
            _int4_codes_tensor(codes),
            onnx.numpy_helper.from_array(scale, name="scale"),
        ],
    )


def test_double_quantization_inserts_nested_dequantize_and_keeps_output_close():
    model = _blockwise_int4_model(k=64, n=8, block_size=8, seed=0)
    onnx.checker.check_model(model)
    q = onnxsim.apply_double_quantization(model, min_elements=4)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("DequantizeLinear") == 2

    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    rel = np.linalg.norm(float_y - q_y) / max(np.linalg.norm(float_y), 1e-6)
    assert rel < 0.05


def test_double_quantization_scale_codes_are_uint8():
    model = _blockwise_int4_model(k=64, n=8, block_size=8, seed=2)
    q = onnxsim.apply_double_quantization(model, min_elements=4)
    codes_init = next(t for t in q.graph.initializer if t.name.endswith("_dblq_codes"))
    assert codes_init.data_type == onnx.TensorProto.UINT8
    codes = onnx.numpy_helper.to_array(codes_init)
    assert codes.min() >= 0 and codes.max() <= 255


def test_double_quantization_declines_small_scale_tensor():
    # Default min_elements=64; an 8-element scale tensor isn't worth it.
    model = _blockwise_int4_model(k=64, n=1, block_size=8, seed=3)
    q = onnxsim.apply_double_quantization(model)  # default min_elements
    assert q.SerializeToString() == model.SerializeToString()


def test_double_quantization_declines_dynamic_scale_input():
    # A scale fed by a graph input (not a constant initializer) -- e.g.
    # quantize_kv_cache's Value-style per-token scale stream -- is left
    # untouched: there is nothing to fold into a constant meta-scale.
    codes = np.zeros((8, 8), dtype=np.int8)
    model = _model(
        """
        g (float[8,8] scale) => (float[8,8] w_hat)
        {
          w_hat = DequantizeLinear(codes, scale)
        }
        """,
        initializer=[_int4_codes_tensor(codes)],
    )
    q = onnxsim.apply_double_quantization(model, min_elements=4)
    assert q.SerializeToString() == model.SerializeToString()


def test_double_quantization_noop_without_dequantize_linear():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_double_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_double_quantization_composes_with_spinquant():
    rng = np.random.default_rng(4)
    K, N = 32, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[onnx.numpy_helper.from_array(weight, name="W")],
    )
    spun = onnxsim.apply_spinquant(model, block_size=8, num_samples=16, seed=5)
    both = onnxsim.apply_double_quantization(spun, min_elements=4)
    onnx.checker.check_model(both)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (spun_y,) = _run(spun, {"X": x})
    (both_y,) = _run(both, {"X": x})
    rel = np.linalg.norm(spun_y - both_y) / max(np.linalg.norm(spun_y), 1e-6)
    assert rel < 0.05
