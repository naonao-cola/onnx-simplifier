"""Tests for ``onnxsim.apply_double_quantization_cpp`` -- the C++-backed
port of ``onnxsim.apply_double_quantization`` (see
``onnxsim/passes/double_quantization.h``). The C++ pass hardcodes
``min_elements=64`` (no per-call parameter, unlike the pure-Python version)
and mints anonymous initializer names, so these tests locate tensors by
shape/dtype rather than by the ``_dblq_*`` naming convention
``test_double_quantization.py`` relies on.
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
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = onnx.TensorProto.INT4
    tensor.dims.extend(codes.shape)

    from onnxsim.adaround import _pack_int4

    tensor.raw_data = _pack_int4(codes.astype(np.int64))
    return tensor


def _blockwise_int4_model(k=64, n=8, block_size=8, seed=0):
    # Mirrors quantize_weight_only_int4's own output shape: INT4 codes
    # [K, N], scale [K/block_size, N], DequantizeLinear(axis=0, block_size).
    # k // block_size * n must be >= 64 (the C++ pass's fixed min_elements)
    # for these tests to actually exercise the rewrite.
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


def test_cpp_double_quantization_inserts_nested_dequantize_and_keeps_output_close():
    model = _blockwise_int4_model(k=512, n=8, block_size=8, seed=0)  # 64x8=512 scales
    onnx.checker.check_model(model)
    q = onnxsim.apply_double_quantization_cpp(model)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("DequantizeLinear") == 2

    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    rel = np.linalg.norm(float_y - q_y) / max(np.linalg.norm(float_y), 1e-6)
    assert rel < 0.05


def test_cpp_double_quantization_scale_codes_are_uint8():
    model = _blockwise_int4_model(k=512, n=8, block_size=8, seed=2)
    q = onnxsim.apply_double_quantization_cpp(model)
    uint8_inits = [
        t for t in q.graph.initializer if t.data_type == onnx.TensorProto.UINT8
    ]
    assert len(uint8_inits) == 1
    codes = onnx.numpy_helper.to_array(uint8_inits[0])
    assert codes.min() >= 0 and codes.max() <= 255


def test_cpp_double_quantization_declines_small_scale_tensor():
    # The C++ pass's fixed min_elements=64; an 8-element scale tensor isn't
    # worth it.
    model = _blockwise_int4_model(k=64, n=1, block_size=8, seed=3)
    q = onnxsim.apply_double_quantization_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_double_quantization_declines_dynamic_scale_input():
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
    q = onnxsim.apply_double_quantization_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_double_quantization_noop_without_dequantize_linear():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_double_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_double_quantization_matches_python_reference_output():
    model = _blockwise_int4_model(k=512, n=8, block_size=8, seed=4)

    q_py = onnxsim.apply_double_quantization(model, min_elements=64)
    q_cpp = onnxsim.apply_double_quantization_cpp(model)
    onnx.checker.check_model(q_py)
    onnx.checker.check_model(q_cpp)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    (y_py,) = _run(q_py, {"X": x})
    (y_cpp,) = _run(q_cpp, {"X": x})
    assert np.allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)
