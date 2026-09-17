"""Tests for ``onnxsim.quantize_weight_only_if4`` (IF4 / Adaptive
Block-Scaled Data Types, see ``onnxsim/if4_quantization.py``).

``quantize_weight_only_if4`` now delegates to the verified C++ port
(:func:`onnxsim.quantize_weight_only_if4_cpp`) -- see
``tests/test_if4_quantization_cpp.py`` for the deep, structural/numeric
verification of the actual quantization algorithm (per-block INT4-vs-FP4
selection, dequantization correctness, etc.). This file only checks that
the public, backward-compatible entry point delegates correctly and that
knobs the C++ port doesn't support raise rather than silently misbehave.
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


def test_quantize_weight_only_if4_delegates_to_cpp_port():
    model = _matmul_model(K=64, N=16, seed=0)
    py_result = onnxsim.quantize_weight_only_if4(model)
    cpp_result = onnxsim.quantize_weight_only_if4_cpp(model)
    assert py_result.SerializeToString() == cpp_result.SerializeToString()


def test_quantize_weight_only_if4_default_args_produce_quantized_model():
    # The delegated C++ pass rewires the matched node's weight input to a
    # freshly created initializer, leaving the original "W" dangling
    # unused -- so the node's own current input name, not initializer list
    # membership, is the only reliable way to find the actual
    # post-quantization weight.
    model = _matmul_model(K=64, N=16, seed=1)
    q = onnxsim.quantize_weight_only_if4(model)
    onnx.checker.check_model(q)
    node = next(n for n in q.graph.node if n.op_type in ("MatMul", "Gemm"))
    assert node.input[1] != "W"
    orig_w = next(t for t in model.graph.initializer if t.name == "W")
    new_w = next(t for t in q.graph.initializer if t.name == node.input[1])
    assert not np.array_equal(
        onnx.numpy_helper.to_array(new_w), onnx.numpy_helper.to_array(orig_w)
    )


def test_quantize_weight_only_if4_rejects_nondefault_block_size():
    model = _matmul_model(K=64, N=16, seed=2)
    with pytest.raises(NotImplementedError):
        onnxsim.quantize_weight_only_if4(model, block_size=8)


def test_quantize_weight_only_if4_rejects_skip_names():
    model = _matmul_model(K=64, N=16, seed=3)
    with pytest.raises(NotImplementedError):
        onnxsim.quantize_weight_only_if4(model, skip_names=["W"])
