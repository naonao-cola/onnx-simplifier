"""Tests for ``onnxsim.quantize_weight_only_nf4`` (bitsandbytes' NF4, see
``onnxsim/nf4.py``).

``quantize_weight_only_nf4`` now delegates to the verified C++ port
(:func:`onnxsim.quantize_weight_only_nf4_cpp`) for its default
``block_size``/``skip_names`` -- see ``tests/test_nf4_cpp.py`` for the
deep, structural/numeric verification of that C++ algorithm. A non-default
``block_size`` or a real ``skip_names`` value (needed by, e.g.,
:func:`onnxsim.lora.apply_qlora`) falls back to this module's own original
pure-Python implementation, exercised here directly since the C++ port
doesn't generalize to it.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.nf4 import NF4_CODEBOOK

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


def test_quantize_weight_only_nf4_delegates_to_cpp_port():
    model = _matmul_model(K=64, N=16, seed=0)
    py_result = onnxsim.quantize_weight_only_nf4(model)
    cpp_result = onnxsim.quantize_weight_only_nf4_cpp(model)
    assert py_result.SerializeToString() == cpp_result.SerializeToString()


def test_quantize_weight_only_nf4_default_args_produce_quantized_model():
    # The delegated C++ pass rewires the matched node's weight input to a
    # freshly created initializer, leaving the original "W" dangling
    # unused -- so the node's own current input name, not initializer list
    # membership, is the only reliable way to find the actual
    # post-quantization weight.
    model = _matmul_model(K=64, N=16, seed=1)
    q = onnxsim.quantize_weight_only_nf4(model)
    onnx.checker.check_model(q)
    node = next(n for n in q.graph.node if n.op_type in ("MatMul", "Gemm"))
    assert node.input[1] != "W"
    orig_w = next(t for t in model.graph.initializer if t.name == "W")
    new_w = next(t for t in q.graph.initializer if t.name == node.input[1])
    assert not np.array_equal(
        onnx.numpy_helper.to_array(new_w), onnx.numpy_helper.to_array(orig_w)
    )


def test_quantize_weight_only_nf4_nondefault_block_size_uses_python_fallback():
    model = _matmul_model(K=64, N=16, seed=2)
    q = onnxsim.quantize_weight_only_nf4(model, block_size=32)
    onnx.checker.check_model(q)
    op_types = {n.op_type for n in q.graph.node}
    # The pure-Python fallback's own Gather/Reshape/Mul graph shape, unlike
    # the C++ port's folded initializer.
    assert "Gather" in op_types


def test_quantize_weight_only_nf4_skip_names_uses_python_fallback():
    # Mirrors onnxsim.lora.apply_qlora's own real use: excluding LoRA
    # adapter weights (structurally indistinguishable plain MatMul weights)
    # from quantization.
    rng = np.random.default_rng(4)
    w_base = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((64, 4)).astype(np.float32) * 0.1
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,16] Y, float[batch,4] H)
        {
          Y = MatMul(X, W)
          H = MatMul(X, W_other)
        }
        """,
        initializer=[_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    q = onnxsim.quantize_weight_only_nf4(model, skip_names=["W_other"])
    onnx.checker.check_model(q)
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)  # untouched, byte-for-byte


def test_nf4_codebook_is_well_formed():
    codebook = np.asarray(NF4_CODEBOOK)
    assert codebook.shape == (16,)
    assert np.all(np.diff(codebook) > 0)  # strictly increasing
    assert codebook[0] == -1.0 and codebook[-1] == 1.0
    assert 0.0 in codebook
