"""Tests for ``onnxsim.quantize_weight_only_mxfp4`` (OCP Microscaling
MXFP4, see ``onnxsim/mx_quantization.py``).

``quantize_weight_only_mxfp4`` now delegates to the verified C++ port
(:func:`onnxsim.quantize_weight_only_mxfp4_cpp`) for its default
``block_size``/``skip_names`` -- see ``tests/test_mx_quantization_cpp.py``
for the deep, structural/numeric verification of that C++ algorithm
(block-wise quantization onto E2M1's own fixed 16-value codebook with a
per-block power-of-two scale, represented via ordinary
Gather/Reshape/Mul). A non-default ``block_size`` or a real ``skip_names``
value falls back to this module's own original pure-Python implementation,
exercised here directly since the C++ port doesn't generalize to it.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.mx_quantization import MXFP4_CODEBOOK

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


def test_quantize_weight_only_mxfp4_delegates_to_cpp_port():
    model = _matmul_model(K=64, N=16, seed=0)
    py_result = onnxsim.quantize_weight_only_mxfp4(model)
    cpp_result = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    assert py_result.SerializeToString() == cpp_result.SerializeToString()


def test_quantize_weight_only_mxfp4_default_args_produce_quantized_model():
    # The delegated C++ pass rewires the matched node's weight input to a
    # freshly created initializer, leaving the original "W" dangling
    # unused -- so the node's own current input name, not initializer list
    # membership, is the only reliable way to find the actual
    # post-quantization weight.
    model = _matmul_model(K=64, N=16, seed=1)
    q = onnxsim.quantize_weight_only_mxfp4(model)
    onnx.checker.check_model(q)
    node = next(n for n in q.graph.node if n.op_type in ("MatMul", "Gemm"))
    assert node.input[1] != "W"


def test_quantize_weight_only_mxfp4_nondefault_block_size_uses_python_fallback():
    model = _matmul_model(K=64, N=16, seed=2)
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=16)
    onnx.checker.check_model(q)
    op_types = {n.op_type for n in q.graph.node}
    # The pure-Python fallback's own Gather/Reshape/Mul graph shape, unlike
    # the C++ port's folded initializer.
    assert "Gather" in op_types
    ws = next(t for t in q.graph.initializer if t.name == "W_mxfp4_scale")
    assert ws.dims[0] == 64 // 16


def test_quantize_weight_only_mxfp4_skip_names_uses_python_fallback():
    rng = np.random.default_rng(8)
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
    q = onnxsim.quantize_weight_only_mxfp4(model, skip_names=["W_other"])
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_mxfp4_q" in names
    assert "W_other_mxfp4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_mxfp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_mxfp4(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_mxfp4_codebook_is_well_formed():
    codebook = np.asarray(MXFP4_CODEBOOK)
    assert codebook.shape == (16,)
    assert np.all(np.diff(codebook) >= 0)  # non-decreasing (E2M1 has two zeros)
    assert codebook[0] == -6.0 and codebook[-1] == 6.0
    assert list(codebook).count(0.0) == 2  # +0.0 and -0.0, distinct bit patterns
