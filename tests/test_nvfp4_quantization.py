"""Tests for ``onnxsim.quantize_weight_only_nvfp4`` (NVIDIA NVFP4, see
``onnxsim/nvfp4_quantization.py``).

``quantize_weight_only_nvfp4`` now delegates to the verified C++ port
(:func:`onnxsim.quantize_weight_only_nvfp4_cpp`) -- see
``tests/test_nvfp4_quantization_cpp.py`` for the deep, structural/numeric
verification of the actual quantization algorithm. This file checks that
the public, backward-compatible entry point delegates correctly, that
knobs the C++ port doesn't support raise rather than silently misbehave,
and that this module's own still-used E4M3 grid utilities (format
documentation, not algorithm internals) remain correct.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.nvfp4_quantization import (
    _E4M3_POSITIVE_GRID,
    FLOAT8_E4M3_MAX,
    _round_to_e4m3,
)

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


def test_quantize_weight_only_nvfp4_delegates_to_cpp_port():
    model = _matmul_model(K=64, N=16, seed=0)
    py_result = onnxsim.quantize_weight_only_nvfp4(model)
    cpp_result = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    assert py_result.SerializeToString() == cpp_result.SerializeToString()


def test_quantize_weight_only_nvfp4_default_args_produce_quantized_model():
    # Unlike nf4/if4 (which fold to a single replacement initializer), the
    # delegated C++ pass keeps the same Gather/Reshape/Mul dequantization
    # graph shape the pure-Python reference used to build directly (see
    # quantize_weight_only_nvfp4_cpp's own docstring) -- so the matched
    # node's own weight input becomes a *node* output, not a new
    # initializer.
    model = _matmul_model(K=64, N=16, seed=1)
    q = onnxsim.quantize_weight_only_nvfp4(model)
    onnx.checker.check_model(q)
    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert "Gather" in op_types


def test_quantize_weight_only_nvfp4_rejects_nondefault_block_size():
    model = _matmul_model(K=64, N=16, seed=2)
    with pytest.raises(NotImplementedError):
        onnxsim.quantize_weight_only_nvfp4(model, block_size=8)


def test_quantize_weight_only_nvfp4_rejects_skip_names():
    model = _matmul_model(K=64, N=16, seed=3)
    with pytest.raises(NotImplementedError):
        onnxsim.quantize_weight_only_nvfp4(model, skip_names=["W"])


def test_e4m3_grid_is_well_formed():
    assert _E4M3_POSITIVE_GRID.shape == (127,)
    assert np.all(np.diff(_E4M3_POSITIVE_GRID) > 0)  # strictly increasing, no dupes
    assert _E4M3_POSITIVE_GRID[0] == 0.0
    assert _E4M3_POSITIVE_GRID[-1] == FLOAT8_E4M3_MAX == 448.0
    assert 1.0 in _E4M3_POSITIVE_GRID  # exactly representable (exponent field 7)


def test_round_to_e4m3_is_idempotent_and_clamps():
    rng = np.random.default_rng(10)
    values = rng.uniform(0, 1000, size=1000)
    rounded = _round_to_e4m3(values)
    assert np.all(rounded <= FLOAT8_E4M3_MAX)
    # Rounding an already-representable value must be a no-op.
    assert np.allclose(_round_to_e4m3(rounded), rounded)
