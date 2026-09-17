"""Tests for ``onnxsim.quantize_weight_only_kmeans`` -- see
``onnxsim/kmeans_quantization.py`` for the technique (a per-layer,
k-means-fitted weight codebook -- unlike NF4/MXFP4's fixed, data-independent
codebooks). This function now delegates to the verified C++ port
(:func:`onnxsim.apply_kmeans_quantization_cpp`) for its own default
parameters -- see ``tests/test_kmeans_quantization_cpp.py`` for the actual
technique-level tests (structural properties, reconstruction quality,
Gemm/bias handling, no-op cases). The tests here only cover this module's
own thin wrapper: the default-args delegation and the ValueError raised for
every parameter the C++ port can't honor.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=32, N=8, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return parser.parse_model(
        f"""
        <
          ir_version: 8,
          opset_import: ["": 13]
        >
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )


def _model_with_weight(K=32, N=8, weight=None, seed=0):
    model = _matmul_model(K, N, weight, seed)
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model.graph.initializer.extend([_f32(weight, "W")])
    return model


def test_quantize_weight_only_kmeans_matches_cpp_port_for_defaults():
    model = _model_with_weight(K=32, N=8, seed=0)
    py_q = onnxsim.quantize_weight_only_kmeans(model)
    cpp_q = onnxsim.apply_kmeans_quantization_cpp(model)
    assert py_q.SerializeToString() == cpp_q.SerializeToString()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 5},
        {"iters": 10},
        {"seed": 1},
        {"skip_names": {"W"}},
    ],
)
def test_quantize_weight_only_kmeans_raises_on_non_default_params(kwargs):
    model = _model_with_weight(K=32, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_kmeans(model, **kwargs)
