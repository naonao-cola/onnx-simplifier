"""Tests for ``onnxsim.quantize_weight_only_lo_bcq`` -- see
``onnxsim/lo_bcq.py`` for the technique (per-block-cluster, Lloyd-max-fitted
weight codebooks). This function now delegates to the verified C++ port
(:func:`onnxsim.apply_lo_bcq_cpp`) for its own default parameters -- see
``tests/test_lo_bcq_cpp.py`` for the actual technique-level tests
(structural properties, reconstruction quality, Gemm/bias handling, no-op
cases). The tests here only cover this module's own thin wrapper: the
default-args delegation and the ValueError raised for every parameter the
C++ port can't honor.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=64, N=8, weight=None, seed=0):
    model = parser.parse_model(
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
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model.graph.initializer.extend([_f32(weight, "W")])
    return model


def test_quantize_weight_only_lo_bcq_matches_cpp_port_for_defaults():
    model = _matmul_model(K=64, N=8, seed=0)
    py_q = onnxsim.quantize_weight_only_lo_bcq(model)
    cpp_q = onnxsim.apply_lo_bcq_cpp(model)
    assert py_q.SerializeToString() == cpp_q.SerializeToString()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 5},
        {"block_size": 64},
        {"num_clusters": 3},
        {"outer_iters": 5},
        {"seed": 1},
        {"skip_names": {"W"}},
    ],
)
def test_quantize_weight_only_lo_bcq_raises_on_non_default_params(kwargs):
    model = _matmul_model(K=64, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_lo_bcq(model, **kwargs)
