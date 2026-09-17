"""Tests for ``onnxsim.apply_gear`` -- see ``onnxsim/gear.py`` for the
technique (GEAR-style low-rank-plus-sparse residual compensation layered on
top of ``onnxsim.kv_cache_quantization``'s own static, per-channel INT8
quantization of a decoder's ``Concat(past, new, axis=seq)`` KV-cache
stream). ``apply_gear`` now delegates unconditionally to the verified C++
port (:func:`onnxsim.apply_gear_cpp`), which shares its own full parameter
set exactly -- see ``tests/test_gear_cpp.py`` for the actual technique-level
tests (structural properties, reconstruction quality, no-op cases). The
test here only confirms the delegation itself is byte-identical.
"""

import numpy as np
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, opset=13, ir_version=8):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _kv_cache_model(batch=1, heads=2, head_dim=8):
    return _model(
        f"""
        g (float[{batch},{heads},seq_past,{head_dim}] past_key,
           float[{batch},{heads},1,{head_dim}] new_key_raw)
          => (float[{batch},{heads},seq_present,{head_dim}] present_key,
              float summary)
        {{
          new_key = Identity(new_key_raw)
          present_key = Concat<axis = 2>(past_key, new_key)
          summary = ReduceSum<keepdims = 0>(present_key)
        }}
        """
    )


def test_apply_gear_matches_cpp_port():
    model = _kv_cache_model()
    calib = [
        {
            "past_key": np.random.default_rng(0)
            .standard_normal((1, 2, 3, 8))
            .astype(np.float32),
            "new_key_raw": np.random.default_rng(1)
            .standard_normal((1, 2, 1, 8))
            .astype(np.float32),
        }
    ]
    py_q = onnxsim.apply_gear(
        model, calibration_data=calib, rank=2, outlier_fraction=0.25
    )
    cpp_q = onnxsim.apply_gear_cpp(
        model, calibration_data=calib, rank=2, outlier_fraction=0.25
    )
    assert py_q.SerializeToString() == cpp_q.SerializeToString()


def test_apply_gear_noop_without_kv_cache_pattern():
    model = _model(
        """
        g (float[4,4] x) => (float[4,4] y)
        {
          y = Relu(x)
        }
        """
    )
    result = onnxsim.apply_gear(model)
    assert result.SerializeToString() == model.SerializeToString()
