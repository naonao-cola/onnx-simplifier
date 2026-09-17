"""Tests for ``onnxsim.quantize_weight_only_adpq`` -- see ``onnxsim/adpq.py``
for the technique (calibration-free, per-group Adaptive-LASSO-style
soft-threshold salient/non-salient split, non-salient elements quantized
block-wise INT4, salient elements reconstructed exactly via a sparse
``ScatterND`` correction).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_GROUP_SIZE = 128


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


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
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
        [_f32(weight, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_adpq_needs_no_calibration_data():
    # Unlike onnxsim.owq/onnxsim.spqr/onnxsim.gptq/onnxsim.billm, this
    # function takes only the model itself -- no calibration_data,
    # num_samples, seed, or providers argument exists to pass real
    # activations through in the first place.
    import inspect

    params = inspect.signature(onnxsim.quantize_weight_only_adpq).parameters
    assert "calibration_data" not in params
    assert "providers" not in params


# quantize_weight_only_adpq now delegates to the verified C++ port
# (apply_adpq_cpp), which hardcodes group_size=128/lambda_=3.0/gamma=0.3
# and folds the round trip directly into a replacement float32
# initializer instead of building a real
# DequantizeLinear+ScatterND+Add graph rewrite -- see onnxsim/adpq.py's
# own docstring. The detailed algorithmic properties (salient-element
# exact reconstruction, non-salient grid size, opset-independence) are
# already covered end to end against apply_adpq_cpp directly in
# tests/test_adpq_cpp.py; the tests below only exercise the thin
# wrapper itself: parameter validation and basic delegation sanity.


def test_adpq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=_GROUP_SIZE, N=8, seed=0)
    q = onnxsim.quantize_weight_only_adpq(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == (_GROUP_SIZE, 8)
    assert new_w.dtype == np.float32

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, _GROUP_SIZE)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_adpq_salient_positions_reconstruct_exactly():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((_GROUP_SIZE, 8)).astype(np.float32) * 0.1
    weight[0, 0] = 50.0
    model = _matmul_model(K=_GROUP_SIZE, N=8, weight=weight)
    q = onnxsim.quantize_weight_only_adpq(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert abs(float(new_w[0, 0]) - 50.0) < 1e-3


def test_adpq_rejects_non_default_group_size():
    model = _matmul_model(K=_GROUP_SIZE, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_adpq(model, group_size=8)


def test_adpq_rejects_non_default_lambda():
    model = _matmul_model(K=_GROUP_SIZE, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_adpq(model, lambda_=1e6)


def test_adpq_rejects_non_default_gamma():
    model = _matmul_model(K=_GROUP_SIZE, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_adpq(model, gamma=0.0)


def test_adpq_declines_when_k_not_divisible_by_group_size():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 128
    q = onnxsim.quantize_weight_only_adpq(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_adpq_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_adpq(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_adpq_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_adpq(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_adpq_gemm_transb_and_bias():
    rng = np.random.default_rng(14)
    K, N = _GROUP_SIZE, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.3
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.quantize_weight_only_adpq(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3
