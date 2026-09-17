"""Tests for ``onnxsim.apply_pruning_finetune_cpp`` -- the C++-backed
port of ``onnxsim.apply_pruning_finetune`` (closed-form ridge-regression
recovery for a pruned MatMul/vanilla-Gemm layer, see
``onnxsim/finetune_entry.h``). Like ``test_gptq_cpp.py``, this runs the
*original* (pre-pruning) model over real calibration data through a real
``onnxruntime``-backed executor -- never a fake/mock executor -- and
checks agreement against the pure-Python reference: both sides recover
the same channel correspondence and solve the identical dense linear
system, so any divergence beyond ordinary floating-point rounding is a
bug, not an accepted tolerance.

Pruning itself is simulated directly here (slicing a weight's own rows or
columns and adjusting the matching graph input/output shape) rather than
routed through ``onnxsim.apply_structured_pruning_cpp`` -- this keeps
each test's own ``keep`` index set explicit and lets input-channel and
output-channel pruning be exercised independently.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.finetune import apply_pruning_finetune

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=9):
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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _matmul_model(K, N, w, opset=18):
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
        opset=opset,
    )


def _gemm_model(K, N, w, b, opset=18):
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
        opset=opset,
    )


def _prune_input_channels(model, keep_in):
    pruned = onnx.ModelProto()
    pruned.CopyFrom(model)
    w = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W")
    )
    w_pruned = w[keep_in, :]
    for t in pruned.graph.initializer:
        if t.name == "W":
            t.CopyFrom(_f32(w_pruned, "W"))
    pruned.graph.input[0].type.tensor_type.shape.dim[1].ClearField("dim_param")
    pruned.graph.input[0].type.tensor_type.shape.dim[1].dim_value = len(keep_in)
    return pruned


def _prune_output_channels(model, keep_out):
    pruned = onnx.ModelProto()
    pruned.CopyFrom(model)
    w = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W")
    )
    w_pruned = w[:, keep_out]
    for t in pruned.graph.initializer:
        if t.name == "W":
            t.CopyFrom(_f32(w_pruned, "W"))
        elif t.name == "B":
            b = onnx.numpy_helper.to_array(t)
            t.CopyFrom(_f32(b[keep_out], "B"))
    pruned.graph.output[0].type.tensor_type.shape.dim[1].ClearField("dim_param")
    pruned.graph.output[0].type.tensor_type.shape.dim[1].dim_value = len(keep_out)
    return pruned


def _correlated_calibration(K, num_samples=64, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _assert_close_parity(original, pruned, calibration_data, keep_k, **kwargs):
    py = apply_pruning_finetune(original, pruned, calibration_data, **kwargs)
    cpp = onnxsim.apply_pruning_finetune_cpp(
        original, pruned, calibration_data, **kwargs
    )
    onnx.checker.check_model(cpp)

    w_py = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name == "W")
    )
    w_cpp = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "W")
    )
    np.testing.assert_allclose(w_cpp, w_py, rtol=1e-5, atol=1e-6)

    rng = np.random.default_rng(99)
    x = rng.standard_normal((4, keep_k)).astype(np.float32)
    (y_py,) = _run(py, {"X": x})
    (y_cpp,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-6)
    return py, cpp


def test_finetune_cpp_matches_python_input_channel_pruning():
    K, N = 32, 16
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    original = _matmul_model(K, N, w)
    keep_in = np.arange(0, K, 2)
    pruned = _prune_input_channels(original, keep_in)
    cal = _correlated_calibration(K, seed=1)

    py, _cpp = _assert_close_parity(original, pruned, cal, len(keep_in))
    # Input-channel pruning loses information -- the fit should have
    # actually moved away from the naively-sliced weight.
    w_naive = w[keep_in, :]
    w_py = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name == "W")
    )
    assert not np.allclose(w_py, w_naive)


def test_finetune_cpp_matches_python_output_channel_pruning():
    K, N = 32, 16
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    original = _matmul_model(K, N, w)
    keep_out = np.arange(0, N, 2)
    pruned = _prune_output_channels(original, keep_out)
    cal = _correlated_calibration(K, seed=3)

    _assert_close_parity(original, pruned, cal, K)


def test_finetune_cpp_matches_python_with_bias_input_pruning():
    K, N = 24, 12
    rng = np.random.default_rng(4)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    b = (rng.standard_normal(N) * 0.1).astype(np.float32)
    original = _gemm_model(K, N, w, b)
    keep_in = np.arange(0, K, 3)
    pruned = _prune_input_channels(original, keep_in)
    cal = _correlated_calibration(K, seed=5)

    py, cpp = _assert_close_parity(original, pruned, cal, len(keep_in))
    b_py = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name == "B")
    )
    b_cpp = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    np.testing.assert_allclose(b_cpp, b_py, rtol=1e-5, atol=1e-6)


def test_finetune_cpp_matches_python_different_reg_param():
    K, N = 40, 8
    rng = np.random.default_rng(6)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    original = _matmul_model(K, N, w)
    keep_in = np.arange(0, K, 2)
    pruned = _prune_input_channels(original, keep_in)
    cal = _correlated_calibration(K, seed=7)

    _assert_close_parity(original, pruned, cal, len(keep_in), reg_param=1.0)
    _assert_close_parity(original, pruned, cal, len(keep_in), reg_param=1e-4)


def test_finetune_cpp_noop_when_both_axes_pruned():
    # Both output AND input channels pruned at once -- declined outright
    # per this technique's own documented boundary (an ambiguous inverse
    # problem, never guessed at).
    K, N = 32, 16
    rng = np.random.default_rng(8)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    original = _matmul_model(K, N, w)
    keep_in = np.arange(0, K, 2)
    keep_out = np.arange(0, N, 2)
    both = _prune_output_channels(_prune_input_channels(original, keep_in), keep_out)
    cal = _correlated_calibration(K, seed=9)

    result = onnxsim.apply_pruning_finetune_cpp(original, both, cal)
    assert result.SerializeToString() == both.SerializeToString()


def test_finetune_cpp_noop_without_matching_node():
    original = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    pruned = onnx.ModelProto()
    pruned.CopyFrom(original)
    cal = [{"X": np.zeros((2, 4), dtype=np.float32)}]
    result = onnxsim.apply_pruning_finetune_cpp(original, pruned, cal)
    assert result.SerializeToString() == pruned.SerializeToString()


def test_finetune_cpp_missing_calibration_input_raises():
    K, N = 16, 8
    rng = np.random.default_rng(10)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    original = _matmul_model(K, N, w)
    keep_in = np.arange(0, K, 2)
    pruned = _prune_input_channels(original, keep_in)
    bad_cal = [{"wrong_name": np.zeros((2, K), dtype=np.float32)}]
    with pytest.raises(ValueError):
        onnxsim.apply_pruning_finetune_cpp(original, pruned, bad_cal)
