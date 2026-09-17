"""Tests for ``onnxsim.apply_smooth_attention_cpp`` -- the C++-backed
port of ``onnxsim.apply_smooth_attention`` (QoQ's SmoothAttention scale
migration, see ``onnxsim/smooth_attention_entry.h``). Like
``test_llm_int8_cpp.py``, this runs the model over real calibration data
through a real ``onnxruntime``-backed executor -- never a fake/mock
executor -- and checks exact (bit-for-bit) parity against the pure-Python
reference: this is a closed-form diagonal rescaling whose only reduction
is a per-channel MAX (order-independent, unlike a sum), so both sides are
expected to agree exactly, not just closely.

Named ``test_smooth_attention_cpp.py`` (not
``test_qoq_smooth_attention_cpp.py``) since ``tests/test_qoq_cpp.py``
already exists for ``quantize_weight_only_qoq`` -- a completely different
function in the same ``onnxsim/qoq.py`` module -- and this name reads
more clearly on its own with no collision.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_smooth_attention_cpp
from onnxsim.qoq import apply_smooth_attention

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=18, ir_version=10):
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


def _attention_model(batch=1, heads=2, seq=4, head_dim=8, opset=18):
    return _model(
        f"""
        g (float[{batch},{heads},{seq},{head_dim}] Q,
           float[{batch},{heads},{head_dim},{seq}] Kt,
           float[{batch},{heads},{seq},{head_dim}] V)
           => (float[{batch},{heads},{seq},{head_dim}] Out)
        {{
          scores = MatMul(Q, Kt)
          scaled = Mul(scores, Scale)
          probs = Softmax(scaled)
          Out = MatMul(probs, V)
        }}
        """,
        initializer=[_f32(np.array(0.125, dtype=np.float32), "Scale")],
        opset=opset,
    )


def _calibration(batch=1, heads=2, seq=4, head_dim=8, num_samples=6, seed=1):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        q = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
        kt = rng.standard_normal((batch, heads, head_dim, seq)).astype(np.float32)
        # Plant a large-magnitude outlier channel in Kt's own head_dim
        # axis (axis -2) -- exactly the case SmoothAttention exists to
        # flatten.
        kt[:, :, 0, :] *= 25.0
        v = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
        batches.append({"Q": q, "Kt": kt, "V": v})
    return batches


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_smooth_attention(model, calibration_data, **kwargs)
    cpp = apply_smooth_attention_cpp(model, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape, a.name
        assert np.array_equal(ta, tb), a.name
    return py, cpp


def test_smooth_attention_cpp_matches_python_exactly():
    _, cpp = _assert_exact_parity(_attention_model(), _calibration())
    ops = sorted(n.op_type for n in cpp.graph.node)
    assert ops.count("Mul") >= 1
    assert ops.count("Div") >= 1
    assert any("smooth_attn" in n.name for n in cpp.graph.node)


def test_smooth_attention_cpp_preserves_attention_output_numerically():
    model = _attention_model()
    cpp = apply_smooth_attention_cpp(model, _calibration())
    onnx.checker.check_model(cpp)

    def _run(m, feeds):
        sess = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)

    rng = np.random.default_rng(42)
    feeds = {
        "Q": rng.standard_normal((1, 2, 4, 8)).astype(np.float32),
        "Kt": rng.standard_normal((1, 2, 8, 4)).astype(np.float32),
        "V": rng.standard_normal((1, 2, 4, 8)).astype(np.float32),
    }
    (float_out,) = _run(model, feeds)
    (smoothed_out,) = _run(cpp, feeds)
    # Provably-lossless migration: outputs should match up to ordinary
    # floating-point rounding, not just approximately.
    np.testing.assert_allclose(float_out, smoothed_out, rtol=1e-4, atol=1e-5)


def test_smooth_attention_cpp_no_match_returns_copy():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """,
        opset=18,
    )
    cpp = apply_smooth_attention_cpp(model, [{"X": np.zeros((2, 4), dtype=np.float32)}])
    assert [n.op_type for n in cpp.graph.node] == ["Relu"]


def test_smooth_attention_cpp_missing_calibration_input_raises():
    model = _attention_model()
    with pytest.raises((RuntimeError, ValueError)):
        apply_smooth_attention_cpp(
            model, [{"Wrong": np.zeros((1, 2, 4, 8), dtype=np.float32)}]
        )


def test_smooth_attention_cpp_epsilon_floors_zero_channel():
    # A calibration set where Kt is identically zero -- every per-channel
    # absmax is 0, so `epsilon` alone determines the scale (s = epsilon).
    model = _attention_model()
    calibration_data = [
        {
            "Q": np.zeros((1, 2, 4, 8), dtype=np.float32),
            "Kt": np.zeros((1, 2, 8, 4), dtype=np.float32),
            "V": np.zeros((1, 2, 4, 8), dtype=np.float32),
        }
    ]
    _assert_exact_parity(model, calibration_data, epsilon=1e-3)
