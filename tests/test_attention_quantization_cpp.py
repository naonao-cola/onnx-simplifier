"""Tests for ``onnxsim.apply_attention_quantization_cpp`` -- the C++-backed
port of ``onnxsim.apply_attention_quantization`` (attention computation
quantization, see ``onnxsim/passes/attention_quantization.h``). Unlike
this repo's weight-only ``*_cpp`` ports, this is not a fold-to-initializer
pass -- it replaces the decomposed attention subgraph's own Q/K/V operands
and Softmax output with new INT8-range quantize/dequantize round-trip
nodes, leaving the score MatMul and the Softmax normalization itself
running in float. This port has no RNG/numerical divergence from the pure
Python reference (a closed-form, deterministic elementwise round trip), so
``test_cpp_matches_python_port_output`` below checks it stays numerically
close, not just structurally similar.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _attention_model(seq=6, head_dim=8, use_mask=False, use_scale=True, opset=18):
    """Mirrors ``tests/test_attention_quantization.py``'s own
    ``_attention_model`` exactly: ``Kt = Transpose(K); scores =
    MatMul(Q, Kt); scaled = Mul(scores, scale) [optional]; masked =
    Add(scaled, mask) [optional]; probs = Softmax(..., axis=-1); Y =
    MatMul(probs, V)``.
    """
    softmax_input = "scores"
    chain = "scores = MatMul(Q, Kt)\n"
    initializer = []
    if use_scale:
        chain += "scaled = Mul(scores, scale)\n"
        softmax_input = "scaled"
        initializer.append(_f32(1.0 / np.sqrt(head_dim), "scale"))
    if use_mask:
        chain += "masked = Add(scaled, mask)\n"
        softmax_input = "masked"
        mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
        initializer.append(_f32(mask, "mask"))

    model = _model(
        f"""
        g (float[{seq},{head_dim}] Q, float[{seq},{head_dim}] K, float[{seq},{head_dim}] V) => (float[{seq},{head_dim}] Y)
        {{
          Kt = Transpose<perm = [1, 0]>(K)
          {chain}
          probs = Softmax<axis = -1>({softmax_input})
          Y = MatMul(probs, V)
        }}
        """,
        initializer,
        opset=opset,
    )
    return model


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _feeds(seq=6, head_dim=8, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "Q": rng.standard_normal((seq, head_dim)).astype(np.float32),
        "K": rng.standard_normal((seq, head_dim)).astype(np.float32),
        "V": rng.standard_normal((seq, head_dim)).astype(np.float32),
    }


def _has_scalar_initializer_close_to(model, value, atol=1e-6):
    for t in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(t)
        if arr.size == 1 and np.isclose(float(arr.reshape(-1)[0]), value, atol=atol):
            return True
    return False


def test_cpp_quantizes_full_attention_subgraph_and_stays_close_to_float():
    model = _attention_model()
    q = onnxsim.apply_attention_quantization_cpp(model)
    onnx.checker.check_model(q)

    feeds = _feeds()
    float_y = _run(model, feeds)
    q_y = _run(q, feeds)
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_handles_masked_variant():
    model = _attention_model(use_mask=True)
    q = onnxsim.apply_attention_quantization_cpp(model)
    onnx.checker.check_model(q)

    feeds = _feeds()
    float_y = _run(model, feeds)
    q_y = _run(q, feeds)
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_handles_no_scale_no_mask_variant():
    # The optional Mul/Add hops are both absent -- Softmax reads straight
    # from the raw QK^T MatMul's own output, the 0-hop case.
    model = _attention_model(use_scale=False, use_mask=False)
    q = onnxsim.apply_attention_quantization_cpp(model)
    onnx.checker.check_model(q)

    feeds = _feeds()
    float_y = _run(model, feeds)
    q_y = _run(q, feeds)
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_leaves_score_matmul_and_softmax_untouched_and_adds_quantize_ops():
    model = _attention_model()
    q = onnxsim.apply_attention_quantization_cpp(model)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("Softmax") == 1
    assert op_types.count("MatMul") == 2  # QK^T and probs@V, both still present
    assert "Round" in op_types  # the new quantize-dequantize machinery
    assert "ReduceMax" in op_types


def test_cpp_probs_use_a_fixed_1_over_255_scale():
    model = _attention_model()
    q = onnxsim.apply_attention_quantization_cpp(model)
    # This port auto-names its own constants (unlike the pure-Python port's
    # "attnq_probs_scale"), so check by value instead of by initializer
    # name.
    assert _has_scalar_initializer_close_to(q, 1.0 / 255.0)
    assert _has_scalar_initializer_close_to(q, 127.0)


def test_cpp_noop_without_attention_pattern():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_attention_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_softmax_output_not_consumed_by_matmul():
    model = _model(
        """
        g (float[4,4] Q, float[4,4] K) => (float[4,4] Y)
        {
          scores = MatMul(Q, K)
          probs = Softmax<axis = -1>(scores)
          Y = Identity(probs)
        }
        """
    )
    result = onnxsim.apply_attention_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_matmul_feeds_softmax():
    # Softmax's own input traces back (through 0 hops, since it isn't a
    # Mul/Div/Add at all) to a bare Relu, not a MatMul -- no candidate.
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          scores = Relu(X)
          probs = Softmax<axis = -1>(scores)
          Y = MatMul(probs, X)
        }
        """
    )
    result = onnxsim.apply_attention_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_below_opset18():
    model = _attention_model(opset=13)
    result = onnxsim.apply_attention_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_matches_python_port_output():
    # No RNG involved on either side (see passes/attention_quantization.h's
    # own "ACCEPTED, PERMANENT DIVERGENCE: none" note) -- both sides build
    # the same closed-form op sequence, so their outputs are expected to
    # agree closely, checked via onnxruntime rather than a direct
    # initializer/node diff since the two sides name/order their new
    # constants and intermediates differently.
    model = _attention_model(seq=5, head_dim=4, use_mask=True)
    py_q = onnxsim.apply_attention_quantization(model)
    cpp_q = onnxsim.apply_attention_quantization_cpp(model)

    feeds = _feeds(seq=5, head_dim=4, seed=7)
    py_y = _run(py_q, feeds)
    cpp_y = _run(cpp_q, feeds)
    np.testing.assert_allclose(py_y, cpp_y, rtol=1e-4, atol=1e-5)
