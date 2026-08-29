"""Tests for ``onnxsim.apply_attention_quantization`` -- see
``onnxsim/attention_quantization.py`` for the technique (INT8 quantization
of the decomposed attention subgraph's own Q/K/V operands and, via a
fixed non-data-dependent scale, its Softmax output).
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=18):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=9
    )


def _attention_model(seq=6, head_dim=8, use_mask=False, opset=18):
    nodes = [
        onnx.helper.make_node("Transpose", ["K"], ["Kt"], perm=[1, 0]),
        onnx.helper.make_node("MatMul", ["Q", "Kt"], ["scores"]),
        onnx.helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
    ]
    softmax_input = "scaled"
    initializer = [_f32(np.array(1.0 / np.sqrt(head_dim)), "scale")]
    if use_mask:
        mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
        nodes.append(onnx.helper.make_node("Add", ["scaled", "mask"], ["masked"]))
        initializer.append(_f32(mask, "mask"))
        softmax_input = "masked"
    nodes.append(onnx.helper.make_node("Softmax", [softmax_input], ["probs"], axis=-1))
    nodes.append(onnx.helper.make_node("MatMul", ["probs", "V"], ["Y"]))
    return _model(
        nodes,
        [
            _vi("Q", [seq, head_dim]),
            _vi("K", [seq, head_dim]),
            _vi("V", [seq, head_dim]),
        ],
        [_vi("Y", [seq, head_dim])],
        initializer,
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


def _feeds(seq=6, head_dim=8, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "Q": rng.standard_normal((seq, head_dim)).astype(np.float32),
        "K": rng.standard_normal((seq, head_dim)).astype(np.float32),
        "V": rng.standard_normal((seq, head_dim)).astype(np.float32),
    }


def test_attention_quantization_output_stays_close_to_float_via_onnxruntime():
    model = _attention_model()
    q = onnxsim.apply_attention_quantization(model)
    onnx.checker.check_model(q)

    feeds = _feeds()
    (float_y,) = _run(model, feeds)
    (q_y,) = _run(q, feeds)
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_attention_quantization_handles_masked_variant():
    model = _attention_model(use_mask=True)
    q = onnxsim.apply_attention_quantization(model)
    onnx.checker.check_model(q)

    feeds = _feeds()
    (float_y,) = _run(model, feeds)
    (q_y,) = _run(q, feeds)
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_attention_quantization_leaves_score_matmul_and_softmax_untouched():
    model = _attention_model()
    q = onnxsim.apply_attention_quantization(model)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("Softmax") == 1
    assert op_types.count("MatMul") == 2  # QK^T and probs@V, both still present
    assert "Round" in op_types  # the new quantize-dequantize machinery
    assert "ReduceMax" in op_types


def test_attention_quantization_probs_use_a_fixed_1_over_255_scale():
    model = _attention_model()
    q = onnxsim.apply_attention_quantization(model)

    scale_init = next(t for t in q.graph.initializer if t.name == "attnq_probs_scale")
    scale = onnx.numpy_helper.to_array(scale_init)
    assert np.isclose(scale, 1.0 / 255.0)


def test_attention_quantization_noop_without_attention_pattern():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_attention_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_attention_quantization_noop_when_softmax_output_not_consumed_by_matmul():
    nodes = [
        onnx.helper.make_node("MatMul", ["Q", "K"], ["scores"]),
        onnx.helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        onnx.helper.make_node("Identity", ["probs"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("Q", [4, 4]), _vi("K", [4, 4])],
        [_vi("Y", [4, 4])],
        [],
    )
    result = onnxsim.apply_attention_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_attention_quantization_declines_below_opset18():
    model = _attention_model(opset=13)
    result = onnxsim.apply_attention_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
