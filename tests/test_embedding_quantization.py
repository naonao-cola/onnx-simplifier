"""Tests for ``onnxsim.quantize_embedding_binary``/``quantize_embedding_int8``
-- see ``onnxsim/embedding_quantization.py`` for the technique (compressing
a retrieval encoder's own output embedding, asymmetrically between a
higher-precision query side and an aggressively compressed document side).
"""

import numpy as np
import onnx
import onnx.numpy_helper
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


def _embed_model(batch=2, embed_dim=16, opset=13, output_name="embedding"):
    return _model(
        f"""
        g (float[{batch},{embed_dim}] X) => (float[{batch},{embed_dim}] {output_name})
        {{
          {output_name} = Identity(X)
        }}
        """,
        opset=opset,
    )


def _ambiguous_model():
    # Two independent Identity outputs from the same input -- neither is
    # unambiguously "the embedding output" without an explicit output_name.
    return _model(
        """
        g (float[2,16] X) => (float[2,16] a, float[2,16] b)
        {
          a = Identity(X)
          b = Identity(X)
        }
        """
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


# ---------------------------------------------------------------- binary ---


def test_binary_output_matches_numpy_packbits_reference():
    batch, embed_dim = 3, 16
    model = _embed_model(batch=batch, embed_dim=embed_dim)
    q = onnxsim.quantize_embedding_binary(model)
    onnx.checker.check_model(q)

    out_vi = q.graph.output[0]
    assert out_vi.type.tensor_type.elem_type == onnx.TensorProto.UINT8
    dims = [d.dim_value for d in out_vi.type.tensor_type.shape.dim]
    assert dims == [batch, embed_dim // 8]

    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch, embed_dim)).astype(np.float32)
    (packed,) = _run(q, {"X": x})
    expected = np.packbits(x > 0, axis=-1, bitorder="big")
    assert packed.dtype == np.uint8
    assert np.array_equal(packed, expected)


def test_binary_declines_when_embed_dim_not_multiple_of_8():
    model = _embed_model(embed_dim=12)
    q = onnxsim.quantize_embedding_binary(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_binary_declines_below_opset13():
    model = _embed_model(opset=12)
    q = onnxsim.quantize_embedding_binary(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_binary_declines_when_output_ambiguous():
    model = _ambiguous_model()
    q = onnxsim.quantize_embedding_binary(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_binary_honors_explicit_output_name():
    model = _ambiguous_model()
    q = onnxsim.quantize_embedding_binary(model, output_name="b")
    onnx.checker.check_model(q)
    assert q.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert q.graph.output[1].type.tensor_type.elem_type == onnx.TensorProto.UINT8


# ------------------------------------------------------------------ int8 ---


def test_int8_output_dtype_and_shape():
    batch, embed_dim = 3, 16
    model = _embed_model(batch=batch, embed_dim=embed_dim)
    q = onnxsim.quantize_embedding_int8(model, num_samples=8, seed=0)
    onnx.checker.check_model(q)

    out_vi = q.graph.output[0]
    assert out_vi.type.tensor_type.elem_type == onnx.TensorProto.INT8
    dims = [d.dim_value for d in out_vi.type.tensor_type.shape.dim]
    assert dims == [batch, embed_dim]
    op_types = [n.op_type for n in q.graph.node]
    assert "QuantizeLinear" in op_types


def test_int8_dequantized_values_close_to_float():
    batch, embed_dim = 4, 32
    model = _embed_model(batch=batch, embed_dim=embed_dim)
    q = onnxsim.quantize_embedding_int8(model, num_samples=32, seed=1)

    quant_node = next(n for n in q.graph.node if n.op_type == "QuantizeLinear")
    scale = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == quant_node.input[1])
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((batch, embed_dim)).astype(np.float32)
    (packed,) = _run(q, {"X": x})
    assert packed.dtype == np.int8

    dequantized = packed.astype(np.float64) * float(scale)
    rel_err = np.linalg.norm(dequantized - x) / max(np.linalg.norm(x), 1e-6)
    assert rel_err < 0.05


def test_int8_declines_below_opset13():
    model = _embed_model(opset=12)
    q = onnxsim.quantize_embedding_int8(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_int8_declines_when_output_ambiguous():
    model = _ambiguous_model()
    q = onnxsim.quantize_embedding_int8(model)
    assert q.SerializeToString() == model.SerializeToString()


# ------------------------------------------------------------- asymmetric --


def test_asymmetric_query_int8_document_binary_from_same_export():
    # The blog's own workflow: the same encoder model, quantized two
    # different ways depending on which side of a retrieval system it
    # serves -- query side keeps more precision, document side compresses
    # harder. Both should still run and agree closely with the float model.
    batch, embed_dim = 2, 16
    model = _embed_model(batch=batch, embed_dim=embed_dim)
    query_model = onnxsim.quantize_embedding_int8(model, num_samples=8, seed=0)
    doc_model = onnxsim.quantize_embedding_binary(model)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((batch, embed_dim)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (query_y,) = _run(query_model, {"X": x})
    (doc_y,) = _run(doc_model, {"X": x})

    assert query_y.dtype == np.int8
    assert doc_y.dtype == np.uint8
    assert doc_y.shape == (batch, embed_dim // 8)
    assert np.array_equal(doc_y, np.packbits(float_y > 0, axis=-1, bitorder="big"))
