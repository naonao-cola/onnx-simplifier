"""Tests for ``onnxsim.quantize_embedding_binary_cpp`` -- the C++-backed
port of ``onnxsim.quantize_embedding_binary`` (embedding-output
binarization, see ``onnxsim/embedding_quantization_entry.h``). Unlike
every ``PredicateBasedPass`` in this repo (which matches a single node
kind), this port targets a whole GRAPH OUTPUT declaration directly -- so
these tests check the resolved output's own new dtype/shape/producer, not
a replacement weight or a matched-node rewrite.

This is a closed-form, deterministic bit-packing scheme with no RNG
anywhere, so ``test_cpp_matches_numpy_packbits_reference`` checks the
packed output against ``numpy.packbits`` directly (the exact operation
``embedding_quantization.py``'s own docstring says this reproduces), not
just a looser structural comparison.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, opset=13, ir_version=9):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _embed_model(embed_dim=16, out_name="Y", opset=13):
    return _model(
        f"""
        g (float[batch,{embed_dim}] X) => (float[batch,{embed_dim}] {out_name})
        {{
          {out_name} = Identity(X)
        }}
        """,
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    return dict(zip(out_names, sess.run(out_names, feeds)))


def test_cpp_matches_numpy_packbits_reference():
    embed_dim = 16
    model = _embed_model(embed_dim=embed_dim)
    q = onnxsim.quantize_embedding_binary_cpp(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, embed_dim)).astype(np.float32)
    # The resolved output's own NAME changes (mirrors quantize_embedding_
    # binary's own `target.name = packed_u8` -- the output binding is
    # retargeted to the new packed tensor's own name, not kept as "Y";
    # confirmed directly against the pure-Python reference), so find it by
    # position (there's only one output) rather than by the stale "Y" name.
    out_name = q.graph.output[0].name
    got = _run(q, {"X": x})[out_name]

    expected = np.packbits(x > 0, axis=-1, bitorder="big")
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, expected)


def test_cpp_output_dtype_and_shape_updated():
    embed_dim = 24
    model = _embed_model(embed_dim=embed_dim)
    q = onnxsim.quantize_embedding_binary_cpp(model)
    onnx.checker.check_model(q)

    out = q.graph.output[0]
    # The output's own name changes away from "Y" (see
    # test_cpp_matches_numpy_packbits_reference's own comment) -- only one
    # output exists, so identify it by position, not by name.
    assert out.name != "Y"
    assert out.type.tensor_type.elem_type == onnx.TensorProto.UINT8
    dims = out.type.tensor_type.shape.dim
    assert len(dims) == 2
    assert not dims[0].HasField("dim_value")  # "batch" stays symbolic
    assert dims[1].dim_value == embed_dim // 8


def test_cpp_preserves_leading_dims_including_symbolic():
    embed_dim = 8
    model = _model(
        f"""
        g (float[batch,seq,{embed_dim}] X) => (float[batch,seq,{embed_dim}] Y)
        {{
          Y = Identity(X)
        }}
        """
    )
    q = onnxsim.quantize_embedding_binary_cpp(model)
    onnx.checker.check_model(q)

    x = np.random.default_rng(1).standard_normal((2, 3, embed_dim)).astype(np.float32)
    out_name = q.graph.output[0].name
    got = _run(q, {"X": x})[out_name]
    assert got.shape == (2, 3, embed_dim // 8)
    np.testing.assert_array_equal(got, np.packbits(x > 0, axis=-1, bitorder="big"))


def test_cpp_explicit_output_name_selects_among_several():
    embed_dim = 8
    model = _model(
        f"""
        g (float[batch,{embed_dim}] X) => (float[batch,{embed_dim}] Y1, float[batch,{embed_dim}] Y2)
        {{
          Y1 = Identity(X)
          Y2 = Identity(X)
        }}
        """
    )
    q = onnxsim.quantize_embedding_binary_cpp(model, output_name="Y2")
    onnx.checker.check_model(q)

    y1 = next(o for o in q.graph.output if o.name == "Y1")
    # Y2 itself gets renamed away (to the new packed tensor's own name,
    # same as the single-output case above) -- it's the OTHER output
    # (Y1, untouched) whose name/dtype stays put, so identify the
    # binarized one by elimination.
    y2 = next(o for o in q.graph.output if o.name != "Y1")
    assert y1.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert y2.type.tensor_type.elem_type == onnx.TensorProto.UINT8


def test_cpp_noop_when_multiple_float_outputs_and_no_name_given():
    embed_dim = 8
    model = _model(
        f"""
        g (float[batch,{embed_dim}] X) => (float[batch,{embed_dim}] Y1, float[batch,{embed_dim}] Y2)
        {{
          Y1 = Identity(X)
          Y2 = Identity(X)
        }}
        """
    )
    result = onnxsim.quantize_embedding_binary_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_output_name_unknown():
    model = _embed_model(embed_dim=8)
    result = onnxsim.quantize_embedding_binary_cpp(model, output_name="NoSuchOutput")
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_last_dim_not_static():
    model = _model(
        """
        g (float[batch,dim] X) => (float[batch,dim] Y)
        {
          Y = Identity(X)
        }
        """
    )
    result = onnxsim.quantize_embedding_binary_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_last_dim_not_multiple_of_8():
    model = _embed_model(embed_dim=12)
    result = onnxsim.quantize_embedding_binary_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset13():
    model = _embed_model(embed_dim=16, opset=12)
    result = onnxsim.quantize_embedding_binary_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
