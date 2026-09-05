"""Tests for ``onnxsim.apply_intactkv`` -- see ``onnxsim/intactkv.py`` for
the technique (splitting a decoder's ``Concat(past, new, axis=seq)``
KV-cache stream into an exact, fixed-size pivot-token prefix and an
ordinary, still-quantizable "rest" stream).
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


def _kv_cache_model(batch=1, heads=2, head_dim=4, opset=13, symbolic_seq=True):
    # Mirrors tests/test_kv_cache_quantization.py's own _kv_cache_model:
    # new_key is routed through an Identity so the matcher can tell "the
    # persistent cache" apart from "this step's fresh token".
    seq_past = "seq_past" if symbolic_seq else 3
    seq_present = "seq_present" if symbolic_seq else 4
    return _model(
        f"""
        g (float[{batch},{heads},{seq_past},{head_dim}] past_key,
           float[{batch},{heads},1,{head_dim}] new_key_raw)
          => (float[{batch},{heads},{seq_present},{head_dim}] present_key,
              float summary)
        {{
          new_key = Identity(new_key_raw)
          present_key = Concat<axis = 2>(past_key, new_key)
          summary = ReduceSum<keepdims = 0>(present_key)
        }}
        """,
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


def test_intactkv_splits_stream_into_pivot_and_rest():
    model = _kv_cache_model()
    m = onnxsim.apply_intactkv(model, num_pivot_tokens=4)
    onnx.checker.check_model(m)

    input_names = {i.name for i in m.graph.input}
    output_names = {o.name for o in m.graph.output}

    assert "past_key_pivot" in input_names
    assert "past_key_rest" in input_names
    assert "past_key" not in input_names  # renamed, not duplicated

    assert "present_key_pivot" in output_names
    assert "present_key_rest" in output_names
    assert "present_key" in output_names  # original binding preserved

    pivot_in = next(i for i in m.graph.input if i.name == "past_key_pivot")
    assert pivot_in.type.tensor_type.shape.dim[2].dim_value == 4

    op_types = [n.op_type for n in m.graph.node]
    # One Identity is the toy model's own new_key = Identity(new_key_raw);
    # the other is apply_intactkv's own pivot passthrough.
    assert op_types.count("Identity") == 2
    assert op_types.count("Concat") == 2  # rest stream + reconstruction

    reconstruct = next(
        n
        for n in m.graph.node
        if n.op_type == "Concat" and n.output[0] == "present_key"
    )
    assert list(reconstruct.input) == ["present_key_pivot", "present_key_rest"]

    # The original attention-math consumer (ReduceSum) still reads
    # present_key by name -- no rewiring needed, since the reconstruction
    # node reuses that exact output binding.
    reduce_node = next(n for n in m.graph.node if n.op_type == "ReduceSum")
    assert reduce_node.input[0] == "present_key"


def test_intactkv_noop_without_kv_cache_pattern():
    model = _model(
        """
        g (float[4,4] x) => (float[4,4] y)
        {
          y = Relu(x)
        }
        """
    )
    result = onnxsim.apply_intactkv(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_intactkv_declines_when_past_has_other_consumers():
    model = _model(
        """
        g (float[1,2,3,4] past_key, float[1,2,1,4] new_key_raw)
          => (float[1,2,4,4] present_key, float summary, float past_summary)
        {
          new_key = Identity(new_key_raw)
          present_key = Concat<axis = 2>(past_key, new_key)
          summary = ReduceSum<keepdims = 0>(present_key)
          past_summary = ReduceSum<keepdims = 0>(past_key)
        }
        """
    )
    result = onnxsim.apply_intactkv(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_intactkv_pivot_passthrough_is_bit_exact():
    # Standalone (no downstream quantizer): the pivot stream must round-trip
    # through Identity bit-for-bit -- the whole point of the technique.
    batch, heads, head_dim = 1, 2, 4
    model = _kv_cache_model(batch=batch, heads=heads, head_dim=head_dim)
    m = onnxsim.apply_intactkv(model, num_pivot_tokens=4)
    onnx.checker.check_model(m)

    rng = np.random.default_rng(0)
    pivot = rng.standard_normal((batch, heads, 4, head_dim)).astype(np.float32)
    rest = np.zeros((batch, heads, 0, head_dim), dtype=np.float32)
    new0 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)

    outputs = _run(
        m,
        {
            "past_key_pivot": pivot,
            "past_key_rest": rest,
            "new_key_raw": new0,
        },
    )
    # graph.output order: [present_key, summary, present_key_rest, present_key_pivot]
    present_key, _summary, present_rest, present_pivot = outputs
    np.testing.assert_array_equal(present_pivot, pivot)
    np.testing.assert_array_equal(present_key[:, :, :4, :], pivot)
    np.testing.assert_array_equal(present_rest, new0)


def test_intactkv_composed_with_quantize_kv_cache_pivots_exact_rest_lossy():
    # The end-to-end property that is the entire point of IntactKV: after
    # composing apply_intactkv with quantize_kv_cache, the pivot positions
    # of the reconstructed cache reconstruct EXACTLY (they never touch a
    # quantizer at all), while the non-pivot positions go through ordinary
    # lossy INT8 quantization.
    batch, heads, head_dim = 1, 2, 8
    num_pivot = 4
    model = _kv_cache_model(batch=batch, heads=heads, head_dim=head_dim)
    split = onnxsim.apply_intactkv(model, num_pivot_tokens=num_pivot)
    q = onnxsim.quantize_kv_cache(split, num_samples=32, seed=2)
    onnx.checker.check_model(q)

    # present_key_rest must now be INT8 -- quantize_kv_cache matched and
    # quantized exactly the stream apply_intactkv carved out for it.
    rest_vi = next(o for o in q.graph.output if o.name == "present_key_rest")
    assert rest_vi.type.tensor_type.elem_type == onnx.TensorProto.INT8
    pivot_vi = next(o for o in q.graph.output if o.name == "present_key_pivot")
    assert pivot_vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    # Probe the dequantized rest tensor directly so we can compare it
    # against float numpy math with a tight tolerance, rather than trusting
    # an end-to-end onnxruntime run's own reduction order.
    dequant_node = next(n for n in q.graph.node if n.op_type == "DequantizeLinear")
    q_probe = onnx.ModelProto()
    q_probe.CopyFrom(q)
    q_probe.graph.output.append(onnx.ValueInfoProto(name=dequant_node.output[0]))

    rng = np.random.default_rng(4)
    pivot = rng.standard_normal((batch, heads, num_pivot, head_dim)).astype(np.float32)
    empty_rest = np.zeros((batch, heads, 0, head_dim), dtype=np.int8)
    new0 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)
    new1 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)

    # q_probe.graph.output order:
    # [present_key, summary, present_key_rest, present_key_pivot, dequant]
    outputs0 = _run(
        q_probe,
        {
            "past_key_pivot": pivot,
            "past_key_rest": empty_rest,
            "new_key_raw": new0,
        },
    )
    present_key0, _summary0, present_rest0, present_pivot0, rest_f0 = outputs0
    assert present_rest0.dtype == np.int8

    outputs1 = _run(
        q_probe,
        {
            "past_key_pivot": present_pivot0,
            "past_key_rest": present_rest0,
            "new_key_raw": new1,
        },
    )
    present_key1, _summary1, present_rest1, present_pivot1, rest_f1 = outputs1

    # Pivot half: bit-for-bit exact, both steps -- never quantized, never
    # revised.
    np.testing.assert_array_equal(present_pivot0, pivot)
    np.testing.assert_array_equal(present_pivot1, pivot)
    np.testing.assert_array_equal(present_key1[:, :, :num_pivot, :], pivot)

    # Rest half (post-dequantization): close to, but not exactly, the
    # float reference -- ordinary lossy INT8 quantization error, checked
    # with a loose relative tolerance (same generous bound
    # tests/test_kv_cache_quantization.py's own round-trip test uses).
    float_rest0, _ = _run(
        model,
        {
            "past_key": np.zeros((batch, heads, 0, head_dim), np.float32),
            "new_key_raw": new0,
        },
    )
    float_rest1, _ = _run(model, {"past_key": float_rest0, "new_key_raw": new1})
    assert _rel_l2(float_rest1, rest_f1) < 0.2
    assert not np.array_equal(rest_f1, float_rest1)  # genuinely lossy

    # And the reconstructed present_key really is pivot-exact ++ rest-lossy
    # concatenated, matching the reconstruction Concat's own inputs.
    np.testing.assert_array_equal(present_key1[:, :, :num_pivot, :], present_pivot1)
    np.testing.assert_array_equal(present_key1[:, :, num_pivot:, :], rest_f1)
