"""Tests for ``onnxsim.apply_intactkv_cpp`` -- the C++-backed port of
``onnxsim.apply_intactkv`` (IntactKV, see ``onnxsim/passes/intactkv.h``).

Unlike every k-means/rotation-family ``*_cpp`` port in this repo, IntactKV's
own rewrite is a closed-form, deterministic graph restructuring with no
fitting/RNG step at all -- so (per ``passes/intactkv.h``'s own "ACCEPTED,
PERMANENT DIVERGENCE: none" note) these tests hold this port to a *tight*
numeric/structural match against the pure-Python ``apply_intactkv``
reference, unlike the looser "reconstruction-quality" comparisons those
other ports' own tests use.

This port hardcodes ``intactkv.py``'s own default ``num_pivot_tokens=4``
rather than exposing it as a parameter (see ``passes/intactkv.h``'s own
SCOPE NARROWING note), so every test below compares against
``onnxsim.apply_intactkv(model)`` called with its own default too, and no
test below varies ``num_pivot_tokens`` through the ``_cpp`` entry point --
there is no knob to vary.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.intactkv import apply_intactkv

ort = pytest.importorskip("onnxruntime")

_NUM_PIVOT = 4


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


def _kv_model(seq_past=6, seq_new=2, batch=2, head_dim=3, extra_consumer=True):
    """A minimal decoder-step-shaped graph: a single KV-cache stream
    (``present_key``, itself a graph output) that ALSO feeds a further
    consumer (``attn_out``, standing in for the attention math) when
    ``extra_consumer`` is set -- exercising IntactKV's own claim that
    *every* pre-existing consumer of ``present_key``, not just the graph
    output, gets retargeted onto the reconstruction.

    ``past_key``'s own seq dim is a SYMBOLIC name, not ``seq_past`` as a
    literal integer: intactkv.py's own ``_split_pivot_stream`` (matched
    exactly by this port) only clears the declared seq dim on the NEW
    ``present_*_rest`` graph OUTPUT it creates -- the pre-existing
    ``past_*`` graph INPUT (renamed to ``past_*_rest`` in place) keeps
    whatever shape it already declared, unchanged. A literal fixed
    ``seq_past`` there would leave ``past_key_rest`` still declaring the
    ORIGINAL (pre-split) length even though callers are now expected to
    feed it only ``seq_past - num_pivot_tokens`` tokens -- onnxruntime
    enforces a graph input's own declared shape strictly, so tests that
    actually execute the split model (below) need a symbolic ``past_key``
    seq dim to avoid a spurious shape-mismatch error unrelated to this
    port's own correctness (confirmed identical against
    ``onnxsim.apply_intactkv`` itself, which hits the exact same static-shape
    friction with a literal seq_past).
    """
    outputs = "float[{b},seq_total,{h}] present_key".format(b=batch, h=head_dim)
    body_extra = ""
    if extra_consumer:
        outputs += ", float[{b},seq_total,{h}] attn_out".format(b=batch, h=head_dim)
        body_extra = "attn_out = Identity(present_key)"
    return _model(
        f"""
        g (float[{batch},seq_past,{head_dim}] past_key,
           float[{batch},{seq_new},{head_dim}] new_key)
        => ({outputs})
        {{
          present_key = Concat<axis=1>(past_key, new_key)
          {body_extra}
        }}
        """
    )


def _names(protos):
    return {p.name for p in protos}


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(out_names, feeds)
    return dict(zip(out_names, outputs))


def test_cpp_splits_pivot_and_rest_streams():
    model = _kv_model(seq_past=6, seq_new=2, batch=2, head_dim=3)
    q = onnxsim.apply_intactkv_cpp(model)
    onnx.checker.check_model(q)

    in_names = _names(q.graph.input)
    out_names = _names(q.graph.output)
    assert "past_key_pivot" in in_names
    assert "past_key_rest" in in_names
    assert "new_key" in in_names  # untouched
    assert "present_key_pivot" in out_names
    assert "present_key_rest" in out_names
    assert "present_key" in out_names  # original binding, now reconstructed
    assert "attn_out" in out_names

    pivot_in = next(i for i in q.graph.input if i.name == "past_key_pivot")
    pivot_dims = [d.dim_value for d in pivot_in.type.tensor_type.shape.dim]
    assert pivot_dims == [2, _NUM_PIVOT, 3]

    # past_key_rest is past_key renamed IN PLACE -- intactkv.py's own
    # _split_pivot_stream (matched exactly here) never touches its own
    # declared shape, so it keeps whatever past_key originally declared for
    # its seq axis ("seq_past", symbolic here) rather than being narrowed to
    # the semantically-correct seq_past - num_pivot_tokens.
    rest_in = next(i for i in q.graph.input if i.name == "past_key_rest")
    dims = rest_in.type.tensor_type.shape.dim
    assert dims[0].dim_value == 2
    assert dims[1].dim_param == "seq_past"
    assert dims[2].dim_value == 3

    # present_key_rest is a NEW graph output, its own declared seq axis
    # cleared entirely (neither dim_value nor dim_param) -- unlike
    # past_key_rest above, this one really is left symbolic/unset.
    rest_out = next(o for o in q.graph.output if o.name == "present_key_rest")
    out_dims = rest_out.type.tensor_type.shape.dim
    assert out_dims[0].dim_value == 2
    assert not out_dims[1].HasField("dim_value") and not out_dims[1].HasField(
        "dim_param"
    )
    assert out_dims[2].dim_value == 3

    # Exactly 2 new nodes added (Identity + reconstruction Concat), the
    # original Concat kept (now renamed/reshaped in place, not replaced).
    assert len(q.graph.node) == len(model.graph.node) + 2
    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("Concat") == 2
    assert op_types.count("Identity") == 2  # original attn_out Identity + new pivot one


def test_cpp_end_to_end_matches_unsplit_reference():
    # Feed a full past_key array, split by hand into pivot/rest halves,
    # through the TRANSFORMED model, and check every output matches the
    # ORIGINAL (unsplit) model run with the concatenated past_key -- a
    # pure reshuffle should reproduce the exact same numbers, not just a
    # close approximation.
    rng = np.random.default_rng(0)
    seq_past, seq_new, batch, head_dim = 6, 2, 2, 3
    full_past = rng.standard_normal((batch, seq_past, head_dim)).astype(np.float32)
    new_key = rng.standard_normal((batch, seq_new, head_dim)).astype(np.float32)

    model = _kv_model(seq_past, seq_new, batch, head_dim)
    ref = _run(model, {"past_key": full_past, "new_key": new_key})

    q = onnxsim.apply_intactkv_cpp(model)
    onnx.checker.check_model(q)
    pivot = full_past[:, :_NUM_PIVOT, :]
    rest = full_past[:, _NUM_PIVOT:, :]
    got = _run(
        q,
        {"past_key_pivot": pivot, "past_key_rest": rest, "new_key": new_key},
    )

    np.testing.assert_array_equal(got["present_key_pivot"], pivot)
    np.testing.assert_array_equal(
        got["present_key_rest"], np.concatenate([rest, new_key], axis=1)
    )
    np.testing.assert_array_equal(got["present_key"], ref["present_key"])
    np.testing.assert_array_equal(got["attn_out"], ref["attn_out"])


def test_cpp_matches_python_reference_structurally_and_numerically():
    model = _kv_model(seq_past=6, seq_new=2, batch=2, head_dim=3)
    py = apply_intactkv(model)
    cpp = onnxsim.apply_intactkv_cpp(model)
    onnx.checker.check_model(cpp)

    assert _names(cpp.graph.input) == _names(py.graph.input)
    assert _names(cpp.graph.output) == _names(py.graph.output)

    rng = np.random.default_rng(1)
    pivot = rng.standard_normal((2, _NUM_PIVOT, 3)).astype(np.float32)
    rest = rng.standard_normal((2, 2, 3)).astype(np.float32)
    new_key = rng.standard_normal((2, 2, 3)).astype(np.float32)
    feeds = {
        "past_key_pivot": pivot,
        "past_key_rest": rest,
        "new_key": new_key,
    }
    py_out = _run(py, feeds)
    cpp_out = _run(cpp, feeds)
    for name in py_out:
        np.testing.assert_array_equal(cpp_out[name], py_out[name])


def test_cpp_handles_multiple_independent_kv_streams():
    # Key and Value streams present together, each its own independent
    # Concat(past, new, axis=seq) candidate -- both should be split, with
    # no cross-interference (this repo's own established multi-match
    # concern for a PredicateBasedPass, checked directly here since a
    # single-candidate test cannot rule out the pass touching an
    # unrelated node's own inputs/outputs by mistake).
    model = _model(
        """
        g (float[2,6,3] past_key, float[2,2,3] new_key,
           float[2,6,3] past_value, float[2,2,3] new_value)
        => (float[2,8,3] present_key, float[2,8,3] present_value)
        {
          present_key = Concat<axis=1>(past_key, new_key)
          present_value = Concat<axis=1>(past_value, new_value)
        }
        """
    )
    q = onnxsim.apply_intactkv_cpp(model)
    onnx.checker.check_model(q)
    in_names = _names(q.graph.input)
    out_names = _names(q.graph.output)
    for base in ("past_key", "past_value"):
        assert f"{base}_pivot" in in_names
        assert f"{base}_rest" in in_names
    for base in ("present_key", "present_value"):
        assert f"{base}_pivot" in out_names
        assert f"{base}_rest" in out_names
        assert base in out_names


def test_cpp_skips_stream_whose_past_has_another_consumer():
    # past_key is ALSO consumed by a second node (Shape) beyond the Concat.
    # intactkv.py's own matcher (and this port) requires the "past" ROLE's
    # own operand specifically to be consumed by nothing else -- but with
    # only past_key disqualified, new_key (single-consumer, float32, a
    # graph input) still independently qualifies for that same role, so
    # new_key_pivot/new_key_rest get split instead (confirmed identical
    # against onnxsim.apply_intactkv itself: this is the shared matcher's
    # own real behavior, not something specific to this port). Giving
    # new_key a second consumer too disqualifies BOTH Concat operands from
    # the "past" role, so nothing matches at all -- a genuine no-op.
    model = _model(
        """
        g (float[2,6,3] past_key, float[2,2,3] new_key)
        => (float[2,8,3] present_key, int64[3] past_key_shape,
            int64[3] new_key_shape)
        {
          present_key = Concat<axis=1>(past_key, new_key)
          past_key_shape = Shape(past_key)
          new_key_shape = Shape(new_key)
        }
        """
    )
    result = onnxsim.apply_intactkv_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_kv_cache_pattern_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_intactkv_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_concat_output_not_a_graph_output():
    # present_key feeds only Identity -- it is never itself a graph
    # output, so the Concat isn't matched at all (matches
    # intactkv.py's own `node.output[0] not in output_names` requirement).
    model = _model(
        """
        g (float[2,6,3] past_key, float[2,2,3] new_key) => (float[2,8,3] attn_out)
        {
          present_key = Concat<axis=1>(past_key, new_key)
          attn_out = Identity(present_key)
        }
        """
    )
    result = onnxsim.apply_intactkv_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
