"""Tests for ``onnxsim.quantize_kv_cache_cpp`` -- the C++-backed port of
``onnxsim.quantize_kv_cache`` (KV-cache quantization, KIVI/KVQuant, see
``onnxsim/kv_cache_quantization_entry.h``). Like
``tests/test_spqr_cpp.py``/``tests/test_pb_llm_cpp.py``, this runs the
model over real calibration data through a real ``onnxruntime``-backed
executor -- never a fake/mock executor. This technique has no RNG or
fitting algorithm anywhere (a closed-form per-channel abs-max for
Key-style, a data-free per-token abs-max for Value-style), so these tests
check numerically tight agreement against the pure-Python reference for
the values that matter (scale/zero-point), not just structural/algebraic
properties -- plus a genuine onnxruntime execution check that the
rewritten graph actually runs.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.kv_cache_quantization import quantize_kv_cache

ort = pytest.importorskip("onnxruntime")


def _model(body, opset=18, ir_version=9):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _kv_model(
    present_name="present_key",
    seq_past=6,
    seq_new=2,
    batch=2,
    head_dim=4,
    opset=18,
    extra_consumer=True,
):
    """A minimal decoder-step-shaped graph: a single KV-cache stream (a
    graph output named ``present_name``) that ALSO feeds a further
    consumer (``attn_out``, standing in for the attention math) when
    ``extra_consumer`` is set -- exercising this port's own claim that
    every pre-existing NODE consumer, not just the graph output binding,
    gets retargeted onto the dequantized reconstruction.

    ``present_name`` is always double-quoted in the generated text: the
    ".value"-suffixed names this module's own Value-style matching cares
    about (e.g. ``present.0.value``) contain a ``.``, which onnx.parser's
    own bare-identifier grammar rejects (``CLAUDE.md``'s documented
    exception for when the text format can't express what's needed) --
    quoting works uniformly for both dotted and plain names.
    """
    quoted = f'"{present_name}"'
    outputs = f"float[{batch},{seq_past + seq_new},{head_dim}] {quoted}"
    body_extra = ""
    if extra_consumer:
        outputs += f", float[{batch},{seq_past + seq_new},{head_dim}] attn_out"
        body_extra = f"attn_out = Identity({quoted})"
    return _model(
        f"""
        g (float[{batch},{seq_past},{head_dim}] past_key,
           float[{batch},{seq_new},{head_dim}] new_key)
        => ({outputs})
        {{
          {quoted} = Concat<axis=1>(past_key, new_key)
          {body_extra}
        }}
        """,
        opset=opset,
    )


def _calibration(seq_past=6, seq_new=2, batch=2, head_dim=4, num_samples=4, seed=1):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        past_key = rng.standard_normal((batch, seq_past, head_dim)).astype(np.float32)
        new_key = rng.standard_normal((batch, seq_new, head_dim)).astype(np.float32)
        new_key[:, :, 0] *= 8.0  # a persistently large-magnitude channel
        batches.append({"past_key": past_key, "new_key": new_key})
    return batches


def _names(protos):
    return {p.name for p in protos}


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(out_names, feeds)
    return dict(zip(out_names, outputs))


def test_cpp_key_style_matches_python_scale_and_zero_point():
    model = _kv_model()
    calib = _calibration()
    py = quantize_kv_cache(model, calibration_data=calib)
    cpp = onnxsim.quantize_kv_cache_cpp(model, calibration_data=calib)
    onnx.checker.check_model(cpp)

    py_scale = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name.endswith("_kv_scale"))
    )
    cpp_scale = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name.endswith("_kv_scale"))
    )
    np.testing.assert_allclose(cpp_scale, py_scale, rtol=1e-6, atol=1e-9)

    py_zp = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name.endswith("_kv_zero_point"))
    )
    cpp_zp = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name.endswith("_kv_zero_point"))
    )
    np.testing.assert_array_equal(cpp_zp, py_zp)

    past_key_in = next(i for i in cpp.graph.input if i.name == "past_key")
    assert past_key_in.type.tensor_type.elem_type == onnx.TensorProto.INT8
    present_key_out = next(o for o in cpp.graph.output if o.name == "present_key")
    assert present_key_out.type.tensor_type.elem_type == onnx.TensorProto.INT8
    op_types = [n.op_type for n in cpp.graph.node]
    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types
    # attn_out is retargeted onto the dequantized float reconstruction, not
    # the now-INT8 present_key.
    dq = next(n for n in cpp.graph.node if n.op_type == "DequantizeLinear")
    attn = next(n for n in cpp.graph.node if n.output[0] == "attn_out")
    assert attn.input[0] == dq.output[0]


def test_cpp_key_style_runs_with_empty_past_cache():
    # An empty starting past_key (seq_past=0) sidesteps needing to
    # pre-quantize meaningful past cache data by hand -- Concat of a
    # zero-length and non-zero-length operand along the sequence axis is
    # ordinary, well-supported ONNX/onnxruntime behavior.
    model = _kv_model(seq_past=0, seq_new=3, batch=1, head_dim=4)
    calib = _calibration(seq_past=0, seq_new=3, batch=1, head_dim=4, seed=2)
    cpp = onnxsim.quantize_kv_cache_cpp(model, calibration_data=calib)
    onnx.checker.check_model(cpp)

    rng = np.random.default_rng(3)
    feeds = {
        "past_key": np.zeros((1, 0, 4), dtype=np.int8),
        "new_key": rng.standard_normal((1, 3, 4)).astype(np.float32),
    }
    outputs = _run(cpp, feeds)
    assert np.all(np.isfinite(outputs["attn_out"]))
    assert outputs["present_key"].dtype == np.int8
    assert outputs["attn_out"].shape == (1, 3, 4)


def test_cpp_value_style_matched_by_name_needs_no_calibration():
    model = _kv_model(present_name="present.0.value", seq_past=0, seq_new=3, batch=1)
    py = quantize_kv_cache(model, calibration_data=[])
    cpp = onnxsim.quantize_kv_cache_cpp(model, calibration_data=[])
    onnx.checker.check_model(cpp)

    assert _names(cpp.graph.input) == _names(py.graph.input)
    assert _names(cpp.graph.output) == _names(py.graph.output)
    assert "past_key_scale" in _names(cpp.graph.input)
    assert "present.0.value_scale" in _names(cpp.graph.output)
    op_types = [n.op_type for n in cpp.graph.node]
    assert "QuantizeLinear" not in op_types  # data-free -- no calibrated scale
    assert "ReduceMax" in op_types

    rng = np.random.default_rng(4)
    feeds = {
        "past_key": np.zeros((1, 0, 4), dtype=np.int8),
        "past_key_scale": np.zeros((1, 0, 1), dtype=np.float32),
        "new_key": rng.standard_normal((1, 3, 4)).astype(np.float32),
    }
    outputs = _run(cpp, feeds)
    assert np.all(np.isfinite(outputs["attn_out"]))
    assert outputs["present.0.value"].dtype == np.int8
    assert outputs["present.0.value_scale"].shape == (1, 3, 1)


def test_cpp_value_style_via_explicit_value_output_names():
    # present_key's own name has no ".value" substring, so it only gets
    # Value-style treatment when explicitly requested.
    model = _kv_model(seq_past=0, seq_new=2, batch=1)
    cpp_default = onnxsim.quantize_kv_cache_cpp(model, calibration_data=[])
    assert "QuantizeLinear" not in [n.op_type for n in cpp_default.graph.node]
    # No calibration data at all and no Value-style match -> genuinely a
    # no-op (channel-style needs calibration it was never given).
    assert cpp_default.SerializeToString() == model.SerializeToString()

    cpp_explicit = onnxsim.quantize_kv_cache_cpp(
        model, calibration_data=[], value_output_names=["present_key"]
    )
    onnx.checker.check_model(cpp_explicit)
    assert "past_key_scale" in _names(cpp_explicit.graph.input)
    assert "present_key_scale" in _names(cpp_explicit.graph.output)


def test_cpp_value_style_declines_pre_opset18():
    model = _kv_model(present_name="present.0.value", opset=13)
    result = onnxsim.quantize_kv_cache_cpp(model, calibration_data=[])
    # Below opset 18, a Value-style-eligible stream is left completely
    # untouched (not downgraded to Key-style) -- and with no channel
    # candidates either, the whole model is unchanged.
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_matches_python_numerically_end_to_end():
    model = _kv_model(seq_past=0, seq_new=2, batch=1, head_dim=4)
    calib = _calibration(seq_past=0, seq_new=2, batch=1, head_dim=4, seed=5)
    py = quantize_kv_cache(model, calibration_data=calib)
    cpp = onnxsim.quantize_kv_cache_cpp(model, calibration_data=calib)

    rng = np.random.default_rng(6)
    feeds = {
        "past_key": np.zeros((1, 0, 4), dtype=np.int8),
        "new_key": rng.standard_normal((1, 2, 4)).astype(np.float32),
    }
    py_out = _run(py, feeds)
    cpp_out = _run(cpp, feeds)
    np.testing.assert_array_equal(cpp_out["present_key"], py_out["present_key"])
    np.testing.assert_allclose(
        cpp_out["attn_out"], py_out["attn_out"], rtol=1e-5, atol=1e-6
    )


def test_cpp_handles_multiple_independent_kv_streams():
    # Key and Value streams present together, each its own independent
    # Concat(past, new, axis=seq) candidate -- both should be handled,
    # with no cross-interference (this repo's own established
    # multiple-KV-stream concern, checked directly here the same way
    # tests/test_intactkv_cpp.py's own
    # test_cpp_handles_multiple_independent_kv_streams does).
    model = _model(
        """
        g (float[1,4,4] past_key, float[1,2,4] new_key,
           float[1,4,4] past_value, float[1,2,4] new_value)
        => (float[1,6,4] present_key, float[1,6,4] present_value_dot_value)
        {
          present_key = Concat<axis=1>(past_key, new_key)
          present_value_dot_value = Concat<axis=1>(past_value, new_value)
        }
        """
    )
    calib = [
        {
            "past_key": np.random.default_rng(7)
            .standard_normal((1, 4, 4))
            .astype(np.float32),
            "new_key": np.random.default_rng(8)
            .standard_normal((1, 2, 4))
            .astype(np.float32),
            "past_value": np.random.default_rng(9)
            .standard_normal((1, 4, 4))
            .astype(np.float32),
            "new_value": np.random.default_rng(10)
            .standard_normal((1, 2, 4))
            .astype(np.float32),
        }
    ]
    cpp = onnxsim.quantize_kv_cache_cpp(
        model, calibration_data=calib, value_output_names=["present_value_dot_value"]
    )
    onnx.checker.check_model(cpp)
    in_names = _names(cpp.graph.input)
    out_names = _names(cpp.graph.output)
    assert "present_key" in out_names
    assert "present_value_dot_value" in out_names
    # Key-style stream: calibrated scale/zero-point, no new scale I/O pair.
    assert any(t.name.endswith("_kv_scale") for t in cpp.graph.initializer)
    # Value-style stream: new scale input/output pair.
    assert "past_value_scale" in in_names
    assert "present_value_dot_value_scale" in out_names


def test_cpp_skips_stream_whose_past_has_another_consumer():
    # past_key is ALSO consumed by a second node (Shape) beyond the
    # Concat -- but with only past_key disqualified from the "past" role
    # (consumed by more than just the Concat), new_key (single-consumer,
    # float32, a graph input) still independently qualifies for that same
    # role instead (confirmed identical against the pure-Python
    # ``quantize_kv_cache`` reference itself: this is the shared matcher's
    # own real, if surprising, behavior -- see
    # ``tests/test_intactkv_cpp.py``'s own identically-shaped test and its
    # own comment for the first place this session worked out why). Giving
    # new_key a second consumer too disqualifies BOTH Concat operands from
    # the "past" role, so nothing matches at all -- a genuine no-op.
    model = _model(
        """
        g (float[1,4,4] past_key, float[1,2,4] new_key)
        => (float[1,6,4] present_key, int64[3] past_key_shape,
            int64[3] new_key_shape)
        {
          present_key = Concat<axis=1>(past_key, new_key)
          past_key_shape = Shape(past_key)
          new_key_shape = Shape(new_key)
        }
        """
    )
    result = onnxsim.quantize_kv_cache_cpp(
        model,
        calibration_data=[
            {
                "past_key": np.zeros((1, 4, 4), dtype=np.float32),
                "new_key": np.zeros((1, 2, 4), dtype=np.float32),
            }
        ],
    )
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
    result = onnxsim.quantize_kv_cache_cpp(model, calibration_data=[])
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset13():
    model = _kv_model(opset=12)
    result = onnxsim.quantize_kv_cache_cpp(model, calibration_data=_calibration())
    assert result.SerializeToString() == model.SerializeToString()
