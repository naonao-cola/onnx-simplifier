"""Tests for ``onnxsim.apply_gear_cpp`` -- the C++-backed port of
``onnxsim.apply_gear`` (GEAR's KV-cache low-rank-plus-sparse residual
compensation, see ``onnxsim/gear_entry.h``). Like
``tests/test_spqr_cpp.py``/``tests/test_pb_llm_cpp.py``, this runs the
model over real calibration data through a real ``onnxruntime``-backed
executor -- never a fake/mock executor.

This port's own low-rank projector fit reuses this repo's own hand-rolled
Jacobi SVD (not LAPACK) and its own outlier-channel selection uses a full
deterministic sort where the Python reference uses ``np.argsort`` (not
guaranteed stable) -- see ``gear_entry.h``'s own "ACCEPTED, PERMANENT
DIVERGENCE" note. So these tests check the reconstructed correction
numerically (via onnxruntime execution, tight tolerance -- Eckart-Young
uniqueness means the fitted projector itself, not just its effect, is
expected to agree closely for well-separated singular values) and the
outlier-channel SET (not raw sort order), rather than raw initializer
byte equality.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.gear import apply_gear
from onnxsim.onnx_simplifier import apply_gear_cpp

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _kv_model(seq_past=6, seq_new=2, batch=2, head_dim=8, extra_consumer=True):
    outputs = "float[{b},{s},{h}] present_key".format(
        b=batch, s=seq_past + seq_new, h=head_dim
    )
    body_extra = ""
    if extra_consumer:
        outputs += ", float[{b},{s},{h}] attn_out".format(
            b=batch, s=seq_past + seq_new, h=head_dim
        )
        body_extra = "attn_out = Identity(present_key)"
    return _model(
        f"""
        g (float[{batch},{seq_past},{head_dim}] past_key,
           float[{batch},{seq_new},{head_dim}] new_key)
        => ({outputs})
        {{
          present_key = Concat<axis=1>(past_key, new_key)
          {body_extra}
        }}
        """
    )


def _calibration(head_dim=8, num_samples=40, seq_new=2, batch=2, rank=2, seed=1):
    # A low-rank-plus-noise activation distribution, the same shape this
    # session's other calibration-driven ports use for a non-trivial
    # (non-identity) residual structure: gives the low-rank projector fit
    # something real to find, and a couple of persistently large-magnitude
    # channels so the sparse-outlier fit is genuinely exercised too.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, head_dim)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, head_dim)).astype(np.float32) * 0.05
    x[:, 0] *= 20.0  # persistent large-magnitude outlier channel.
    return [
        {
            "past_key": rng.standard_normal((batch, 6, head_dim)).astype(np.float32),
            "new_key": x[: batch * seq_new].reshape(batch, seq_new, head_dim),
        }
    ]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _feeds_for(quantized_model, seq_past=6, seq_new=2, batch=2, head_dim=8, seed=5):
    # ApplyGear (and apply_gear itself) rewrites `past_key`'s own graph
    # INPUT declaration from FLOAT to INT8 (see gear.py's own docstring --
    # `past` is meant to already be a quantized cache, DequantizeLinear'd
    # internally; only `new_key` stays declared FLOAT, QuantizeLinear'd
    # internally). So a feed for the TRANSFORMED model must supply a real
    # INT8 `past_key`, quantized through the same per-channel scale the fit
    # actually produced -- not raw float32 noise, which onnxruntime's own
    # session.run() rejects with a dtype mismatch (confirmed this is not a
    # port-specific issue: the pure-Python apply_gear's own output model
    # fails the identical way when fed float32 past_key).
    rng = np.random.default_rng(seed)
    past_float = rng.standard_normal((batch, seq_past, head_dim)).astype(np.float64)
    scale_t = next(
        t for t in quantized_model.graph.initializer if t.name.endswith("_scale")
    )
    scale = onnx.numpy_helper.to_array(scale_t).astype(np.float64)
    past_codes = np.clip(np.round(past_float / scale), -128, 127).astype(np.int8)
    return {
        "past_key": past_codes,
        "new_key": rng.standard_normal((batch, seq_new, head_dim)).astype(np.float32),
    }


def test_cpp_changes_past_and_present_dtype_to_int8():
    model = _kv_model()
    q = apply_gear_cpp(model, calibration_data=_calibration())
    onnx.checker.check_model(q)

    past_in = next(i for i in q.graph.input if i.name == "past_key")
    present_out = next(o for o in q.graph.output if o.name == "present_key")
    assert past_in.type.tensor_type.elem_type == onnx.TensorProto.INT8
    assert present_out.type.tensor_type.elem_type == onnx.TensorProto.INT8


def test_cpp_adds_correction_nodes_and_rewires_other_consumer():
    model = _kv_model()
    q = apply_gear_cpp(model, calibration_data=_calibration())

    op_types = [n.op_type for n in q.graph.node]
    assert "QuantizeLinear" in op_types
    assert op_types.count("DequantizeLinear") == 2
    assert op_types.count("Concat") == 2  # raw present_key + present_corrected
    assert "MatMul" in op_types  # low-rank term (rank default is nonzero)

    # attn_out (a real downstream consumer of the ORIGINAL present_key) must
    # now read the corrected reconstruction, not the raw INT8 present_key.
    attn_out_node = next(n for n in q.graph.node if n.op_type == "Identity")
    present_key_producer = next(
        n for n in q.graph.node if n.op_type == "Concat" and "past_key" in n.input
    )
    assert attn_out_node.input[0] != present_key_producer.output[0]


def test_cpp_correction_reduces_reconstruction_error_vs_plain_int8():
    # The whole point of GEAR: the low-rank+sparse correction should
    # reconstruct the "new" token's own true value more closely than plain
    # per-channel INT8 quantize/dequantize alone.
    head_dim = 8
    model = _kv_model(head_dim=head_dim)
    calib = _calibration(head_dim=head_dim)
    q = apply_gear_cpp(model, calibration_data=calib, rank=2, outlier_fraction=0.25)
    onnx.checker.check_model(q)

    feeds = _feeds_for(q, head_dim=head_dim, seed=9)
    # present_key itself stays the RAW INT8 stream by design (gear.py's own
    # graph-output DECLARATIONS are deliberately left alone -- only a real
    # downstream CONSUMER, like attn_out here via the rewired Identity, gets
    # the corrected float reconstruction; see
    # test_cpp_adds_correction_nodes_and_rewires_other_consumer's own
    # assertion of this same fact). So the "corrected" comparison below
    # must read attn_out, not present_key.
    (_present_key, attn_out) = _run(q, feeds)
    corrected_new = attn_out[:, 6:, :]  # the "new" tokens' own slice

    # Plain per-channel INT8 baseline, computed directly (same formula the
    # port's own fit uses for the base scale, no correction).
    scale_t = next(t for t in q.graph.initializer if t.name.endswith("_scale"))
    scale = onnx.numpy_helper.to_array(scale_t)
    new_raw = feeds["new_key"].astype(np.float64)
    codes = np.clip(np.round(new_raw / scale), -128, 127)
    baseline_new = (codes * scale).astype(np.float32)

    baseline_err = np.mean((new_raw.astype(np.float32) - baseline_new) ** 2)
    corrected_err = np.mean((new_raw.astype(np.float32) - corrected_new) ** 2)
    assert corrected_err < baseline_err


def test_cpp_matches_python_reference_numerically():
    head_dim = 8
    model = _kv_model(head_dim=head_dim)
    calib = _calibration(head_dim=head_dim, seed=3)

    py = apply_gear(model, calibration_data=calib, rank=2, outlier_fraction=0.25)
    cpp = apply_gear_cpp(model, calibration_data=calib, rank=2, outlier_fraction=0.25)
    onnx.checker.check_model(py)
    onnx.checker.check_model(cpp)

    # Same feed for both sides (quantized through `py`'s own fitted scale,
    # which should match `cpp`'s closely -- this is a closed-form abs-max
    # statistic, no RNG involved on either side) so the comparison is
    # apples-to-apples on identical INT8 past_key codes.
    feeds = _feeds_for(py, head_dim=head_dim, seed=11)
    py_out = _run(py, feeds)
    cpp_out = _run(cpp, feeds)
    for a, b in zip(py_out, cpp_out):
        np.testing.assert_allclose(
            a.astype(np.float64), b.astype(np.float64), rtol=1e-3, atol=1e-4
        )


def test_cpp_matches_python_sparse_mask_channel_set():
    # Outlier-channel SELECTION order can differ (documented divergence),
    # but the SET of channels chosen must agree.
    head_dim = 8
    model = _kv_model(head_dim=head_dim)
    calib = _calibration(head_dim=head_dim, seed=4)

    py = apply_gear(model, calibration_data=calib, rank=2, outlier_fraction=0.25)
    cpp = apply_gear_cpp(model, calibration_data=calib, rank=2, outlier_fraction=0.25)

    py_mask = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name.endswith("_sparse_mask"))
    )
    cpp_mask = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name.endswith("_sparse_mask"))
    )
    assert set(np.nonzero(py_mask)[0].tolist()) == set(np.nonzero(cpp_mask)[0].tolist())


def test_cpp_rank_zero_disables_low_rank_term():
    head_dim = 8
    model = _kv_model(head_dim=head_dim)
    calib = _calibration(head_dim=head_dim, seed=6)
    q = apply_gear_cpp(model, calibration_data=calib, rank=0, outlier_fraction=0.25)
    onnx.checker.check_model(q)
    op_types = [n.op_type for n in q.graph.node]
    assert "MatMul" not in op_types
    assert "Mul" in op_types  # sparse term still present


def test_cpp_outlier_fraction_zero_disables_sparse_term():
    head_dim = 8
    model = _kv_model(head_dim=head_dim)
    calib = _calibration(head_dim=head_dim, seed=7)
    q = apply_gear_cpp(model, calibration_data=calib, rank=2, outlier_fraction=0.0)
    onnx.checker.check_model(q)
    op_types = [n.op_type for n in q.graph.node]
    assert "Mul" not in op_types
    assert "MatMul" in op_types  # low-rank term still present


def test_cpp_noop_when_no_kv_cache_pattern_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_gear_cpp(
        model, calibration_data=[{"X": np.zeros((2, 4), np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_past_has_another_consumer():
    # Giving ONLY past_key a second consumer does NOT make this a no-op --
    # the matcher just swaps roles and treats new_key as "past" instead
    # (still single-consumer), the same subtlety already documented by
    # tests/test_intactkv_cpp.py's own precedent (and reconfirmed here
    # directly against the pure-Python apply_gear reference). Both operands
    # need a second consumer for a genuine no-op.
    model = _model(
        """
        g (float[2,6,8] past_key, float[2,2,8] new_key)
        => (float[2,8,8] present_key, int64[3] past_key_shape,
            int64[3] new_key_shape)
        {
          present_key = Concat<axis=1>(past_key, new_key)
          past_key_shape = Shape(past_key)
          new_key_shape = Shape(new_key)
        }
        """
    )
    result = apply_gear_cpp(
        model, calibration_data=_calibration(head_dim=8, seq_new=2, batch=2)
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset13():
    model = _kv_model()
    model.opset_import[0].version = 11
    result = apply_gear_cpp(model, calibration_data=_calibration())
    assert result.SerializeToString() == model.SerializeToString()
