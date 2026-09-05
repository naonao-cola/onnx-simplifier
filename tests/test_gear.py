"""Tests for ``onnxsim.apply_gear`` -- see ``onnxsim/gear.py`` for the
technique (GEAR-style low-rank-plus-sparse residual compensation layered on
top of ``onnxsim.kv_cache_quantization``'s own static, per-channel INT8
quantization of a decoder's ``Concat(past, new, axis=seq)`` KV-cache
stream).
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


def _kv_cache_model(batch=1, heads=2, head_dim=8, opset=13, symbolic_seq=True):
    # new_key is routed through an Identity node rather than being a bare
    # graph input, matching tests/test_kv_cache_quantization.py's own
    # rationale: a real exported decoder's "new" K/V is always a freshly
    # computed activation, never a raw model input, and the matcher can't
    # otherwise tell "the persistent cache" apart from "this step's fresh
    # token" when both Concat operands are plain graph inputs with a single
    # consumer.
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


def test_gear_changes_input_and_output_to_int8():
    model = _kv_cache_model()
    q = onnxsim.apply_gear(model, num_samples=4, seed=0)
    onnx.checker.check_model(q)

    past_vi = next(i for i in q.graph.input if i.name == "past_key")
    present_vi = next(o for o in q.graph.output if o.name == "present_key")
    assert past_vi.type.tensor_type.elem_type == onnx.TensorProto.INT8
    assert present_vi.type.tensor_type.elem_type == onnx.TensorProto.INT8
    # new_key_raw/the Identity's "new_key" output are untouched float --
    # only Concat's own operand is rewired to a freshly quantized tensor.
    new_vi = next(i for i in q.graph.input if i.name == "new_key_raw")
    assert new_vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_gear_inserts_low_rank_and_sparse_correction_nodes():
    model = _kv_cache_model()
    q = onnxsim.apply_gear(model, num_samples=4, seed=0, rank=2, outlier_fraction=0.25)
    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("QuantizeLinear") == 1
    assert op_types.count("DequantizeLinear") == 2  # past + new
    assert "MatMul" in op_types  # low-rank projector
    assert "Mul" in op_types  # sparse mask
    assert op_types.count("Concat") == 2  # codes stream + corrected-float stream
    assert op_types.count("Add") == 2  # dequant + low_rank, then + sparse

    # ReduceSum (the "attention math" stand-in) must consume the corrected
    # float reconstruction, not the raw int8 Concat output.
    reduce_node = next(n for n in q.graph.node if n.op_type == "ReduceSum")
    corrected_concat = [n for n in q.graph.node if n.op_type == "Concat"][-1]
    assert reduce_node.input[0] == corrected_concat.output[0]


def test_gear_rank_zero_skips_low_rank_term():
    model = _kv_cache_model()
    q = onnxsim.apply_gear(model, num_samples=4, seed=0, rank=0, outlier_fraction=0.25)
    op_types = [n.op_type for n in q.graph.node]
    assert "MatMul" not in op_types
    assert "Mul" in op_types
    assert op_types.count("Add") == 1  # only the sparse term is added


def test_gear_outlier_fraction_zero_skips_sparse_term():
    model = _kv_cache_model()
    q = onnxsim.apply_gear(model, num_samples=4, seed=0, rank=2, outlier_fraction=0.0)
    op_types = [n.op_type for n in q.graph.node]
    assert "MatMul" in op_types
    assert "Mul" not in op_types
    assert op_types.count("Add") == 1  # only the low-rank term is added


def test_gear_noop_without_kv_cache_pattern():
    model = _model(
        """
        g (float[4,4] x) => (float[4,4] y)
        {
          y = Relu(x)
        }
        """
    )
    result = onnxsim.apply_gear(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gear_noop_below_opset13():
    model = _kv_cache_model(opset=12)
    result = onnxsim.apply_gear(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gear_declines_when_past_has_other_consumers():
    model = _model(
        """
        g (float[1,2,3,8] past_key, float[1,2,1,8] new_key_raw)
          => (float[1,2,4,8] present_key, float summary, float past_summary)
        {
          new_key = Identity(new_key_raw)
          present_key = Concat<axis = 2>(past_key, new_key)
          summary = ReduceSum<keepdims = 0>(present_key)
          past_summary = ReduceSum<keepdims = 0>(past_key)
        }
        """
    )
    q = onnxsim.apply_gear(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_gear_new_token_reconstruction_matches_numpy_reference():
    # Verify the graph's own low-rank+sparse correction against a from-
    # scratch numpy re-derivation of the same calibration fit, computed
    # directly from the ONNX initializers this module wrote -- per this
    # project's own numerics convention, a tight *relative* tolerance
    # against numpy rather than a loose absolute tolerance against an
    # onnxruntime round-trip (onnxruntime's own MatMul reduction order is
    # not bit-exact across CPU architectures).
    batch, heads, head_dim = 1, 2, 8
    model = _kv_cache_model(batch=batch, heads=heads, head_dim=head_dim)
    q = onnxsim.apply_gear(model, num_samples=32, seed=1, rank=2, outlier_fraction=0.25)
    onnx.checker.check_model(q)

    init = {t.name: onnx.numpy_helper.to_array(t) for t in q.graph.initializer}
    scale = init["present_key_gear_scale"].astype(np.float64)
    zero_point = init["present_key_gear_zero_point"].astype(np.float64)
    projector = init["present_key_gear_p"].astype(np.float64)
    mask = init["present_key_gear_sparse_mask"].astype(np.float64)

    rng = np.random.default_rng(7)
    empty_past = np.zeros((batch, heads, 0, head_dim), dtype=np.int8)
    new0 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)

    corrected_concat = [n for n in q.graph.node if n.op_type == "Concat"][-1]
    q_probe = onnx.ModelProto()
    q_probe.CopyFrom(q)
    q_probe.graph.output.append(onnx.ValueInfoProto(name=corrected_concat.output[0]))

    _present0, _summary0, present0_corrected = _run(
        q_probe, {"past_key": empty_past, "new_key_raw": new0}
    )

    new0_f64 = new0.astype(np.float64)
    codes = np.clip(np.round(new0_f64 / scale), -128, 127)
    new_dequant = (codes - zero_point) * scale
    residual = new0_f64 - new_dequant
    low_rank = residual @ projector
    remainder = residual - low_rank
    sparse = remainder * mask
    expected = new_dequant + low_rank + sparse

    rel_err = np.linalg.norm(present0_corrected - expected) / np.linalg.norm(expected)
    assert rel_err < 1e-5


def test_gear_two_step_round_trip_close_to_float():
    batch, heads, head_dim = 1, 2, 8
    model = _kv_cache_model(batch=batch, heads=heads, head_dim=head_dim)
    q = onnxsim.apply_gear(model, num_samples=64, seed=2, rank=2, outlier_fraction=0.25)
    corrected_concat = [n for n in q.graph.node if n.op_type == "Concat"][-1]
    q_probe = onnx.ModelProto()
    q_probe.CopyFrom(q)
    q_probe.graph.output.append(onnx.ValueInfoProto(name=corrected_concat.output[0]))

    rng = np.random.default_rng(4)
    empty_past = np.zeros((batch, heads, 0, head_dim), dtype=np.float32)
    new0 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)
    new1 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)

    float_present0, _ = _run(model, {"past_key": empty_past, "new_key_raw": new0})
    float_present1, _ = _run(model, {"past_key": float_present0, "new_key_raw": new1})

    empty_past_q = np.zeros((batch, heads, 0, head_dim), dtype=np.int8)
    q_present0, _ = _run(q, {"past_key": empty_past_q, "new_key_raw": new0})
    assert q_present0.dtype == np.int8
    q_present1, _, q_present1_corrected = _run(
        q_probe, {"past_key": q_present0, "new_key_raw": new1}
    )
    assert q_present1.dtype == np.int8

    # A generous bound, not a tight one -- matches this project's own
    # convention for a small random-calibration-set end-to-end check (see
    # tests/test_kv_cache_quantization.py's own 0.2 bound).
    assert _rel_l2(float_present1, q_present1_corrected) < 0.2
