"""Tests for ``onnxsim.apply_dsq_cpp`` -- the C++-backed port of
``onnxsim.apply_dsq`` (D2Quant's Dual-Scale Quantizer, see
``onnxsim/passes/d2quant_dsq.h``). This is a closed-form, deterministic
scheme with no RNG or fitting algorithm beyond a fixed alternating
least-squares loop, so these tests check tight numeric agreement against
the pure-Python reference, not just structural/algebraic properties.

Unlike ``tests/test_d2quant.py``'s own tests (which exercise the Python
reference's own ``block_size`` parameter, e.g. ``block_size=4``), this
port hardcodes D2Quant's own default ``block_size=32``/``num_iterations=15``
(see ``d2quant_dsq.h``'s own SCOPE NARROWING note), so every SwiGLU test
fixture below uses ``H`` (the gated hidden dimension) as a multiple of 32,
and every pure-Python comparison calls ``onnxsim.apply_dsq`` with no
explicit ``block_size``/``num_iterations`` (using its own matching
defaults).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_H = 32  # this port's own hardcoded block_size


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _decode_int4_signed(t: onnx.TensorProto) -> np.ndarray:
    dims = list(t.dims)
    numel = int(np.prod(dims)) if dims else 0
    raw = np.frombuffer(t.raw_data, dtype=np.uint8)
    lo = raw & 0x0F
    hi = (raw >> 4) & 0x0F
    nibbles = np.empty(numel, dtype=np.uint8)
    nibbles[0::2] = lo[: (numel + 1) // 2]
    nibbles[1::2] = hi[: numel // 2]
    signed = nibbles.astype(np.int64)
    signed = np.where(signed >= 8, signed - 16, signed)
    return signed.reshape(dims)


def _cpp_up_weight(q: onnx.ModelProto) -> np.ndarray:
    """Finds the up-projection's own CURRENT weight in the C++ port's own
    output by following the actual graph wiring, not by name: unlike
    ``apply_dsq``'s own in-place `up_w_init.CopyFrom(...)` (which keeps the
    "Wup" name), this port -- like every other ``*_cpp`` port in this repo
    -- rescales the up-projection's weight into a freshly created
    initializer with an auto-generated name, rewiring UpProj's own input
    to it and leaving the ORIGINAL "Wup" initializer orphaned, unchanged,
    and unused in the graph. Looking it up by the stale "Wup" name would
    silently find that orphaned original instead of the real value the
    model actually computes with.
    """
    up_node = next(
        n for n in q.graph.node if n.op_type == "MatMul" and n.output[0] == "UpProj"
    )
    return onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == up_node.input[1])
    )


def _swiglu_model(
    D=8,
    H=_H,
    Dout=6,
    extra_up_consumer=False,
    tied_up_weight=False,
    gemm_down=False,
    seed=0,
):
    # Mirrors tests/test_d2quant.py's own _swiglu_model exactly (same
    # SwiGLU shape: down(silu(gate(x)) * up(x))), generalized with a
    # tied_up_weight knob this port's own test suite adds to exercise the
    # "up-proj's own WEIGHT has an extra consumer" decline path (distinct
    # from "up-proj's own OUTPUT has an extra consumer",
    # extra_up_consumer).
    rng = np.random.default_rng(seed)
    w_gate = rng.standard_normal((D, H)).astype(np.float32) * 0.3
    w_up = rng.standard_normal((D, H)).astype(np.float32) * 0.3
    w_down = rng.standard_normal((H, Dout)).astype(np.float32) * 0.3
    # A few outlier-heavy columns in w_down -- DSQ's own target scenario.
    w_down[:, 0] *= 6.0

    down_body = (
        "Y = MatMul(Gated, Wdown)"
        if not gemm_down
        else "Y = Gemm<transB = 1>(Gated, WdownT)"
    )
    w_down_for_graph = w_down if not gemm_down else w_down.T.copy()
    down_init = _f32(w_down_for_graph, "WdownT" if gemm_down else "Wdown")

    extra_output = ""
    extra_line = ""
    if extra_up_consumer:
        extra_output = f", float[batch,{H}] UpOut"
        extra_line = "UpOut = Identity(UpProj)"
    elif tied_up_weight:
        extra_output = f", float[batch,{H}] TiedOut"
        extra_line = "TiedOut = MatMul(X, Wup)"

    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{Dout}] Y{extra_output})
        {{
          GateProj = MatMul(X, Wgate)
          UpProj = MatMul(X, Wup)
          Sig = Sigmoid(GateProj)
          Silu = Mul(GateProj, Sig)
          Gated = Mul(Silu, UpProj)
          {down_body}
          {extra_line}
        }}
        """,
        [_f32(w_gate, "Wgate"), _f32(w_up, "Wup"), down_init],
    )
    return model, w_gate, w_up, w_down


def test_cpp_quantizes_down_proj_and_rescales_up_proj():
    model, w_gate, w_up, w_down = _swiglu_model(seed=0)
    q = onnxsim.apply_dsq_cpp(model)
    onnx.checker.check_model(q)

    down_node = next(
        n for n in q.graph.node if n.op_type == "MatMul" and n.output[0] == "Y"
    )
    dq_node = next(n for n in q.graph.node if n.output[0] == down_node.input[1])
    assert dq_node.op_type == "DequantizeLinear"
    attrs = {a.name: a for a in dq_node.attribute}
    assert attrs["axis"].i == 0
    assert attrs["block_size"].i == _H

    wq = next(t for t in q.graph.initializer if t.name == dq_node.input[0])
    assert wq.data_type == onnx.TensorProto.INT4
    assert list(wq.dims) == [_H, 6]

    up_after = _cpp_up_weight(q)
    assert not np.allclose(up_after, w_up)
    # Every column's rescale ratio must be a single constant down that
    # column (only the up-proj's own OUTPUT channel -- H -- is rescaled).
    ratio = up_after.astype(np.float64) / w_up.astype(np.float64)
    assert np.allclose(ratio, ratio[0:1, :], rtol=1e-5)


def test_cpp_matches_python_reference_exactly():
    model, _, w_up, w_down = _swiglu_model(seed=1)
    py = onnxsim.apply_dsq(model)
    cpp = onnxsim.apply_dsq_cpp(model)
    onnx.checker.check_model(cpp)

    py_wq = next(t for t in py.graph.initializer if t.name.endswith("_dsq_q"))
    py_ws = next(t for t in py.graph.initializer if t.name.endswith("_dsq_scale"))
    py_codes = _decode_int4_signed(py_wq)

    # This port never names its own new initializers (unlike the Python
    # reference's own "_dsq_q"/"_dsq_scale" suffixes) -- find the INT4
    # weight directly by dtype, then follow the DequantizeLinear node that
    # actually consumes it (wq, ws) to the paired scale initializer, the
    # same robust-by-wiring-not-by-name convention this session's other
    # exact-parity tests already establish.
    cpp_wq = next(
        t for t in cpp.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    dq_node = next(
        n
        for n in cpp.graph.node
        if n.op_type == "DequantizeLinear" and n.input[0] == cpp_wq.name
    )
    cpp_ws = next(t for t in cpp.graph.initializer if t.name == dq_node.input[1])
    cpp_codes = _decode_int4_signed(cpp_wq)
    np.testing.assert_array_equal(cpp_codes, py_codes)
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(cpp_ws),
        onnx.numpy_helper.to_array(py_ws),
        rtol=1e-4,
        atol=1e-6,
    )

    py_up = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name == "Wup")
    )
    cpp_up = _cpp_up_weight(cpp)
    np.testing.assert_allclose(cpp_up, py_up, rtol=1e-4, atol=1e-6)


def test_cpp_reconstruction_matches_original_weight():
    D, Dout = 8, 6
    model, _, w_up, w_down = _swiglu_model(D=D, Dout=Dout, seed=2)
    q = onnxsim.apply_dsq_cpp(model)

    down_node = next(
        n for n in q.graph.node if n.op_type == "MatMul" and n.output[0] == "Y"
    )
    dq_node = next(n for n in q.graph.node if n.output[0] == down_node.input[1])
    wq = next(t for t in q.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in q.graph.initializer if t.name == dq_node.input[1])
    codes = _decode_int4_signed(wq).astype(np.float64)  # [H, Dout]
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)  # [1, Dout]
    dequant_normalized = codes * scale

    up_after = _cpp_up_weight(q).astype(np.float64)
    s_c_recovered = (up_after / w_up.astype(np.float64)).mean(axis=0)  # [H]

    reconstructed = dequant_normalized * s_c_recovered[:, np.newaxis]
    rel_err = np.linalg.norm(reconstructed - w_down) / np.linalg.norm(w_down)
    assert rel_err < 0.1


def test_cpp_output_stays_close_to_float_via_onnxruntime():
    model, *_ = _swiglu_model(seed=3)
    q = onnxsim.apply_dsq_cpp(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 8)).astype(np.float32)
    (float_y,) = _run(model, {"X": x}, output_names=["Y"]).values()
    (q_y,) = _run(q, {"X": x}, output_names=["Y"]).values()
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.25


def test_cpp_gemm_transb_down_proj():
    model, *_ = _swiglu_model(gemm_down=True, seed=5)
    q = onnxsim.apply_dsq_cpp(model)
    onnx.checker.check_model(q)

    dq_nodes = [n for n in q.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dq_nodes) == 1
    attrs = {a.name: a for a in dq_nodes[0].attribute}
    assert attrs["axis"].i == 1  # transB=1 stores W as [N, K] -- reduction is axis 1

    rng = np.random.default_rng(6)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    (float_y,) = _run(model, {"X": x}, output_names=["Y"]).values()
    (q_y,) = _run(q, {"X": x}, output_names=["Y"]).values()
    assert _rel_l2(float_y, q_y) < 0.25


def test_cpp_declines_when_up_proj_output_has_extra_consumer():
    model, *_ = _swiglu_model(extra_up_consumer=True, seed=7)
    q = onnxsim.apply_dsq_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_declines_when_up_proj_weight_is_tied():
    model, *_ = _swiglu_model(tied_up_weight=True, seed=8)
    q = onnxsim.apply_dsq_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_swiglu_pattern():
    model = _model(
        f"""
        g (float[4,{_H}] X) => (float[4,6] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(np.random.default_rng(9).standard_normal((_H, 6)), "W")],
    )
    q = onnxsim.apply_dsq_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_h_not_divisible_by_block_size():
    # H = 17 is not a multiple of this port's own hardcoded block_size
    # (32) -- d2quant.py's own encoder skips this block entirely, and this
    # port matches that exactly.
    model, *_ = _swiglu_model(H=17, seed=10)
    q = onnxsim.apply_dsq_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_noop_below_opset_21():
    model, *_ = _swiglu_model(seed=11)
    model.opset_import[0].version = 17
    q = onnxsim.apply_dsq_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()
