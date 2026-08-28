"""Integration tests: onnxsim.simplify() against ONNX graphs following
TensorRT's documented explicit-quantization QDQ conventions.

TensorRT's explicit-quantization mode (the mode ONNX-based INT8/FP8 workflows
use -- e.g. the real NVIDIA ModelOpt output covered by
``test_modelopt_integration.py``) takes the ``QuantizeLinear``/
``DequantizeLinear`` (QDQ) node pairs already present in an ONNX graph as
ground truth for where quantization happens and at what scale, then fuses
each QDQ pair into its neighboring op during engine building. TensorRT's own
developer guide ("Q/DQ Layer-Placement Recommendations") documents specific
conventions that placement is expected to follow:

- weights are quantized **per-channel** (one scale per output channel, axis 0
  for Conv/Deconv/Gemm-style weights), while activations are quantized
  **per-tensor** (a single scalar scale);
- INT8 quantization is **symmetric** (zero point 0) -- TensorRT's explicit
  INT8 path does not support a nonzero zero point;
- a residual/shortcut ``Add``'s branches are **each independently quantized**
  (their own QDQ pair per branch, even when both happen to use the same
  scale) so the whole ``Add`` -- not just the convs feeding it -- can run and
  fuse in INT8.

onnxsim's own job here is narrower than any of that: ``simplify()`` must
never *change* a QDQ graph's meaning while cleaning it up, so a model already
following these conventions before simplification must still follow them
after. These tests build small graphs by hand (no NVIDIA ModelOpt dependency
-- unlike ``test_modelopt_integration.py``, these always run, in exchange for
being convention-shaped rather than literal real-world output) that each
isolate one of the conventions above, and check that ``simplify()`` leaves
the QDQ structure, per-channel/per-tensor granularity, symmetric zero points,
and independent branch quantization all intact -- and that the simplified
model still executes and agrees with the pre-simplified one.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import TensorProto, parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=17, ir_version=8):
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


def _tensor(array, name, dtype=np.float32):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=dtype), name)


def _op_domains(model, op_type):
    return [n.domain for n in model.graph.node if n.op_type == op_type]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def test_simplify_preserves_per_channel_weight_qdq_for_conv():
    # TensorRT quantizes Conv/Deconv weights per output channel (axis=0),
    # while the activation input stays per-tensor (a scalar scale) -- the
    # documented, and only well-supported, granularity split.
    rng = np.random.default_rng(0)
    cout, cin = 6, 3
    w = rng.standard_normal((cout, cin, 3, 3)).astype(np.float32) * 0.3
    w_scale = np.abs(w).reshape(cout, -1).max(axis=1) / 127.0
    w_scale = np.maximum(w_scale, 1e-6).astype(np.float32)
    w_zp = np.zeros(cout, dtype=np.int8)

    model = _model(
        f"""
        g (float[1,{cin},8,8] X) => (float[1,{cout},8,8] Y)
        <float a_scale = {{0.05}}, int8 a_zp = {{0}}>
        {{
          Xq = QuantizeLinear(X, a_scale, a_zp)
          Xdq = DequantizeLinear(Xq, a_scale, a_zp)
          Wq = QuantizeLinear<axis = 0>(W, w_scale, w_zp)
          Wdq = DequantizeLinear<axis = 0>(Wq, w_scale, w_zp)
          Y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(Xdq, Wdq)
        }}
        """,
        [
            _tensor(w, "W"),
            _tensor(w_scale, "w_scale"),
            _tensor(w_zp, "w_zp", dtype=np.int8),
        ],
    )

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # Structure must survive: still one QDQ pair on the activation, one on
    # the weight, both in the default domain TensorRT's parser expects.
    assert _op_domains(sim_model, "QuantizeLinear") == [""] * 2
    assert _op_domains(sim_model, "DequantizeLinear") == [""] * 2

    weight_dq = next(
        n
        for n in sim_model.graph.node
        if n.op_type == "DequantizeLinear" and n.input[0] not in ("Xq",)
    )
    dq_axis = next((a.i for a in weight_dq.attribute if a.name == "axis"), 1)
    assert dq_axis == 0  # per-output-channel, not collapsed to per-tensor

    scale_init = next(
        t for t in sim_model.graph.initializer if t.name == weight_dq.input[1]
    )
    assert list(scale_init.dims) == [cout]  # one scale per output channel

    x = rng.standard_normal((1, cin, 8, 8)).astype(np.float32)
    (before,) = _run(model, {"X": x})
    (after,) = _run(sim_model, {"X": x})
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)


def test_simplify_preserves_symmetric_int8_zero_point():
    # TensorRT's explicit INT8 path is symmetric only -- both weight and
    # activation zero points must stay exactly 0 (int8), not be widened,
    # dropped, or altered into something with a nonzero offset.
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 4)).astype(np.float32) * 0.2

    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        <float a_scale = {0.03}, int8 a_zp = {0},
         float w_scale = {0.01}, int8 w_zp = {0}>
        {
          Xq = QuantizeLinear(X, a_scale, a_zp)
          Xdq = DequantizeLinear(Xq, a_scale, a_zp)
          Wq = QuantizeLinear(W, w_scale, w_zp)
          Wdq = DequantizeLinear(Wq, w_scale, w_zp)
          Y = MatMul(Xdq, Wdq)
        }
        """,
        [_tensor(w, "W")],
    )

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    for dq in (n for n in sim_model.graph.node if n.op_type == "DequantizeLinear"):
        zp_init = next(t for t in sim_model.graph.initializer if t.name == dq.input[2])
        assert zp_init.data_type == TensorProto.INT8
        zp = onnx.numpy_helper.to_array(zp_init)
        assert np.all(zp == 0)


def test_simplify_preserves_independent_residual_branch_qdq():
    # A residual/shortcut Add: each branch gets its own QDQ pair (per
    # TensorRT's recommendation) so the whole Add can run and fuse in INT8,
    # even when -- as here -- both branches happen to share the same scale
    # value. onnxsim must not treat the two DequantizeLinear nodes as
    # redundant and merge/eliminate either one: they consume different data
    # (different conv outputs), so collapsing them would silently change
    # which branch feeds the Add.
    rng = np.random.default_rng(2)
    cout = 4
    w1 = rng.standard_normal((cout, cout, 3, 3)).astype(np.float32) * 0.1
    w2 = rng.standard_normal((cout, cout, 3, 3)).astype(np.float32) * 0.1

    model = _model(
        f"""
        g (float[1,{cout},8,8] X) => (float[1,{cout},8,8] Y)
        <float branch_scale = {{0.02}}, int8 branch_zp = {{0}}>
        {{
          conv1 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W1)
          q1 = QuantizeLinear(conv1, branch_scale, branch_zp)
          B1 = DequantizeLinear(q1, branch_scale, branch_zp)
          conv2 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W2)
          q2 = QuantizeLinear(conv2, branch_scale, branch_zp)
          B2 = DequantizeLinear(q2, branch_scale, branch_zp)
          Y = Add(B1, B2)
        }}
        """,
        [_tensor(w1, "W1"), _tensor(w2, "W2")],
    )

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # Both branches' QDQ pairs must survive as distinct nodes, and the Add
    # must still consume two *different* dequantized tensors.
    assert _op_domains(sim_model, "QuantizeLinear") == [""] * 2
    assert _op_domains(sim_model, "DequantizeLinear") == [""] * 2
    (add_node,) = [n for n in sim_model.graph.node if n.op_type == "Add"]
    assert add_node.input[0] != add_node.input[1]

    x = rng.standard_normal((1, cout, 8, 8)).astype(np.float32)
    (before,) = _run(model, {"X": x})
    (after,) = _run(sim_model, {"X": x})
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)


def test_simplify_keeps_qdq_nodes_in_default_domain():
    # TensorRT's ONNX parser only recognizes QuantizeLinear/DequantizeLinear
    # in the default ("") domain -- simplify() must never move, wrap, or
    # otherwise re-domain them (e.g. while registering/optimizing alongside
    # onnxsim's own com.microsoft-domain contrib-op passes).
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 4)).astype(np.float32) * 0.2
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        <float a_scale = {0.03}, int8 a_zp = {0},
         float w_scale = {0.01}, int8 w_zp = {0}>
        {
          Xq = QuantizeLinear(X, a_scale, a_zp)
          Xdq = DequantizeLinear(Xq, a_scale, a_zp)
          Wq = QuantizeLinear(W, w_scale, w_zp)
          Wdq = DequantizeLinear(Wq, w_scale, w_zp)
          Y = MatMul(Xdq, Wdq)
        }
        """,
        [_tensor(w, "W")],
    )
    assert _op_domains(model, "QuantizeLinear") == [""] * 2
    assert _op_domains(model, "DequantizeLinear") == [""] * 2

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _op_domains(sim_model, "QuantizeLinear") == [""] * 2
    assert _op_domains(sim_model, "DequantizeLinear") == [""] * 2
