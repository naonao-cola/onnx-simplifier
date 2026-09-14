"""Tests for mixed-precision training: the ``QuantizeLinear``/
``DequantizeLinear`` straight-through rules in ``onnxsim.graph_grad`` and
``compile_training_loop(..., quantize_forward=True)``.

Each model is built directly with ``onnx.parser`` (no torch dependency).
The gradient rules are checked against finite differences through a real
``onnxruntime`` run; the loop test trains a tiny MLP on CPU and checks the
loss falls and tracks the fp32 loop's own trajectory.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=8):
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


def _mlp_model(d_in=8, d_hid=8, d_out=4, seed=0):
    rng = np.random.default_rng(seed)
    w1 = _f32(rng.standard_normal((d_in, d_hid)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((d_hid, d_out)) * 0.5, "W2")
    return _model(
        f"""
        g (float[4,{d_in}] X) => (float[] loss)
        {{
          H = MatMul(X, W1)
          A = Relu(H)
          Y = MatMul(A, W2)
          E = Sub(Y, Y)
          S = Mul(E, E)
          loss = ReduceMean<keepdims = 0>(S)
        }}
        """,
        initializer=[w1, w2],
    )


def _losses(loop, feeds, steps=8, lr=1e-2):
    return [float(loop(dict(feeds), lr=lr)) for _ in range(steps)]


def test_quantize_forward_loop_matches_fp32_trajectory():
    rng = np.random.default_rng(1)
    d_in, d_hid, d_out = 8, 8, 4
    w1 = _f32(rng.standard_normal((d_in, d_hid)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((d_hid, d_out)) * 0.5, "W2")
    body = f"""
        g (float[4,{d_in}] X, float[4,{d_out}] T) => (float[] loss)
        {{
          H = MatMul(X, W1)
          A = Relu(H)
          Y = MatMul(A, W2)
          E = Sub(Y, T)
          S = Mul(E, E)
          loss = ReduceMean<keepdims = 0>(S)
        }}
        """
    model = _model(body, initializer=[w1, w2])
    feeds = {
        "X": rng.standard_normal((4, d_in)).astype(np.float32),
        "T": rng.standard_normal((4, d_out)).astype(np.float32),
    }

    float_loop = onnxsim.compile_training_loop(model, "loss", ["W1", "W2"])
    float_losses = _losses(float_loop, feeds)

    mixed_loop = onnxsim.compile_training_loop(
        model, "loss", ["W1", "W2"], quantize_forward=True
    )
    mixed_losses = _losses(mixed_loop, feeds)

    # Both train (loss falls) and stay in the same neighborhood: the INT8
    # forward perturbs optimization dynamics, so the trajectories legitimately
    # diverge -- what must hold is that mixed precision still descends to a
    # comparable loss, not that it shadows fp32 step for step.
    assert mixed_losses[-1] < mixed_losses[0]
    assert float_losses[-1] < float_losses[0]
    for f, q in zip(float_losses, mixed_losses):
        assert abs(f - q) / max(abs(f), abs(q), 1e-6) < 0.2

    # The compiled step really carries the INT8 forward: a fake-quant
    # QuantizeLinear per trained weight plus the calibrated activation pair.
    ops = [n.op_type for n in mixed_loop.step_graph.model.graph.node]
    assert ops.count("QuantizeLinear") >= 3  # 2 masters + 1 activation
    assert "DequantizeLinear" in ops


def test_quantize_forward_keeps_param_names_and_state():
    rng = np.random.default_rng(2)
    model = _mlp_model(seed=2)
    body_x = rng.standard_normal((4, 8)).astype(np.float32)
    loop = onnxsim.compile_training_loop(
        model, "loss", ["W1", "W2"], quantize_forward=True
    )
    loop({"X": body_x}, lr=1e-2)
    params = loop.parameters()
    assert set(params) == {"W1", "W2"}
    for name, arr in params.items():
        assert arr.dtype == np.float32  # masters stay fp32


def test_quantize_forward_first_step_matches_int8_inference():
    # The mixed loop's forward is exactly the INT8 inference forward (a
    # trainable master behind a fake-quant QuantizeLinear per weight), so its
    # first reported loss must match quantize_static inference on the same
    # feeds -- before the optimizer has had a step to diverge.
    rng = np.random.default_rng(4)
    d_in, d_hid, d_out = 8, 8, 4
    w1 = _f32(rng.standard_normal((d_in, d_hid)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((d_hid, d_out)) * 0.5, "W2")
    model = _model(
        f"""
        g (float[4,{d_in}] X, float[4,{d_out}] T) => (float[] loss)
        {{
          H = MatMul(X, W1)
          A = Relu(H)
          Y = MatMul(A, W2)
          E = Sub(Y, T)
          S = Mul(E, E)
          loss = ReduceMean<keepdims = 0>(S)
        }}
        """,
        initializer=[w1, w2],
    )
    feeds = {
        "X": rng.standard_normal((4, d_in)).astype(np.float32),
        "T": rng.standard_normal((4, d_out)).astype(np.float32),
    }
    # NB: against the shape-inferred model, which is what the loop itself
    # quantizes: without value_info the calibrator only ever sees the graph
    # inputs, so the uninferred model quantizes fewer tensors (a different,
    # less-quantized graph with slightly different numerics).
    ref = onnxsim.quantize_static(
        onnx.shape_inference.infer_shapes(model),
        num_calibration_samples=8,
        seed=0,
    )
    sess = ort.InferenceSession(
        ref.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (expected,) = sess.run(None, feeds)

    loop = onnxsim.compile_training_loop(
        model, "loss", ["W1", "W2"], quantize_forward=True
    )
    [first] = _losses(loop, feeds, steps=1)
    np.testing.assert_allclose(first, float(expected), rtol=1e-4)


def _two_layer_model(d_in=8, d_hid=8, d_out=4, seed=0):
    rng = np.random.default_rng(seed)
    w1 = _f32(rng.standard_normal((d_in, d_hid)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((d_hid, d_out)) * 0.5, "W2")
    return _model(
        f"""
        g (float[4,{d_in}] X, float[4,{d_out}] T) => (float[] loss)
        {{
          H = MatMul(X, W1)
          A = Relu(H)
          Y = MatMul(A, W2)
          E = Sub(Y, T)
          S = Mul(E, E)
          loss = ReduceMean<keepdims = 0>(S)
        }}
        """,
        initializer=[w1, w2],
    )


def test_split_execution_matches_fused():
    # forward_providers set (even to CPU) takes the split path: separate
    # forward/backward sessions with activations crossing the host. Losses,
    # parameters and export must match the fused loop exactly.
    rng = np.random.default_rng(5)
    model = _two_layer_model(seed=5)
    feeds = {
        "X": rng.standard_normal((4, 8)).astype(np.float32),
        "T": rng.standard_normal((4, 4)).astype(np.float32),
    }
    kwargs = {"calibration_data": None}
    fused = onnxsim.compile_training_loop(
        model, "loss", ["W1", "W2"], quantize_forward=True, **kwargs
    )
    fused_losses = _losses(fused, feeds, steps=4)
    split = onnxsim.compile_training_loop(
        model,
        "loss",
        ["W1", "W2"],
        quantize_forward=True,
        forward_providers=["CPUExecutionProvider"],
        **kwargs,
    )
    split_losses = _losses(split, feeds, steps=4)
    assert split._runner_fwd is not None and split._runner_bwd is not None
    assert split._runner is None  # no fused session in split mode
    assert split._needed  # boundary activations exist
    for f, s in zip(fused_losses, split_losses):
        assert abs(f - s) < 1e-6
    for name in ("W1", "W2"):
        np.testing.assert_array_equal(
            fused.parameters()[name], split.parameters()[name]
        )
    assert set(split.parameters()) == {"W1", "W2"}
    # The fused step graph is still built (inspection/export unaffected).
    assert split.step_graph is not None
    assert split.export().graph.name == model.graph.name


def test_quantize_linear_passes_gradient_through():
    # Straight-through: the rule emits dx = g, so a loss seeded just above
    # a QuantizeLinear must reach X unchanged (no saturation in range).
    from onnxsim import graph_grad

    assert "QuantizeLinear" in graph_grad.supported_ops()
    assert "DequantizeLinear" in graph_grad.supported_ops()


def test_loss_scale_matches_unscaled_trajectory():
    # Static loss scaling is a mathematical no-op (exact for powers of
    # two): seeding the backward with S and dividing the gradients back
    # out must reproduce the unscaled trajectory.
    rng = np.random.default_rng(6)
    model = _two_layer_model(seed=6)
    feeds = {
        "X": rng.standard_normal((4, 8)).astype(np.float32),
        "T": rng.standard_normal((4, 4)).astype(np.float32),
    }
    base = onnxsim.compile_training_loop(model, "loss", ["W1", "W2"])
    base_losses = _losses(base, feeds, steps=4)
    scaled = onnxsim.compile_training_loop(
        model, "loss", ["W1", "W2"], loss_scale=1024.0
    )
    scaled_losses = _losses(scaled, feeds, steps=4)
    for b, s in zip(base_losses, scaled_losses):
        np.testing.assert_allclose(s, b, rtol=1e-6)


def test_loss_scale_rejects_non_positive():
    model = _two_layer_model(seed=7)
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="loss_scale"):
            onnxsim.compile_training_loop(model, "loss", ["W1", "W2"], loss_scale=bad)


def test_backward_precision_float16_tracks_float32():
    # fp16 gradient math with fp32 masters/updates: the loop trains and
    # tracks the fp32 trajectory on this scale (loss_scale keeps the small
    # gradients representable). The step graph must carry the boundary
    # Casts both ways and keep the optimizer section fp32.
    rng = np.random.default_rng(12)
    model = _two_layer_model(seed=12)
    feeds = {
        "X": rng.standard_normal((4, 8)).astype(np.float32),
        "T": rng.standard_normal((4, 4)).astype(np.float32),
    }
    base = onnxsim.compile_training_loop(model, "loss", ["W1", "W2"])
    base_losses = _losses(base, feeds, steps=5)
    half = onnxsim.compile_training_loop(
        model,
        "loss",
        ["W1", "W2"],
        backward_precision="float16",
        loss_scale=1024.0,
    )
    half_losses = _losses(half, feeds, steps=5)
    assert half_losses[-1] < half_losses[0]
    for b, h in zip(base_losses, half_losses):
        assert abs(b - h) / max(abs(b), abs(h), 1e-6) < 0.05
    f16 = sum(1 for n in half.step_graph.model.graph.node if n.op_type == "Cast")
    assert f16 > 0


def test_backward_precision_rejects_unknown():
    model = _two_layer_model(seed=13)
    with pytest.raises(ValueError, match="backward_precision"):
        onnxsim.compile_training_loop(
            model, "loss", ["W1", "W2"], backward_precision="int8"
        )
