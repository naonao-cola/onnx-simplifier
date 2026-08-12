"""End-to-end coverage for the optimizer passes that onnxsim owns.

These four passes -- ``fuse_mul_into_conv``, ``fuse_consecutive_mul``,
``fuse_matmul_add_bias_into_gemm_batched`` and
``eliminate_reshape_around_elementwise`` -- are onnxsim-specific graph rewrites
that live under ``onnxsim/passes/`` and are injected into onnxoptimizer's
registry at runtime (see ``onnxsim/custom_optimizer_passes.*``). They used to be
tested in the onnxoptimizer fork in isolation; here they are exercised through
the full ``onnxsim.simplify`` pipeline, which also guards each rewrite with
onnxsim's own random-input equivalence check (``check_n``).
"""

import collections

import numpy as np
import onnx

import onnxsim


def _simplify(model):
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    # Pin a low IR version so the model loads under the older onnxruntime
    # bundled with some CI wheels; onnxsim's check_n runs the model through it.
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _conv_out_feeds_mul(model):
    conv_outs = {o for n in model.graph.node if n.op_type == "Conv" for o in n.output}
    return any(
        n.op_type == "Mul" and any(i in conv_outs for i in n.input)
        for n in model.graph.node
    )


# --------------------------------------------------------------------------- #
# fuse_mul_into_conv
# --------------------------------------------------------------------------- #
def test_fuse_mul_into_conv_per_channel():
    # Conv -> Mul(per-output-channel [1, C, 1, 1] scale): the scale folds into
    # the Conv weights, so nothing multiplies the Conv output afterwards.
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    s = _f32(np.random.rand(1, 4, 1, 1), "S")
    nodes = [
        onnx.helper.make_node("Conv", ["X", "W"], ["Z"], pads=[1, 1, 1, 1]),
        onnx.helper.make_node("Mul", ["Z", "S"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", (1, 3, 8, 8))], [_vi("Y", (1, 4, 8, 8))], [w, s])
    sim, ops = _simplify(model)
    assert ops["Conv"] == 1
    assert not _conv_out_feeds_mul(sim)


def test_fuse_mul_into_conv_scalar():
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    s = _f32(np.array(2.0), "S")  # scalar scale
    nodes = [
        onnx.helper.make_node("Conv", ["X", "W"], ["Z"], pads=[1, 1, 1, 1]),
        onnx.helper.make_node("Mul", ["Z", "S"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", (1, 3, 8, 8))], [_vi("Y", (1, 4, 8, 8))], [w, s])
    sim, _ = _simplify(model)
    assert not _conv_out_feeds_mul(sim)


# --------------------------------------------------------------------------- #
# fuse_preceding_mul_into_conv
# --------------------------------------------------------------------------- #
def _mul_feeds_conv_in(model):
    conv_first_inputs = {n.input[0] for n in model.graph.node if n.op_type == "Conv"}
    return any(
        n.op_type == "Mul" and n.output[0] in conv_first_inputs
        for n in model.graph.node
    )


def test_fuse_preceding_mul_into_conv_per_channel():
    # Mul(X, per-input-channel [1, C, 1, 1] scale) -> Conv: the scale folds
    # into the Conv weights, so nothing multiplies the Conv input beforehand.
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    b = _f32(np.random.rand(4), "B")
    s = _f32(np.random.rand(1, 3, 1, 1), "S")
    nodes = [
        onnx.helper.make_node("Mul", ["X", "S"], ["X2"]),
        onnx.helper.make_node("Conv", ["X2", "W", "B"], ["Y"], pads=[1, 1, 1, 1]),
    ]
    model = _model(nodes, [_vi("X", (1, 3, 8, 8))], [_vi("Y", (1, 4, 8, 8))], [w, b, s])
    sim, ops = _simplify(model)
    assert ops["Conv"] == 1
    assert not _mul_feeds_conv_in(sim)


def test_fuse_preceding_mul_into_conv_scalar():
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    s = _f32(np.array(2.0), "S")  # scalar scale
    nodes = [
        onnx.helper.make_node("Mul", ["X", "S"], ["X2"]),
        onnx.helper.make_node("Conv", ["X2", "W"], ["Y"], pads=[1, 1, 1, 1]),
    ]
    model = _model(nodes, [_vi("X", (1, 3, 8, 8))], [_vi("Y", (1, 4, 8, 8))], [w, s])
    sim, _ = _simplify(model)
    assert not _mul_feeds_conv_in(sim)


def test_fuse_preceding_mul_into_conv_grouped_per_channel_not_fused():
    # A per-channel scale on a grouped Conv is left alone: it would need
    # re-slicing per group to line up with the weight layout, which this pass
    # does not attempt. The model must still simplify correctly.
    w = _f32(np.random.rand(4, 1, 3, 3), "W")
    s = _f32(np.random.rand(1, 4, 1, 1), "S")
    nodes = [
        onnx.helper.make_node("Mul", ["X", "S"], ["X2"]),
        onnx.helper.make_node("Conv", ["X2", "W"], ["Y"], pads=[1, 1, 1, 1], group=4),
    ]
    model = _model(nodes, [_vi("X", (1, 4, 8, 8))], [_vi("Y", (1, 4, 8, 8))], [w, s])
    sim, ops = _simplify(model)
    assert ops["Conv"] == 1


# --------------------------------------------------------------------------- #
# fuse_consecutive_mul
# --------------------------------------------------------------------------- #
def test_fuse_consecutive_mul_scalar():
    # Mul(X, C1) -> Mul(., C2) with constant C1/C2 collapses to a single Mul by
    # a fused C1*C2 constant (X is a runtime input, so it cannot be folded away).
    c1 = _f32(np.array(2.0), "C1")
    c2 = _f32(np.array(3.0), "C2")
    nodes = [
        onnx.helper.make_node("Mul", ["X", "C1"], ["Y"]),
        onnx.helper.make_node("Mul", ["Y", "C2"], ["Z"]),
    ]
    model = _model(nodes, [_vi("X", (2, 3))], [_vi("Z", (2, 3))], [c1, c2])
    sim, ops = _simplify(model)
    assert ops["Mul"] == 1


def test_fuse_consecutive_mul_per_channel():
    # Per-channel (C, 1, 1) scale composed with a scalar factor (LayerScale).
    c1 = _f32(np.arange(4).reshape(4, 1, 1) + 1.0, "C1")
    c2 = _f32(np.array(0.5), "C2")
    nodes = [
        onnx.helper.make_node("Mul", ["X", "C1"], ["Y"]),
        onnx.helper.make_node("Mul", ["Y", "C2"], ["Z"]),
    ]
    model = _model(nodes, [_vi("X", (1, 4, 2, 2))], [_vi("Z", (1, 4, 2, 2))], [c1, c2])
    sim, ops = _simplify(model)
    assert ops["Mul"] == 1


# --------------------------------------------------------------------------- #
# fuse_matmul_add_bias_into_gemm_batched
# --------------------------------------------------------------------------- #
def test_fuse_matmul_add_bias_into_gemm_batched():
    # A rank-3 linear layer MatMul(X[2,3,4], W[4,5]) + b[5] that the 2-D-only
    # fuse_matmul_add_bias_into_gemm cannot fuse becomes a Gemm (with reshape
    # scaffolding), so no batched MatMul remains.
    w = _f32(np.random.randn(4, 5), "W")
    b = _f32(np.random.randn(5), "B")
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["Z"]),
        onnx.helper.make_node("Add", ["Z", "B"], ["A"]),
    ]
    model = _model(nodes, [_vi("X", (2, 3, 4))], [_vi("A", (2, 3, 5))], [w, b])
    sim, ops = _simplify(model)
    assert ops["Gemm"] >= 1
    assert "MatMul" not in ops


# --------------------------------------------------------------------------- #
# eliminate_reshape_around_elementwise (cancels the batched-Gemm reshape
# scaffolding between two linear layers separated by an element-wise op)
# --------------------------------------------------------------------------- #
def test_eliminate_reshape_around_elementwise():
    # Two rank-3 linear layers with a Relu in between. Each linear becomes a
    # Gemm wrapped in Reshape(2-D)/Reshape(N-D); the inverse reshapes around the
    # Relu cancel, leaving the Gemms without the intermediate reshape pair.
    w0 = _f32(np.random.randn(4, 6), "W0")
    b0 = _f32(np.random.randn(6), "B0")
    w1 = _f32(np.random.randn(6, 5), "W1")
    b1 = _f32(np.random.randn(5), "B1")
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W0"], ["Z0"]),
        onnx.helper.make_node("Add", ["Z0", "B0"], ["A0"]),
        onnx.helper.make_node("Relu", ["A0"], ["R"]),
        onnx.helper.make_node("MatMul", ["R", "W1"], ["Z1"]),
        onnx.helper.make_node("Add", ["Z1", "B1"], ["A1"]),
    ]
    model = _model(
        nodes, [_vi("X", (2, 3, 4))], [_vi("A1", (2, 3, 5))], [w0, b0, w1, b1]
    )
    sim, ops = _simplify(model)
    # Both linear layers dispatch as Gemms and the batched MatMuls are gone.
    assert ops["Gemm"] >= 2
    assert "MatMul" not in ops
    # The reshape scaffolding around the Relu is cancelled: at most the two
    # outer reshapes survive (never the 4 a naive per-layer rewrite would add).
    assert ops["Reshape"] <= 2
