"""Pass-isolated fusion / elimination tests.

OnnxSlim ships a suite of small, single-pattern tests (``test_fusion_patterns``,
``test_dead_node_elimination``, ``test_elimination_patterns``,
``test_subexpression_elimination``) that each check one optimization in
isolation. onnxsim performs the equivalent optimizations through onnxoptimizer
and its constant folding, but only exercised them indirectly through full
torchvision / timm models. This module adds the missing isolated coverage.

Every model is built directly with ``onnx.helper`` (no torch dependency) and run
through ``onnxsim.simplify`` with ``check_n=3`` so onnxsim's own random-input
equivalence check guards correctness of each rewrite.
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
    # bundled with some CI wheels (which cap at IR version 11); onnxsim's
    # check_n runs the model through onnxruntime.
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10)


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


# --------------------------------------------------------------------------- #
# Fusion patterns
# --------------------------------------------------------------------------- #
def test_fuse_conv_bn_into_conv():
    # Conv followed by BatchNormalization folds the BN affine transform into the
    # Conv weights/bias (fuse_bn_into_conv), leaving a single Conv.
    inits = [
        _f32(np.random.randn(8, 3, 3, 3), "W"),
        _f32(np.random.rand(8) + 0.5, "scale"),
        _f32(np.random.randn(8), "bias"),
        _f32(np.random.randn(8), "mean"),
        _f32(np.random.rand(8) + 0.5, "var"),
    ]
    nodes = [
        onnx.helper.make_node(
            "Conv", ["X", "W"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        onnx.helper.make_node(
            "BatchNormalization", ["c", "scale", "bias", "mean", "var"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [1, 3, 16, 16])], [_vi("Y", [1, 8, 16, 16])], inits)
    _, ops = _simplify(model)
    assert ops["BatchNormalization"] == 0
    assert ops["Conv"] == 1


def test_fuse_matmul_add_into_gemm():
    # MatMul followed by a bias Add on 2-D inputs fuses into a single Gemm.
    inits = [_f32(np.random.randn(16, 8), "W"), _f32(np.random.randn(8), "B")]
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["mm"]),
        onnx.helper.make_node("Add", ["mm", "B"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 16])], [_vi("Y", [4, 8])], inits)
    _, ops = _simplify(model)
    assert ops["Gemm"] == 1
    assert ops["MatMul"] == 0 and ops["Add"] == 0


def test_fuse_pad_into_conv():
    # A constant zero-value Pad on the spatial dims is folded into the Conv pads
    # attribute (fuse_pad_into_conv), removing the Pad node.
    inits = [
        _i64([0, 0, 1, 1, 0, 0, 1, 1], "pads"),
        _f32(np.random.randn(8, 3, 3, 3), "W"),
    ]
    nodes = [
        onnx.helper.make_node("Pad", ["X", "pads"], ["p"], mode="constant"),
        onnx.helper.make_node("Conv", ["p", "W"], ["Y"], kernel_shape=[3, 3]),
    ]
    model = _model(nodes, [_vi("X", [1, 3, 16, 16])], [_vi("Y", [1, 8, 16, 16])], inits)
    _, ops = _simplify(model)
    assert ops["Pad"] == 0
    assert ops["Conv"] == 1


def test_fuse_consecutive_reduce_unsqueeze():
    # ReduceSum(keepdims=0) immediately followed by an Unsqueeze on the reduced
    # axis collapses into a single keepdims reduction.
    inits = [_i64([2], "raxes"), _i64([2], "uaxes")]
    nodes = [
        onnx.helper.make_node("ReduceSum", ["X", "raxes"], ["r"], keepdims=0),
        onnx.helper.make_node("Unsqueeze", ["r", "uaxes"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [2, 3, 4])], [_vi("Y", [2, 3, 1])], inits)
    _, ops = _simplify(model)
    assert ops["Unsqueeze"] == 0
    assert ops["ReduceSum"] == 1


def test_fuse_concat_into_reshape():
    # A Concat of constant shape pieces feeding a Reshape is folded into a single
    # Reshape with a constant target shape (fuse_concat_into_reshape).
    inits = [_i64([2], "c0"), _i64([-1], "c1")]
    nodes = [
        onnx.helper.make_node("Concat", ["c0", "c1"], ["shape"], axis=0),
        onnx.helper.make_node("Reshape", ["X", "shape"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [2, 3, 4])], [_vi("Y", [2, 12])], inits)
    _, ops = _simplify(model)
    assert ops["Concat"] == 0
    assert ops["Reshape"] == 1


# --------------------------------------------------------------------------- #
# Dead-node / no-op elimination
# --------------------------------------------------------------------------- #
def test_eliminate_identity():
    nodes = [
        onnx.helper.make_node("Identity", ["X"], ["a"]),
        onnx.helper.make_node("Relu", ["a"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])
    _, ops = _simplify(model)
    assert ops["Identity"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_transpose():
    # A Transpose whose permutation is the identity ordering is a no-op.
    nodes = [
        onnx.helper.make_node("Relu", ["X"], ["a"]),
        onnx.helper.make_node("Transpose", ["a"], ["Y"], perm=[0, 1]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])
    _, ops = _simplify(model)
    assert ops["Transpose"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_expand():
    # Expand to the already-existing shape does nothing and is removed.
    inits = [_i64([4, 8], "eshape")]
    nodes = [
        onnx.helper.make_node("Relu", ["X"], ["a"]),
        onnx.helper.make_node("Expand", ["a", "eshape"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], inits)
    _, ops = _simplify(model)
    assert ops["Expand"] == 0
    assert ops["Relu"] == 1


def test_eliminate_mul_by_one():
    # Multiplying by a unit constant is a no-op and is eliminated.
    inits = [_f32(np.array([1.0]), "one")]
    nodes = [
        onnx.helper.make_node("Relu", ["X"], ["a"]),
        onnx.helper.make_node("Mul", ["a", "one"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], inits)
    _, ops = _simplify(model)
    assert ops["Mul"] == 0
    assert ops["Relu"] == 1


def test_eliminate_consecutive_cancelling_transposes():
    # Two transposes that invert each other collapse away entirely.
    nodes = [
        onnx.helper.make_node("Transpose", ["X"], ["t"], perm=[1, 0]),
        onnx.helper.make_node("Transpose", ["t"], ["Y"], perm=[1, 0]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])
    sim_model, ops = _simplify(model)
    # The pair either cancels to a bare passthrough (<=1 node) with no residual
    # transpose logic changing the data.
    assert ops["Transpose"] <= 1


# --------------------------------------------------------------------------- #
# Common subexpression elimination
# --------------------------------------------------------------------------- #
def test_eliminate_common_subexpression():
    # Two structurally identical Sqrt nodes over the same input are deduplicated
    # into one shared node (eliminate_common_subexpression).
    nodes = [
        onnx.helper.make_node("Sqrt", ["X"], ["s1"]),
        onnx.helper.make_node("Sqrt", ["X"], ["s2"]),
        onnx.helper.make_node("Add", ["s1", "s2"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])
    _, ops = _simplify(model)
    assert ops["Sqrt"] == 1
    assert ops["Add"] == 1
