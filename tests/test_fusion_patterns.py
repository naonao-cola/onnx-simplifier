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

ConvTranspose+BN, ConvTranspose+Add-bias and no-op ``Dropout`` fusions -- once
gaps versus OnnxSlim -- are now covered by onnxsim's optimizer (issue #543) and
are regular tests below. The final section holds the remaining ``xfail`` cases
(GELU-subgraph fusion) that OnnxSlim performs but onnxsim does not; they document
the gap and will XPASS if onnxsim ever gains the pass.
"""

import collections

import numpy as np
import onnx
import pytest

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
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


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
            "Conv", ["X", "W"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        onnx.helper.make_node(
            "BatchNormalization", ["c", "scale", "bias", "mean", "var"], ["Y"]
        ),
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


def test_batched_matmul_add_into_gemm():
    # A transformer linear layer applies a 2-D weight to a rank-3 activation and
    # adds a bias (exported bias-first, as HuggingFace does). onnxsim converts
    # the batched MatMul+Add into a Gemm (fuse_matmul_add_bias_into_gemm_batched,
    # opted in by onnxsim) so runtimes can dispatch tuned GEMM kernels.
    inits = [_f32(np.random.randn(8, 16), "W"), _f32(np.random.randn(16), "b")]
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["mm"]),
        onnx.helper.make_node("Add", ["b", "mm"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [2, 4, 8])], [_vi("Y", [2, 4, 16])], inits)
    _, ops = _simplify(model)
    assert ops["Gemm"] == 1
    assert ops["MatMul"] == 0


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


# --------------------------------------------------------------------------- #
# Deferred folding of large-tensor ops (ConstantOfShape / Constant -> Expand)
#
# With --no-large-tensor these ops are not folded into a (large) initializer,
# but they are still treated as constant so a downstream node whose output is
# small keeps folding: the producing op is inlined into the sub-model the
# operator executor runs, so the large intermediate is computed transiently and
# never stored. The result is that the whole constant chain collapses to the
# small final initializer.
# --------------------------------------------------------------------------- #
def _simplify_no_large_tensor(model):
    sim_model, check_ok = onnxsim.simplify(
        model, check_n=3, tensor_size_threshold="1KB"
    )
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def test_defer_constantofshape_folds_small_consumer():
    # ConstantOfShape produces a large (256*256 f32 = 256KB > 1KB) tensor, so it
    # is not materialized under --no-large-tensor. ReduceSum over it yields a
    # scalar, which the executor still folds by running ConstantOfShape+ReduceSum
    # together; the scalar feeds a runtime Add. The large tensor is never stored
    # and the ConstantOfShape/ReduceSum chain disappears.
    inits = [_i64([256, 256], "shape")]
    value = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT, [1], [2.0])
    nodes = [
        onnx.helper.make_node("ConstantOfShape", ["shape"], ["big"], value=value),
        onnx.helper.make_node("ReduceSum", ["big"], ["s"], keepdims=0),
        onnx.helper.make_node("Add", ["X", "s"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], inits, opset=11)
    sim_model, ops = _simplify_no_large_tensor(model)
    assert ops["ConstantOfShape"] == 0
    assert ops["ReduceSum"] == 0
    assert ops["Add"] == 1
    # No large intermediate tensor was materialized as an initializer.
    assert all(
        onnx.numpy_helper.to_array(init).size < 256 * 256
        for init in sim_model.graph.initializer
    )


def test_defer_constant_expand_folds_small_consumer():
    # Constant -> Expand blows a scalar up to a large (512*512) tensor. Under
    # --no-large-tensor the Expand is not materialized, but ReduceSum over it is
    # a small (scalar) foldable output whose value depends on the data (so it is
    # not handled by shape propagation). The executor still folds it by running
    # Expand+ReduceSum together, and the scalar feeds a runtime Add, so the whole
    # constant chain collapses without ever storing the expanded tensor.
    inits = [_i64([512, 512], "eshape")]
    scalar = onnx.helper.make_tensor("c", onnx.TensorProto.FLOAT, [1, 1], [3.0])
    nodes = [
        onnx.helper.make_node("Constant", [], ["small"], value=scalar),
        onnx.helper.make_node("Expand", ["small", "eshape"], ["big"]),
        onnx.helper.make_node("ReduceSum", ["big"], ["s"], keepdims=0),
        onnx.helper.make_node("Add", ["X", "s"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], inits, opset=11)
    sim_model, ops = _simplify_no_large_tensor(model)
    assert ops["Expand"] == 0
    assert ops["ReduceSum"] == 0
    assert ops["Constant"] == 0
    assert ops["Add"] == 1
    assert all(
        onnx.numpy_helper.to_array(init).size < 512 * 512
        for init in sim_model.graph.initializer
    )


# --------------------------------------------------------------------------- #
# ConvTranspose / no-op Dropout fusions (issue #543). Formerly xfail gaps versus
# OnnxSlim, now handled by onnxsim's optimizer: no-op Dropout removal and the
# fuse_bn_into_conv / fuse_add_bias_into_conv passes extended to ConvTranspose.
# --------------------------------------------------------------------------- #
def test_eliminate_nop_dropout():
    inits = [onnx.numpy_helper.from_array(np.array(0.0, dtype=np.float32), "ratio")]
    nodes = [
        onnx.helper.make_node("Relu", ["X"], ["a"]),
        onnx.helper.make_node("Dropout", ["a", "ratio"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], inits)
    _, ops = _simplify(model)
    assert ops["Dropout"] == 0
    assert ops["Relu"] == 1


def test_fuse_convtranspose_bn():
    inits = [
        _f32(np.random.randn(3, 8, 3, 3), "W"),  # ConvTranspose: [Cin, Cout, kH, kW]
        _f32(np.random.rand(8) + 0.5, "scale"),
        _f32(np.random.randn(8), "bias"),
        _f32(np.random.randn(8), "mean"),
        _f32(np.random.rand(8) + 0.5, "var"),
    ]
    nodes = [
        onnx.helper.make_node("ConvTranspose", ["X", "W"], ["c"], kernel_shape=[3, 3]),
        onnx.helper.make_node(
            "BatchNormalization", ["c", "scale", "bias", "mean", "var"], ["Y"]
        ),
    ]
    model = _model(nodes, [_vi("X", [1, 3, 8, 8])], [_vi("Y", [1, 8, 10, 10])], inits)
    _, ops = _simplify(model)
    assert ops["BatchNormalization"] == 0
    assert ops["ConvTranspose"] == 1


def test_fuse_convtranspose_add():
    inits = [
        _f32(np.random.randn(3, 8, 3, 3), "W"),
        _f32(np.random.randn(1, 8, 1, 1), "bias"),
    ]
    nodes = [
        onnx.helper.make_node("ConvTranspose", ["X", "W"], ["c"], kernel_shape=[3, 3]),
        onnx.helper.make_node("Add", ["c", "bias"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [1, 3, 8, 8])], [_vi("Y", [1, 8, 10, 10])], inits)
    _, ops = _simplify(model)
    assert ops["Add"] == 0
    assert ops["ConvTranspose"] == 1


# --------------------------------------------------------------------------- #
# Reshape cancellation around a batched-Gemm element-wise chain. onnxsim's
# batched MatMul+bias -> Gemm rewrite brackets each rank-3 linear layer in a
# Reshape(-> 2-D) / Gemm / Reshape(-> N-D) sandwich so runtimes can dispatch a
# tuned GEMM kernel. When two such linears surround an element-wise chain, the
# inverse reshapes between them cancel (eliminate_reshape_around_elementwise):
# the chain runs on the 2-D Gemm output and both Gemms are kept. This is the
# node-count half of the batched-Gemm rewrite (issue: model-regression node
# reduction).
# --------------------------------------------------------------------------- #
def test_eliminate_reshape_around_elementwise():
    # The post-batched-fusion shape, built explicitly: Gemm -> Reshape(N-D) ->
    # Relu -> Reshape(2-D) -> Gemm. The two middle reshapes are exact inverses
    # (they only split / merge the leading dims), so they collapse and the Relu
    # ends up directly between the two Gemms.
    inits = [
        _i64([-1, 8], "s1"),
        _f32(np.random.randn(8, 16), "W1"),
        _f32(np.random.randn(16), "b1"),
        _i64([2, 4, 16], "s2"),
        _i64([-1, 16], "s3"),
        _f32(np.random.randn(16, 8), "W2"),
        _f32(np.random.randn(8), "b2"),
        _i64([2, 4, 8], "s4"),
    ]
    nodes = [
        onnx.helper.make_node("Reshape", ["X", "s1"], ["rx1"]),
        onnx.helper.make_node("Gemm", ["rx1", "W1", "b1"], ["g1"]),
        onnx.helper.make_node("Reshape", ["g1", "s2"], ["u1"]),
        onnx.helper.make_node("Relu", ["u1"], ["r"]),
        onnx.helper.make_node("Reshape", ["r", "s3"], ["f2"]),
        onnx.helper.make_node("Gemm", ["f2", "W2", "b2"], ["g2"]),
        onnx.helper.make_node("Reshape", ["g2", "s4"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [2, 4, 8])], [_vi("Y", [2, 4, 8])], inits)
    _, ops = _simplify(model)
    # Both Gemms are kept (tuned-kernel dispatch preserved)...
    assert ops["Gemm"] == 2
    assert ops["Relu"] == 1
    # ...and the two inverse reshapes bracketing the Relu are gone: only the
    # entry (X -> 2-D) and the final (2-D -> N-D output) reshapes remain.
    assert ops["Reshape"] == 2


# --------------------------------------------------------------------------- #
# Remaining gap: passes OnnxSlim performs but onnxsim does not. Marked xfail
# (strict=False) so CI stays green; each XPASSes and can be promoted if onnxsim
# gains the pass.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="onnxsim has no GELU subgraph fusion; OnnxSlim ships a (currently "
    "disabled) FusionGelu matcher for this exact pattern",
    strict=False,
)
def test_fuse_gelu():
    # 0.5 * x * (1 + erf(x / sqrt(2))) is the exact-erf GELU formulation.
    inits = [
        _f32(np.array([0.5]), "half"),
        _f32(np.array([1.0]), "one"),
        _f32(np.array([1.4142135623730951]), "sqrt2"),
    ]
    nodes = [
        onnx.helper.make_node("Div", ["X", "sqrt2"], ["t0"]),
        onnx.helper.make_node("Erf", ["t0"], ["t1"]),
        onnx.helper.make_node("Add", ["t1", "one"], ["t2"]),
        onnx.helper.make_node("Mul", ["X", "t2"], ["t3"]),
        onnx.helper.make_node("Mul", ["t3", "half"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], inits)
    _, ops = _simplify(model)
    assert ops["Gelu"] == 1
    assert ops["Erf"] == 0
