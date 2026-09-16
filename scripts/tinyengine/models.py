#!/usr/bin/env python3
"""TinyEngine-side model suite: the shared suite plus TinyEngine-specific fixtures.

Re-exports `scripts/common/synthetic_models.py` (see that module) and adds:

- `tinyengine_dynamic_batch_leaf`: a symbolic-batch-dimension Conv->Relu leaf,
  to exercise `tinyengine_ops.has_dynamic_shape` -- a source-code generator
  needs every shape fixed ahead of time, a constraint none of the shared
  suite's fixed-shape models trip.
- `mobilenet_dw_block`: an inverted-residual bottleneck (the same block shape
  as `scripts/edgeai/models.py`'s `mobilenet_block`, built independently here
  since each vendor directory keeps its own fixtures per this repo's
  convention). Deliberately demonstrates a *positive* interaction with
  onnxsim: the block's BN Mul/Add pairs sit between each Conv and its Clip
  (Relu6) activation, so before simplification the Clip is **not** eligible
  for TinyEngine's Conv-activation fusion (it isn't the conv's sole,
  immediate consumer -- see `tinyengine_ops.is_fusable_activation`) and this
  heuristic reports a blocker. Once onnxsim folds each BN pair into its
  preceding Conv, the Clip becomes the conv's direct, sole consumer and the
  blocker clears -- onnxsim's own fold is what makes the block
  TinyEngine-eligible in the first place, not just a harmless cleanup.
- `se_gate_block`: a depthwise-conv squeeze-and-excite gate --
  `GlobalAveragePool -> Conv(fc1) -> Relu -> Conv(fc2) -> Add(+bias) ->
  Mul(*scale) -> Mul(onto the depthwise branch)` -- built to land the
  trailing `Mul` inside the exact `Add -> Mul -> Mul` window
  `tinyengine_ops.se_window_mul_names` (and TinyEngine's own real
  `checkIfRequireSEelementmult`) recognize as a fused SE gate, the *only*
  way a `Mul` is ever supported by this dispatch.
- `standalone_mul_leaf`: the common Sigmoid-gated Swish/SiLU activation
  (`Conv -> Sigmoid -> Mul`, i.e.
  `scripts/common/synthetic_models.sigmoid_mul_swish`'s own shape) is
  already in the shared suite and already demonstrates this; this fixture
  isolates just the `Mul` half against a plain elementwise scale (`Conv ->
  Mul` by another Conv's output, no `Add`/`Sigmoid` anywhere) to show the
  same rejection happens even without an activation confusing the picture:
  `Mul` outside the `Add -> Mul -> Mul` window is never supported, full stop.
- `prelu_leaf`: `Conv -> PRelu`, the deliberate "has a real TFLite opcode,
  still unsupported" case -- see `tinyengine_ops.UNSUPPORTED_WITH_REAL_TFLITE_OPCODE`'s
  docstring for why this is worth its own fixture rather than folding into
  the generic "unrecognized op" case.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# See scripts/edgeai/models.py's docstring (which cites scripts/axera/models.py's
# own docstring) for why scripts/ is only kept on sys.path for the duration of
# this import.
_inserted = _SCRIPTS_DIR not in sys.path
if _inserted:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from common.synthetic_models import (  # noqa: E402,F401
        all_models as _shared_all_models,
        build as _shared_build,
        conv_bn_relu,
        foldable_shape_reshape,
        matmul_bias_tanh,
        names as _shared_names,
        redundant_transpose,
        sigmoid_mul_swish,
    )
finally:
    if _inserted:
        sys.path.remove(_SCRIPTS_DIR)


def _rand(*shape, seed=0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


_DYNAMIC_BATCH_LEAF_NAME = "tinyengine_dynamic_batch_leaf"


def tinyengine_dynamic_batch_leaf() -> onnx.ModelProto:
    """Conv -> Relu with a symbolic ("N") batch dimension on the input.

    Not run through the shared suite's fixed-shape helper -- see
    `scripts/edgeai/models.py`'s `edgeai_dynamic_batch_leaf` for the same
    pattern.
    """
    w = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 3, 3, 3], [0.0] * (4 * 3 * 3 * 3)
    )
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"], name="relu"),
    ]
    x_type = helper.make_tensor_value_info("x", TensorProto.FLOAT, None)
    x_type.type.tensor_type.shape.dim.add().dim_param = "N"
    for d in (3, 8, 8):
        x_type.type.tensor_type.shape.dim.add().dim_value = d
    y_type = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    y_type.type.tensor_type.shape.dim.add().dim_param = "N"
    for d in (4, 8, 8):
        y_type.type.tensor_type.shape.dim.add().dim_value = d
    graph = helper.make_graph(nodes, _DYNAMIC_BATCH_LEAF_NAME, [x_type], [y_type], [w])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


_MOBILENET_DW_BLOCK_NAME = "mobilenet_dw_block"


def mobilenet_dw_block() -> onnx.ModelProto:
    """Expand/depthwise/project inverted-residual bottleneck, BN unfolded.

    See this module's docstring: before `onnxsim.simplify()`, each Clip
    (Relu6) sits after a BN Mul/Add pair rather than directly after its
    Conv, so `tinyengine_ops.is_fusable_activation` -- and TinyEngine's real
    Conv-activation fusion -- doesn't apply. Folding the BN into the Conv is
    what clears the blocker, not just a node-count cleanup.
    """
    c_in, c_mid = 16, 32
    expand_w = numpy_helper.from_array(_rand(c_mid, c_in, 1, 1, seed=10), "expand_w")
    expand_scale = numpy_helper.from_array(
        _rand(1, c_mid, 1, 1, seed=11), "expand_scale"
    )
    expand_shift = numpy_helper.from_array(
        _rand(1, c_mid, 1, 1, seed=12), "expand_shift"
    )
    dw_w = numpy_helper.from_array(_rand(c_mid, 1, 3, 3, seed=13), "dw_w")
    dw_scale = numpy_helper.from_array(_rand(1, c_mid, 1, 1, seed=14), "dw_scale")
    dw_shift = numpy_helper.from_array(_rand(1, c_mid, 1, 1, seed=15), "dw_shift")
    project_w = numpy_helper.from_array(_rand(c_in, c_mid, 1, 1, seed=16), "project_w")
    project_scale = numpy_helper.from_array(
        _rand(1, c_in, 1, 1, seed=17), "project_scale"
    )
    project_shift = numpy_helper.from_array(
        _rand(1, c_in, 1, 1, seed=18), "project_shift"
    )
    zero = numpy_helper.from_array(np.array(0.0, np.float32), "zero")
    six = numpy_helper.from_array(np.array(6.0, np.float32), "six")

    nodes = [
        helper.make_node("Conv", ["x", "expand_w"], ["e"], name="expand_conv"),
        helper.make_node("Mul", ["e", "expand_scale"], ["e_bn1"], name="expand_bn_mul"),
        helper.make_node(
            "Add", ["e_bn1", "expand_shift"], ["e_bn"], name="expand_bn_add"
        ),
        helper.make_node("Clip", ["e_bn", "zero", "six"], ["e_act"], name="expand_act"),
        helper.make_node(
            "Conv",
            ["e_act", "dw_w"],
            ["d"],
            name="depthwise_conv",
            pads=[1, 1, 1, 1],
            group=c_mid,
        ),
        helper.make_node("Mul", ["d", "dw_scale"], ["d_bn1"], name="dw_bn_mul"),
        helper.make_node("Add", ["d_bn1", "dw_shift"], ["d_bn"], name="dw_bn_add"),
        helper.make_node("Clip", ["d_bn", "zero", "six"], ["d_act"], name="dw_act"),
        helper.make_node("Conv", ["d_act", "project_w"], ["p"], name="project_conv"),
        helper.make_node(
            "Mul", ["p", "project_scale"], ["p_bn1"], name="project_bn_mul"
        ),
        helper.make_node(
            "Add", ["p_bn1", "project_shift"], ["p_bn"], name="project_bn_add"
        ),
        helper.make_node("Add", ["x", "p_bn"], ["y"], name="residual_add"),
    ]
    graph = helper.make_graph(
        nodes,
        _MOBILENET_DW_BLOCK_NAME,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c_in, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c_in, 8, 8])],
        [
            expand_w,
            expand_scale,
            expand_shift,
            dw_w,
            dw_scale,
            dw_shift,
            project_w,
            project_scale,
            project_shift,
            zero,
            six,
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


_SE_GATE_BLOCK_NAME = "se_gate_block"


def se_gate_block() -> onnx.ModelProto:
    """A depthwise-conv squeeze-and-excite gate, landing on the exact
    `Add -> Mul -> Mul` window TinyEngine's real `checkIfRequireSEelementmult`
    (and this module's `se_window_mul_names`) recognizes.

    `DepthwiseConv -> GlobalAveragePool -> Conv(fc1, 1x1) -> Relu ->
    Conv(fc2, 1x1) -> Add(+bias) -> Mul(*scale) -> Mul(onto the depthwise
    branch)`. The trailing `Add -> Mul -> Mul` is deliberately just that
    three-op shape -- not a claim that this is numerically a specific
    activation function (e.g. a hard-sigmoid); TinyEngine's real
    `checkIfRequireSEelementmult` lookahead is keyed on the op-type
    sequence alone (see `tinyengine_ops.py`'s docstring), so this fixture
    matches that, without asserting more than was actually read in the
    source.
    """
    c = 8
    dw_w = numpy_helper.from_array(_rand(c, 1, 3, 3, seed=20), "se_dw_w")
    fc1_w = numpy_helper.from_array(_rand(c, c, 1, 1, seed=21), "se_fc1_w")
    fc2_w = numpy_helper.from_array(_rand(c, c, 1, 1, seed=22), "se_fc2_w")
    bias = numpy_helper.from_array(_rand(1, c, 1, 1, seed=27), "se_bias")
    scale = numpy_helper.from_array(_rand(1, c, 1, 1, seed=28), "se_scale")

    nodes = [
        helper.make_node(
            "Conv",
            ["x", "se_dw_w"],
            ["dw"],
            name="se_dwconv",
            pads=[1, 1, 1, 1],
            group=c,
        ),
        helper.make_node("GlobalAveragePool", ["dw"], ["pooled"], name="se_pool"),
        helper.make_node("Conv", ["pooled", "se_fc1_w"], ["fc1"], name="se_fc1"),
        helper.make_node("Relu", ["fc1"], ["fc1_act"], name="se_fc1_relu"),
        helper.make_node("Conv", ["fc1_act", "se_fc2_w"], ["fc2"], name="se_fc2"),
        helper.make_node("Add", ["fc2", "se_bias"], ["gate_add"], name="se_gate_add"),
        helper.make_node("Mul", ["gate_add", "se_scale"], ["gate"], name="se_gate_mul"),
        helper.make_node("Mul", ["dw", "gate"], ["y"], name="se_apply_mul"),
    ]
    graph = helper.make_graph(
        nodes,
        _SE_GATE_BLOCK_NAME,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c, 8, 8])],
        [dw_w, fc1_w, fc2_w, bias, scale],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


_STANDALONE_MUL_LEAF_NAME = "standalone_mul_leaf"


def standalone_mul_leaf() -> onnx.ModelProto:
    """Two Conv branches combined with a plain elementwise `Mul` -- no
    `Add` anywhere upstream, so `se_window_mul_names` (and
    TinyEngine's real SE lookahead) never fires. Demonstrates the sharp
    constraint holds even without an activation (Sigmoid, etc.) in the
    picture: `Mul` outside that exact fusion window is simply unsupported.
    """
    wa = numpy_helper.from_array(_rand(4, 3, 1, 1, seed=23), "mul_wa")
    wb = numpy_helper.from_array(_rand(4, 3, 1, 1, seed=24), "mul_wb")
    nodes = [
        helper.make_node("Conv", ["x", "mul_wa"], ["a"], name="mul_conv_a"),
        helper.make_node("Conv", ["x", "mul_wb"], ["b"], name="mul_conv_b"),
        helper.make_node("Mul", ["a", "b"], ["y"], name="standalone_mul"),
    ]
    graph = helper.make_graph(
        nodes,
        _STANDALONE_MUL_LEAF_NAME,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [wa, wb],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


_PRELU_LEAF_NAME = "prelu_leaf"


def prelu_leaf() -> onnx.ModelProto:
    """`Conv -> PRelu` -- `PRelu` has a real TFLite `BuiltinOperator` opcode
    (`code_generator/tflite/BuiltinOperator.py`'s `PRELU = 54`) but no case
    in `TfliteConvertor._handleOperator`'s dispatch; see
    `tinyengine_ops.UNSUPPORTED_WITH_REAL_TFLITE_OPCODE`'s docstring.
    """
    w = numpy_helper.from_array(_rand(4, 3, 1, 1, seed=25), "prelu_w")
    slope = numpy_helper.from_array(_rand(4, 1, 1, seed=26), "prelu_slope")
    nodes = [
        helper.make_node("Conv", ["x", "prelu_w"], ["c"], name="prelu_conv"),
        helper.make_node("PRelu", ["c", "prelu_slope"], ["y"], name="prelu_act"),
    ]
    graph = helper.make_graph(
        nodes,
        _PRELU_LEAF_NAME,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [w, slope],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


_LOCAL_BUILDERS = {
    _DYNAMIC_BATCH_LEAF_NAME: tinyengine_dynamic_batch_leaf,
    _MOBILENET_DW_BLOCK_NAME: mobilenet_dw_block,
    _SE_GATE_BLOCK_NAME: se_gate_block,
    _STANDALONE_MUL_LEAF_NAME: standalone_mul_leaf,
    _PRELU_LEAF_NAME: prelu_leaf,
}

# `se_gate_block` is deliberately excluded from `all_models()`/`names()`'s
# default suite (though still reachable via `build("se_gate_block")`, and
# used directly by tests/test_tinyengine_compat.py's own regression test).
# It is not a "this should simplify cleanly" fixture like the rest of this
# suite: onnxsim's own affine-fold collapses its Add(bias)/Mul(scale) pair
# straight into the preceding Conv's weight and bias (the same fold that
# clears mobilenet_dw_block's blockers), which here destroys the exact
# Add->Mul->Mul node-type signature `tinyengine_ops.se_window_mul_names`
# (and TinyEngine's real `checkIfRequireSEelementmult`) depends on --
# turning a graph with full coverage *before* simplification into one
# `new_blocking_op_types` correctly flags as regressed *after* it. Folding
# it into the default suite would make the "every suite model simplifies
# to status=='ok'" test in tests/test_tinyengine_compat.py fail by design;
# it is a real, worth-documenting finding, not a bug in this heuristic --
# see that test file and scripts/tinyengine/README.md for the full writeup.
_SUITE_EXCLUDED_NAMES = frozenset({_SE_GATE_BLOCK_NAME})


def all_models():
    models = _shared_all_models()
    for name, builder in _LOCAL_BUILDERS.items():
        if name not in _SUITE_EXCLUDED_NAMES:
            models[name] = builder()
    return models


def names():
    return [
        *_shared_names(),
        *(n for n in _LOCAL_BUILDERS if n not in _SUITE_EXCLUDED_NAMES),
    ]


def build(name: str) -> onnx.ModelProto:
    if name in _LOCAL_BUILDERS:
        return _LOCAL_BUILDERS[name]()
    return _shared_build(name)
