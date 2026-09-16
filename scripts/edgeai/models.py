#!/usr/bin/env python3
"""Edgeai-side model suite: the shared suite plus TIDL-specific fixtures.

Re-exports `scripts/common/synthetic_models.py` (see that module) and adds:

- `edgeai_dynamic_batch_leaf`: a graph with a symbolic batch dimension, to
  exercise `tidl_ops.has_dynamic_shape` -- TIDL requires every input shape to
  be fully static (see `tidl_ops.py`'s docstring), a constraint none of the
  shared suite's fixed-shape models trip.
- `mobilenet_block`: a small MobileNetV2-style inverted-residual bottleneck
  (expand/depthwise/project convs + a residual Add). MobileNetV2 is the
  backbone edgeai-tidl-tools' own real example configs use for object
  detection (`od-ort-ssd-lite_mobilenetv2_fpn`) and segmentation
  (`ss-ort-deeplabv3lite_mobilenetv2`, both in
  `runtimes/examples/python/basic_example/config.yaml`) -- verified
  directly, not "the quickstart model" (that's a plain ResNet18,
  `resnet18_opset9.onnx`, the first ONNX Runtime entry in that same file).
- `vision_transformer_block`: one pre-LN Vision Transformer encoder block
  (LayerNorm, MatMul-based attention, LayerNorm, an exact/erf-based-GELU
  MLP, both residual). Built from the fused `LayerNormalization` op (a
  directly supported layer per `docs/operators.md`) but the *decomposed*
  `Div`/`Erf`/`Add`/`Mul`/`Mul` GELU sequence, not the literal ONNX `Gelu`
  op -- `docs/operators.md` has no `Gelu` entry at all; see
  `tidl_ops.py`'s docstring and `legalize.py`'s `unfuse_gelu_to_erf` for
  why the decomposed form is what the real importer actually wants here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only keep scripts/ on sys.path for the duration of this import -- see
# scripts/axera/models.py's docstring for why (namespace-package shadowing
# for directories like scripts/rfdetr with no __init__.py).
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


_DYNAMIC_BATCH_LEAF_NAME = "edgeai_dynamic_batch_leaf"


def edgeai_dynamic_batch_leaf() -> onnx.ModelProto:
    """Conv -> Relu with a symbolic ("N") batch dimension on the input.

    Deliberately not run through the shared `_model` helper (which always
    builds fully-static shapes): the point of this fixture is the one
    dimension every other model in this suite lacks.
    """
    w = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 3, 3, 3], [0.0] * (4 * 3 * 3 * 3)
    )
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
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


_MOBILENET_BLOCK_NAME = "mobilenet_block"


def mobilenet_block() -> onnx.ModelProto:
    """MobileNetV2's inverted-residual bottleneck, at a small scale.

    Expand (1x1 Conv) -> BN -> ReLU6 -> depthwise (3x3 Conv, group=C) -> BN
    -> ReLU6 -> project (1x1 Conv) -> BN -> residual Add -- the block shape
    MobileNetV2 is built from (see this module's docstring for where it's
    verified to actually appear in edgeai-tidl-tools' own example configs).
    ReLU6 is expressed as `Clip(0, 6)`, matching the real network's
    activation rather than substituting a plain Relu. BN is expressed as
    the usual foldable Mul/Add pair (onnxsim fuses this into the preceding
    Conv), same convention as `scripts/common/synthetic_models.conv_bn_relu`.
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
        helper.make_node("Mul", ["e", "expand_scale"], ["e_bn1"]),
        helper.make_node("Add", ["e_bn1", "expand_shift"], ["e_bn"]),
        helper.make_node("Clip", ["e_bn", "zero", "six"], ["e_act"]),
        helper.make_node(
            "Conv",
            ["e_act", "dw_w"],
            ["d"],
            name="depthwise_conv",
            pads=[1, 1, 1, 1],
            group=c_mid,
        ),
        helper.make_node("Mul", ["d", "dw_scale"], ["d_bn1"]),
        helper.make_node("Add", ["d_bn1", "dw_shift"], ["d_bn"]),
        helper.make_node("Clip", ["d_bn", "zero", "six"], ["d_act"]),
        helper.make_node("Conv", ["d_act", "project_w"], ["p"], name="project_conv"),
        helper.make_node("Mul", ["p", "project_scale"], ["p_bn1"]),
        helper.make_node("Add", ["p_bn1", "project_shift"], ["p_bn"]),
        helper.make_node("Add", ["x", "p_bn"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        _MOBILENET_BLOCK_NAME,
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


_VIT_BLOCK_NAME = "vision_transformer_block"


def vision_transformer_block() -> onnx.ModelProto:
    """One pre-LN Vision Transformer encoder block.

    LayerNorm -> MatMul-based Q/K/V -> scaled dot-product attention (MatMul,
    Softmax, MatMul) -> residual Add -> LayerNorm -> MatMul/GELU/MatMul MLP
    -> residual Add -- the standard ViT encoder block. Uses the fused
    `LayerNormalization` op (a directly supported TIDL layer per
    `docs/operators.md`) but the *decomposed*, exact/erf-based GELU
    (`Div`/`Erf`/`Add`/`Mul`/`Mul`), matching what edgeai-tidl-tools'
    real importer recognizes -- not the literal ONNX `Gelu` op, which has
    no entry in that same doc (see `tidl_ops.py`'s docstring).
    """
    tokens, dim, hidden = 4, 8, 16
    scale = numpy_helper.from_array(
        np.array(1.0 / (dim**0.5), np.float32), "attn_scale"
    )
    ln1_scale = numpy_helper.from_array(_rand(dim, seed=30), "ln1_scale")
    ln1_bias = numpy_helper.from_array(_rand(dim, seed=31), "ln1_bias")
    ln2_scale = numpy_helper.from_array(_rand(dim, seed=32), "ln2_scale")
    ln2_bias = numpy_helper.from_array(_rand(dim, seed=33), "ln2_bias")
    wq = numpy_helper.from_array(_rand(dim, dim, seed=34), "wq")
    wk = numpy_helper.from_array(_rand(dim, dim, seed=35), "wk")
    wv = numpy_helper.from_array(_rand(dim, dim, seed=36), "wv")
    w1 = numpy_helper.from_array(_rand(dim, hidden, seed=37), "w1")
    w2 = numpy_helper.from_array(_rand(hidden, dim, seed=38), "w2")
    gelu_sqrt2 = numpy_helper.from_array(np.array(2.0**0.5, np.float32), "gelu_sqrt2")
    gelu_one = numpy_helper.from_array(np.array(1.0, np.float32), "gelu_one")
    gelu_half = numpy_helper.from_array(np.array(0.5, np.float32), "gelu_half")

    nodes = [
        helper.make_node("LayerNormalization", ["x", "ln1_scale", "ln1_bias"], ["ln1"]),
        helper.make_node("MatMul", ["ln1", "wq"], ["q"]),
        helper.make_node("MatMul", ["ln1", "wk"], ["k"]),
        helper.make_node("MatMul", ["ln1", "wv"], ["v"]),
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        helper.make_node("Mul", ["scores", "attn_scale"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["attn"], axis=-1),
        helper.make_node("MatMul", ["attn", "v"], ["ctx"]),
        helper.make_node("Add", ["x", "ctx"], ["res1"]),
        helper.make_node(
            "LayerNormalization", ["res1", "ln2_scale", "ln2_bias"], ["ln2"]
        ),
        helper.make_node("MatMul", ["ln2", "w1"], ["fc1"]),
        helper.make_node("Div", ["fc1", "gelu_sqrt2"], ["gelu_t0"]),
        helper.make_node("Erf", ["gelu_t0"], ["gelu_t1"]),
        helper.make_node("Add", ["gelu_t1", "gelu_one"], ["gelu_t2"]),
        helper.make_node("Mul", ["fc1", "gelu_t2"], ["gelu_t3"]),
        helper.make_node("Mul", ["gelu_t3", "gelu_half"], ["act"]),
        helper.make_node("MatMul", ["act", "w2"], ["fc2"]),
        helper.make_node("Add", ["res1", "fc2"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        _VIT_BLOCK_NAME,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, tokens, dim])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, tokens, dim])],
        [
            scale,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            wq,
            wk,
            wv,
            w1,
            w2,
            gelu_sqrt2,
            gelu_one,
            gelu_half,
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


_LOCAL_BUILDERS = {
    _DYNAMIC_BATCH_LEAF_NAME: edgeai_dynamic_batch_leaf,
    _MOBILENET_BLOCK_NAME: mobilenet_block,
    _VIT_BLOCK_NAME: vision_transformer_block,
}


def all_models():
    models = _shared_all_models()
    for name, builder in _LOCAL_BUILDERS.items():
        models[name] = builder()
    return models


def names():
    return [*_shared_names(), *_LOCAL_BUILDERS]


def build(name: str) -> onnx.ModelProto:
    if name in _LOCAL_BUILDERS:
        return _LOCAL_BUILDERS[name]()
    return _shared_build(name)
