#!/usr/bin/env python3
"""A small, self-contained suite of ONNX graphs for execution-provider checks.

Shared by the vendor execution-provider compatibility harnesses (Qualcomm QNN,
Apple CoreML, Intel OpenVINO, ...) under ``scripts/``. These are synthetic
graphs, deliberately tiny, chosen to (a) cover common layer patterns a target
backend would see and (b) include the redundancy that onnxsim is meant to
remove -- foldable constant subgraphs, shape/gather chains feeding a Reshape,
no-op transposes -- so the checks actually exercise "did simplification
change the on-device path" rather than just "does a clean graph convert".

Kept network-free on purpose: the fast CI jobs need no downloads. Heavier
scheduled jobs can layer real onnxmodelzoo models on top later.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_OPSET = 18
_IR = 10


def _model(nodes, inputs, outputs, initializers, name) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], ir_version=_IR
    )
    onnx.checker.check_model(model)
    return model


def _const(name: str, array: np.ndarray):
    return numpy_helper.from_array(array, name)


def _rand(*shape, seed=0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def conv_bn_relu() -> onnx.ModelProto:
    """Conv -> (scale/shift as Mul/Add, foldable) -> Relu -> GlobalAveragePool."""
    w = _const("w", _rand(8, 3, 3, 3, seed=1))
    b = _const("b", np.zeros(8, np.float32))
    scale = _const("scale", _rand(1, 8, 1, 1, seed=2))
    shift = _const("shift", _rand(1, 8, 1, 1, seed=3))
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Mul", ["c", "scale"], ["m"]),
        helper.make_node("Add", ["m", "shift"], ["a"]),
        helper.make_node("Relu", ["a"], ["r"]),
        helper.make_node("GlobalAveragePool", ["r"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 1, 1])],
        [w, b, scale, shift],
        "conv_bn_relu",
    )


def foldable_shape_reshape() -> onnx.ModelProto:
    """Shape -> Gather -> Concat -> Reshape: the classic chain onnxsim folds.

    The reshape target is fully determined by constants, so onnxsim collapses
    the whole shape-computation subgraph into a single constant. A good check
    that the folded result still converts and matches.
    """
    w = _const("w", _rand(8, 3, 3, 3, seed=1))
    idx = _const("idx", np.array([0], np.int64))
    minus1 = _const("m1", np.array([-1], np.int64))
    ch = _const("ch", np.array([8], np.int64))
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Shape", ["r"], ["shp"]),
        helper.make_node("Gather", ["shp", "idx"], ["n"], axis=0),
        helper.make_node("Concat", ["n", "ch", "m1"], ["newshape"], axis=0),
        helper.make_node("Reshape", ["r", "newshape"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 64])],
        [w, idx, minus1, ch],
        "foldable_shape_reshape",
    )


def matmul_bias_tanh() -> onnx.ModelProto:
    """A small MLP block: MatMul -> Add(bias) -> Tanh.

    The bias is expressed as ``bias0 + bias1`` so onnxsim folds it to a single
    constant; the backend result must be unchanged by that fold.
    """
    wt = _const("wt", _rand(32, 64, seed=4))
    bias0 = _const("bias0", _rand(64, seed=5))
    bias1 = _const("bias1", _rand(64, seed=6))
    nodes = [
        helper.make_node("Add", ["bias0", "bias1"], ["bias"]),
        helper.make_node("MatMul", ["x", "wt"], ["mm"]),
        helper.make_node("Add", ["mm", "bias"], ["h"]),
        helper.make_node("Tanh", ["h"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 64])],
        [wt, bias0, bias1],
        "matmul_bias_tanh",
    )


def redundant_transpose() -> onnx.ModelProto:
    """Two transposes that cancel, plus an Add with a foldable constant.

    onnxsim removes the no-op transpose pair and folds ``k0 + k1``; the
    backend result should be unchanged.
    """
    k0 = _const("k0", _rand(1, 4, 8, 8, seed=6))
    k1 = _const("k1", _rand(1, 4, 8, 8, seed=7))
    nodes = [
        helper.make_node("Add", ["k0", "k1"], ["k"]),
        helper.make_node("Add", ["x", "k"], ["s"]),
        helper.make_node("Transpose", ["s"], ["t1"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["t1"], ["t2"], perm=[0, 3, 1, 2]),
        helper.make_node("Relu", ["t2"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [k0, k1],
        "redundant_transpose",
    )


def sigmoid_mul_swish() -> onnx.ModelProto:
    """Conv -> Sigmoid -> Mul (Swish/SiLU), a common mobile activation."""
    w = _const("w", _rand(6, 3, 1, 1, seed=8))
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"]),
        helper.make_node("Sigmoid", ["c"], ["s"]),
        helper.make_node("Mul", ["c", "s"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 12, 12])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 12, 12])],
        [w],
        "sigmoid_mul_swish",
    )


# Ordered so the registry is stable across runs.
_BUILDERS: List[Tuple[str, Callable[[], onnx.ModelProto]]] = [
    ("conv_bn_relu", conv_bn_relu),
    ("foldable_shape_reshape", foldable_shape_reshape),
    ("matmul_bias_tanh", matmul_bias_tanh),
    ("redundant_transpose", redundant_transpose),
    ("sigmoid_mul_swish", sigmoid_mul_swish),
]


def all_models() -> Dict[str, onnx.ModelProto]:
    """Build the whole suite: {name: ModelProto}."""
    return {name: build() for name, build in _BUILDERS}


def names() -> List[str]:
    return [name for name, _ in _BUILDERS]


def build(name: str) -> onnx.ModelProto:
    for n, builder in _BUILDERS:
        if n == name:
            return builder()
    raise KeyError(name)


if __name__ == "__main__":
    for n, m in all_models().items():
        print(f"{n:24} {len(m.graph.node)} nodes")
