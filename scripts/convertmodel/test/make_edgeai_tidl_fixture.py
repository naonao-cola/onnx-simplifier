#!/usr/bin/env python3
"""Generate the tiny ONNX fixtures edgeai_tidl_check.test.mjs checks against.

Four fixtures, matching the cases scripts/edgeai/tests/test_edgeai_tidl_compat.py
already covers on the Python side:

- clean.onnx: Conv -> Relu, no blockers, static shape.
- control_flow.onnx: a single `If` node -- a top-level blocker.
- dynamic_shape.onnx: Conv -> Relu with a symbolic ("N") batch dimension.
- qoperator.onnx: a single `QLinearConv` node -- the QOperator-format
  blocker (see scripts/edgeai/tidl_ops.py's QOPERATOR_OPS).

    python3 make_edgeai_tidl_fixture.py
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = pathlib.Path(__file__).parent


def _rand(*shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def clean_model() -> onnx.ModelProto:
    w = numpy_helper.from_array(_rand(4, 3, 3, 3, seed=1), "w")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "clean",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [w],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.checker.check_model(model)
    return model


def control_flow_model() -> onnx.ModelProto:
    then_g = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "then",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    else_g = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "else",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    cond = helper.make_tensor("cond", TensorProto.BOOL, [], [True])
    if_node = helper.make_node(
        "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
    )
    graph = helper.make_graph(
        [if_node],
        "with_if",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        [cond],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


def dynamic_shape_model() -> onnx.ModelProto:
    w = numpy_helper.from_array(_rand(4, 3, 3, 3, seed=2), "w")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    x_type = helper.make_tensor_value_info("x", TensorProto.FLOAT, None)
    x_type.type.tensor_type.shape.dim.add().dim_param = "N"
    for d in (3, 8, 8):
        x_type.type.tensor_type.shape.dim.add().dim_value = d
    y_type = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
    graph = helper.make_graph(nodes, "dynamic_shape", [x_type], [y_type], [w])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


def qoperator_model() -> onnx.ModelProto:
    node = helper.make_node(
        "QLinearConv",
        ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"],
        ["y"],
    )
    graph = helper.make_graph(
        [node],
        "qlinearconv_leaf",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 1, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    # Deliberately not onnx.checker.check_model'd: the initializers for
    # xs/xz/w/ws/wz/ys/yz are omitted since this fixture only needs a
    # QLinearConv node to exist for the op-type check, not a runnable graph.
    return model


if __name__ == "__main__":
    for name, builder in (
        ("clean.onnx", clean_model),
        ("control_flow.onnx", control_flow_model),
        ("dynamic_shape.onnx", dynamic_shape_model),
        ("qoperator.onnx", qoperator_model),
    ):
        onnx.save(builder(), str(HERE / name))
        print(f"wrote {name}")
