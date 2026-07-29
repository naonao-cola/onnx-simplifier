#!/usr/bin/env python3
"""Generate the tiny ONNX model + reference IO used by the inference test.

The model is deliberately small but does real compute (a MatMul, a bias Add and
a Relu) with baked-in constant weights, so running it must produce the same
numbers on every iteration and on every execution provider. Re-run this to
regenerate ``model.onnx`` and ``io.json`` after changing the graph.

    python3 make_fixture.py
"""

import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = pathlib.Path(__file__).parent

# Fixed weights -> deterministic reference outputs. Chosen so the pre-Relu
# result has both positive and negative entries, exercising the Relu clamp.
W = np.array(
    [
        [0.5, -1.0, 0.25],
        [-0.5, 0.5, 1.0],
        [1.0, 0.0, -0.75],
        [0.25, 0.5, -0.5],
    ],
    dtype=np.float32,
)
B = np.array([0.1, -0.2, 0.3], dtype=np.float32)
X = np.array([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)

REF = np.maximum(X @ W + B, 0.0)


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "B"], ["xwb"]),
        helper.make_node("Relu", ["xwb"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "inference_fixture",
        [x],
        [y],
        initializer=[
            numpy_helper.from_array(W, name="W"),
            numpy_helper.from_array(B, name="B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9  # onnxruntime-web 1.27 supports up to IR 10; 9 is safe.
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_model()
    onnx.save(model, HERE / "model.onnx")
    io = {
        "input": {"name": "X", "dims": list(X.shape), "data": X.flatten().tolist()},
        "output": {"name": "Y", "dims": list(REF.shape), "data": REF.flatten().tolist()},
    }
    (HERE / "io.json").write_text(json.dumps(io, indent=2) + "\n")
    print("wrote model.onnx and io.json; reference Y =", REF.flatten().tolist())


if __name__ == "__main__":
    main()
