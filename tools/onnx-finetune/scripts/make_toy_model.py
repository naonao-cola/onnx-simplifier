#!/usr/bin/env python3
"""Write a tiny 2-layer MLP regressor to demo/test onnx-finetune against."""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

p = argparse.ArgumentParser()
p.add_argument("-o", "--output", default="toy_model.onnx")
p.add_argument("--input-dim", type=int, default=4)
p.add_argument("--hidden-dim", type=int, default=16)
p.add_argument("--output-dim", type=int, default=1)
args = p.parse_args()

rng = np.random.default_rng(0)


def linear(name, in_dim, out_dim):
    w = (rng.standard_normal((in_dim, out_dim)) * 0.1).astype(np.float32)
    b = np.zeros(out_dim, dtype=np.float32)
    return numpy_helper.from_array(w, f"{name}.weight"), numpy_helper.from_array(b, f"{name}.bias")


w1, b1 = linear("fc1", args.input_dim, args.hidden_dim)
w2, b2 = linear("fc2", args.hidden_dim, args.output_dim)

nodes = [
    helper.make_node("MatMul", ["input", "fc1.weight"], ["mm1"]),
    helper.make_node("Add", ["mm1", "fc1.bias"], ["add1"]),
    helper.make_node("Relu", ["add1"], ["relu1"]),
    helper.make_node("MatMul", ["relu1", "fc2.weight"], ["mm2"]),
    helper.make_node("Add", ["mm2", "fc2.bias"], ["output"]),
]

graph = helper.make_graph(
    nodes,
    "toy_mlp",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, args.input_dim])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, args.output_dim])],
    initializer=[w1, b1, w2, b2],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, args.output)
print(f"wrote {args.output} (input_dim={args.input_dim} hidden_dim={args.hidden_dim} output_dim={args.output_dim})")
