#!/usr/bin/env python3
"""Rewrite backbone.onnx so every ResNet residual Add is an int8 QDQ unit QNN can run natively.

As shipped (onnxmodelzoo MaskRCNN-12-qdq), each bottleneck ends `DQ(conv) + shortcut -> Add ->
Relu -> Q`, and in 12 of the 16 blocks the Relu output also feeds the *next* block's Add
directly in float. Neither the Add (output goes to Relu, not Q) nor the Relu (input is not a DQ)
forms a QDQ node unit, so QNN runs Add, Relu and the Q/DQ around them as float ops on
full-resolution tensors -- the single largest cost in the HTP profile.

Two rewrites, both per Relu:
  1. Drop the Relu and feed the Add straight into the Relu's QuantizeLinear. Exact: every such Q
     is uint8 with zero point 0, so Q already clamps negatives to 0 (Q(relu(x)) == Q(x)).
  2. Feed the next block's Add from DQ(that Q) instead of the float Relu output. This quantizes
     the shortcut to uint8 with the block output's own scale -- the only numeric change, and the
     same thing any int8 backend has to do (TVM's FakeQuantizationToInteger did it too).
After this every Add has DQ inputs and a Q output: a quantized Add on the HTP.

usage: quantize_residuals.py <in.onnx> <out.onnx>
"""
import collections
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    m = onnx.load(sys.argv[1])
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    remove, new_nodes, n_relu, n_short = [], [], 0, 0
    for r in list(g.node):
        if r.op_type != "Relu" or prod[r.input[0]].op_type != "Add":
            continue
        qs = [c for c in cons[r.output[0]] if c.op_type == "QuantizeLinear"]
        others = [c for c in cons[r.output[0]] if c.op_type != "QuantizeLinear"]
        assert len(qs) == 1, r.name
        q = qs[0]
        zp = init[q.input[2]]
        assert zp.dtype == np.uint8 and int(zp) == 0, (r.name, zp)
        q.input[0] = r.input[0]  # 1. Add -> Q directly; Q's zp=0 clamp is the Relu
        remove.append(r)
        n_relu += 1
        if others:  # 2. quantized shortcut for the next block's Add
            dq_out = r.output[0] + "_shortcut_dq"
            new_nodes.append(helper.make_node("DequantizeLinear", [q.output[0], q.input[1], q.input[2]],
                                              [dq_out], name=r.name + "_shortcut_dq"))
            for c in others:
                assert c.op_type == "Add", (r.name, c.op_type)
                for k, i in enumerate(c.input):
                    if i == r.output[0]:
                        c.input[k] = dq_out
                        n_short += 1
    for r in remove:
        g.node.remove(r)
    # insert each new DQ right after its producing Q to keep the graph topologically sorted
    for dq in new_nodes:
        qi = next(k for k, n in enumerate(g.node) if dq.input[0] in n.output)
        g.node.insert(qi + 1, dq)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    adds = [n for n in g.node if n.op_type == "Add"]
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    ok = sum(all(prod[i].op_type == "DequantizeLinear" for i in a.input) and
             [c.op_type for c in cons[a.output[0]]] == ["QuantizeLinear"] for a in adds)
    print(f"removed {n_relu} Relu, requantized {n_short} float shortcuts; "
          f"{ok}/{len(adds)} Adds are now DQ,DQ->Add->Q units")


if __name__ == "__main__":
    main()
