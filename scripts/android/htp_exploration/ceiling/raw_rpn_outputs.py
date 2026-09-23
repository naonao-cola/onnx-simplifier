#!/usr/bin/env python3
"""Expose the RPN head convs' raw outputs instead of the per-anchor Reshape/Transpose chains.

Each of the 10 RPN outputs (box deltas + objectness for P2-P6) is its conv's NCHW output pushed
through Reshape -> Transpose -> Reshape (-> Sigmoid for scores), with a Q/DQ pair between every
step, to reach per-anchor layout [1, n_anchors(, 4)]. On the HTP that pure data movement costs
~11% of accelerator time (P2 alone is 163200 anchors). This rewrite makes the graph output the
conv result directly -- uint8 for the delta convs (they end in a QuantizeLinear), float logits for
the tiny score convs (they feed Reshape with no Q) -- and prunes the dead chains. The consumer then
owns layout + sigmoid; this project's fused RPN DSP kernel already reads raw uint8 deltas.

usage: raw_rpn_outputs.py <in.onnx> <out.onnx>
"""
import collections
import sys

import onnx


def main():
    m = onnx.load(sys.argv[1])
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    vi = {v.name: v for v in list(g.value_info)}
    inferred = onnx.shape_inference.infer_shapes(m).graph
    shapes = {v.name: v for v in list(inferred.value_info) + list(inferred.output)}
    new_outputs = []
    for o in g.output:
        t, hops = o.name, 0
        while prod[t].op_type != "Conv":
            t = prod[t].input[0]
            hops += 1
        if hops <= 1:  # FPN maps: Conv -> Q -> output, already raw
            new_outputs.append(o)
            continue
        conv_out = t
        after = cons[conv_out]
        if len(after) == 1 and after[0].op_type == "QuantizeLinear":
            t = after[0].output[0]  # delta head: uint8 conv output
        new_outputs.append(shapes[t] if t in shapes else vi[t])
    del g.output[:]
    g.output.extend(new_outputs)
    live = {o.name for o in g.output}
    changed = True
    while changed:  # prune nodes whose outputs are all dead
        changed = False
        used = {i for n in g.node for i in n.input} | live
        for n in list(g.node):
            if not any(o in used for o in n.output):
                g.node.remove(n)
                changed = True
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim],
                        onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)) for o in g.output])


if __name__ == "__main__":
    main()
