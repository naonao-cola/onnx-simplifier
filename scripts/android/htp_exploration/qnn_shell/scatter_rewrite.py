"""Rewrite rest.onnx's RoiAlign level merge from ScatterElements to ScatterND.

torchvision's MultiScaleRoIAlign (as exported) merges the per-FPN-level RoiAlign outputs with
    ScatterElements(data, Expand(Reshape(idx, [n,1,1,1]), [n,C,H,W]), level_out, axis=0)
i.e. an element-wise scatter whose index tensor is idx broadcast over every non-batch axis. That
is exactly a row scatter, ScatterND(data, Reshape(idx, [n,1]), level_out), which ORT's CPU kernel
does as n contiguous row copies instead of n*C*H*W indexed element writes. On the phone this
merge is the single most expensive part of rest.onnx (see ../rest_htp_findings.md).

usage: python scatter_rewrite.py <rest.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(src, dst):
    m = onnx.load(src)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    shape_name = "scatter_rewrite_row_index_shape"
    g.initializer.append(
        numpy_helper.from_array(np.array([-1, 1], np.int64), shape_name)
    )
    new_nodes, count = [], 0
    for n in g.node:
        if n.op_type == "ScatterElements":
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            exp = prod.get(n.input[1])
            src_idx = (
                prod.get(exp.input[0])
                if exp is not None and exp.op_type == "Expand"
                else None
            )
            if axis == 0 and src_idx is not None and src_idx.op_type == "Reshape":
                idx2 = f"{n.name}_row_index"
                new_nodes.append(
                    helper.make_node(
                        "Reshape",
                        [src_idx.input[0], shape_name],
                        [idx2],
                        name=f"{n.name}_row_index_reshape",
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "ScatterND",
                        [n.input[0], idx2, n.input[2]],
                        list(n.output),
                        name=f"{n.name}_as_scatternd",
                    )
                )
                count += 1
                continue
        new_nodes.append(n)
    # drop what the old index path (Expand of the broadcast shape) no longer feeds
    live = {o.name for o in g.output}
    kept = []
    for n in reversed(new_nodes):
        if any(o in live for o in n.output):
            kept.append(n)
            live.update(n.input)
    del g.node[:]
    g.node.extend(reversed(kept))
    inits = [i for i in g.initializer if i.name in live]
    del g.initializer[:]
    g.initializer.extend(inits)
    onnx.save(m, dst)
    print(f"removed {len(new_nodes) - len(kept)} dead nodes")
    print(f"rewrote {count} ScatterElements -> ScatterND")


if __name__ == "__main__":
    rewrite(*sys.argv[1:3])
