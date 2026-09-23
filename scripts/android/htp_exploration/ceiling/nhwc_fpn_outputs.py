#!/usr/bin/env python3
"""Emit the four FPN maps channels-last (NHWC) instead of NCHW.

The HTP computes in NHWC internally; an NCHW graph output costs it a final transpose of every FPN
map (the largest, P2, is 256x200x272). This project's fast RoiAlign DSP kernel wants NHWC anyway
(see ../../tinygrad_hexagon_bridge/README.md, RoiAlign Stage 3), so emitting NHWC removes that
transpose on both sides. Lossless (a pure layout change).

usage: nhwc_fpn_outputs.py <in.onnx> <out.onnx>
"""
import sys

import onnx
from onnx import helper


def main():
    m = onnx.load(sys.argv[1])
    g = m.graph
    outs = []
    for o in g.output:
        n, c, h, w = [d.dim_value for d in o.type.tensor_type.shape.dim]
        if c != 256:
            outs.append(o)
            continue
        t = o.name + "_nhwc"
        g.node.append(helper.make_node("Transpose", [o.name], [t], perm=[0, 2, 3, 1], name=t))
        outs.append(helper.make_tensor_value_info(t, o.type.tensor_type.elem_type, [n, h, w, c]))
    del g.output[:]
    g.output.extend(outs)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])


if __name__ == "__main__":
    main()
