#!/usr/bin/env python3
"""Make the backbone take the image already quantized to uint8 (the graph's own input scale/zp),
optionally in NHWC, so the HTP skips its fp32 input quantize (`image_q`) and, for NHWC, may skip
its NCHW->NHWC input transpose too.

  image(fp32 CHW) -> Unsqueeze -> Q(s=1.0776, zp=114) -> DQ -> stem conv      (as shipped)
  nchw: image_u8 [1,3,H,W] ------------------------------> DQ -> stem conv
  nhwc: image_u8 [1,H,W,3] -> Transpose(0,3,1,2) -> DQ -> stem conv

The preprocessing that produces the fp32 image can emit round(x/s)+zp just as cheaply, so this is
lossless end to end (the graph's first op was exactly that quantization).

usage: quantized_input.py <in.onnx> <out.onnx> nchw|nhwc
"""
import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    src, dst, layout = sys.argv[1:4]
    m = onnx.load(src)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    (unsq,) = cons[g.input[0].name]
    assert unsq.op_type == "Unsqueeze"
    (q,) = cons[unsq.output[0]]
    assert q.op_type == "QuantizeLinear"
    (dq,) = cons[q.output[0]]
    c, h, w = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    scale, zp = float(init[q.input[1]]), int(init[q.input[2]])
    if layout == "nchw":
        new_in = helper.make_tensor_value_info("image_u8", TensorProto.UINT8, [1, c, h, w])
        dq.input[0] = "image_u8"
    else:
        new_in = helper.make_tensor_value_info("image_u8", TensorProto.UINT8, [1, h, w, c])
        t = helper.make_node("Transpose", ["image_u8"], ["image_u8_nchw"], perm=[0, 3, 1, 2],
                             name="image_u8_to_nchw")
        g.node.insert(0, t)
        dq.input[0] = "image_u8_nchw"
    g.node.remove(unsq)
    g.node.remove(q)
    del g.input[:]
    g.input.append(new_in)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    Path(dst).with_name("input_qparams.json").write_text(
        json.dumps({"scale": scale, "zero_point": zp, "layout": layout}))
    print(f"input image_u8 {layout} scale={scale} zp={zp}")


if __name__ == "__main__":
    main()
