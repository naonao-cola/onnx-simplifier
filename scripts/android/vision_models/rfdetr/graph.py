"""ONNX-side passes for the exported RF-DETR graph (run with the repo's onnxsim environment):

- `simplify`: onnxsim.
- `fold_rank5_transposes`: the DINOv2 window partition / merge is Reshape(a,b,c,d,e) ->
  Transpose(0,2,1,3,4) -> Reshape. Merging the last two (untouched) axes gives the identical element
  order at rank 4: Reshape(a,b,c,d*e) -> Transpose(0,2,1,3) -> Reshape. QNN refuses the rank-5 form.
- `u8_input`: the graph input becomes uint8 NHWC RGB (the camera's bytes, resized to R x R):
  DequantizeLinear(scale 1, zp 0) -> Transpose -> the patch-embed Conv, with RF-DETR's /255 and
  ImageNet mean/std folded into that Conv's weight and bias. The patch embed has stride == kernel
  and no padding, so every input pixel is used and the fold is exact.

usage: graph.py <raw.onnx> <out.sim.onnx> <out.u8.onnx>
"""

from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def fold_rank5_transposes(m: onnx.ModelProto) -> int:
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    done = 0
    for t in g.node:
        if t.op_type != "Transpose":
            continue
        perm = list(helper.get_attribute_value(t.attribute[0]))
        if perm != [0, 2, 1, 3, 4]:
            continue
        r = prod.get(t.input[0])
        if r is None or r.op_type != "Reshape" or r.input[1] not in inits:
            continue
        if not all(c.op_type == "Reshape" for c in cons.get(t.output[0], [])):
            continue
        shape = numpy_helper.to_array(inits[r.input[1]]).astype(np.int64)
        if (
            len(shape) != 5
            or (shape == 0).any()
            or (shape < 0).sum() > 1
            or shape[3] < 0
        ):
            continue
        # a single -1 stays inferable: in a/b/c it is untouched, as e it becomes the merged d*e
        new = np.array(
            [*shape[:3], -1 if shape[4] < 0 else shape[3] * shape[4]], np.int64
        )
        name = r.input[1] + "_r4"
        if name not in inits:
            inits[name] = numpy_helper.from_array(new, name)
            g.initializer.append(inits[name])
        r.input[1] = name
        t.attribute[0].CopyFrom(helper.make_attribute("perm", [0, 2, 1, 3]))
        done += 1
    return done


def u8_input(m: onnx.ModelProto) -> onnx.ModelProto:
    m2 = onnx.ModelProto()
    m2.CopyFrom(m)
    g = m2.graph
    inp = g.input[0]
    _, c, h, w = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    users = [n for n in g.node if inp.name in n.input]
    assert len(users) == 1 and users[0].op_type == "Conv", [n.op_type for n in users]
    conv = users[0]
    attrs = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
    k = list(attrs["kernel_shape"])
    assert list(attrs.get("strides", [1, 1])) == k and not any(
        attrs.get("pads", [0, 0, 0, 0])
    ), attrs
    inits = {i.name: i for i in g.initializer}
    W = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float64)
    b = (
        numpy_helper.to_array(inits[conv.input[2]]).astype(np.float64)
        if len(conv.input) > 2
        else np.zeros(W.shape[0])
    )
    W2 = W / (255.0 * STD[None, :, None, None])
    b2 = b - (W * (MEAN / STD)[None, :, None, None]).sum((1, 2, 3))
    g.initializer.append(numpy_helper.from_array(W2.astype(np.float32), "pe_w_u8"))
    g.initializer.append(numpy_helper.from_array(b2.astype(np.float32), "pe_b_u8"))
    g.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), "u8_scale"))
    g.initializer.append(numpy_helper.from_array(np.array(0, np.uint8), "u8_zp"))
    new_in = helper.make_tensor_value_info(
        "image", onnx.TensorProto.UINT8, [1, h, w, c]
    )
    nodes = [
        helper.make_node(
            "DequantizeLinear", ["image", "u8_scale", "u8_zp"], ["image_f"]
        ),
        helper.make_node("Transpose", ["image_f"], ["image_nchw"], perm=[0, 3, 1, 2]),
    ]
    conv.input[0] = "image_nchw"
    conv.input[1] = "pe_w_u8"
    if len(conv.input) > 2:
        conv.input[2] = "pe_b_u8"
    else:
        conv.input.append("pe_b_u8")
    g.input.remove(inp)
    g.input.insert(0, new_in)
    for i, n in enumerate(nodes):
        g.node.insert(i, n)
    return m2


def main():
    import onnxsim

    raw, out_sim, out_u8 = sys.argv[1:4]
    m = onnx.load(raw)
    m, ok = onnxsim.simplify(m)
    assert ok
    n = fold_rank5_transposes(m)
    del m.graph.value_info[:]  # onnxsim's value_info still has the rank-5 shapes
    m = onnx.shape_inference.infer_shapes(m)
    rk = {
        v.name: len(v.type.tensor_type.shape.dim)
        for v in list(m.graph.value_info) + list(m.graph.output)
    }
    print(
        f"folded {n} rank-5 transposes; max rank {max(rk.values())}, "
        f"rank>4: {[k for k, v in rk.items() if v > 4][:6]}"
    )
    del m.graph.value_info[:]
    onnx.save(m, out_sim)
    onnx.save(u8_input(m), out_u8)


if __name__ == "__main__":
    main()
