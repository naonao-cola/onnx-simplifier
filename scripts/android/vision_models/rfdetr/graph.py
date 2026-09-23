"""ONNX-side passes for the exported RF-DETR graph (run with the repo's onnxsim environment):

- `simplify`: onnxsim.
- `fold_rank5_transposes`: the DINOv2 window partition / merge is Reshape(a,b,c,d,e) ->
  Transpose(0,2,1,3,4) -> Reshape. Merging the last two (untouched) axes gives the identical element
  order at rank 4: Reshape(a,b,c,d*e) -> Transpose(0,2,1,3) -> Reshape. QNN refuses the rank-5 form.
- `fold_backbone_elementwise` (exact up to float rounding; opt-in, `FOLD=1`): the DINOv2 blocks'
  small elementwise ops folded into the neighbouring Linear weights --
  * q/k/v: MatMul -> Split -> 3 bias Adds becomes one 2-D Gemm with the concatenated bias
    (Reshape -> Gemm -> Reshape) -> Split, which also makes it a plain 2-D Linear for SmoothQuant;
  * the attention scale sqrt(1/sqrt(d)) that the SDPA decomposition multiplies into both q and k
    (Mul after their Transposes) goes into the q and k columns of that MatMul and bias;
  * LayerScale (Mul by a per-channel gamma after the attention output dense and fc2 Gemms) goes
    into those Gemms' weight columns and bias.
- `u8_input`: the graph input becomes uint8 NHWC RGB (the camera's bytes, resized to R x R):
  DequantizeLinear(scale 1, zp 0) -> Transpose -> the patch-embed Conv, with RF-DETR's /255 and
  ImageNet mean/std folded into that Conv's weight and bias. The patch embed has stride == kernel
  and no padding, so every input pixel is used and the fold is exact.

usage: graph.py <raw.onnx> <out.sim.onnx> <out.u8.onnx>   (also <out.f255.onnx>: float [0, 255] NHWC
input, the same graph before its input is quantized; quantize.py starts from it)
"""

from __future__ import annotations

import os
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


def fold_backbone_elementwise(m: onnx.ModelProto) -> dict:
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    arr = lambda x: numpy_helper.to_array(inits[x])  # noqa: E731

    def put(name, a):
        t = numpy_helper.from_array(a.astype(np.float32), name)
        if name in inits:
            inits[name].CopyFrom(t)
        else:
            inits[name] = t
            g.initializer.append(t)

    def bypass(n):  # consumers of n's output read n's non-constant input instead
        src = next(x for x in n.input if x not in inits)
        for c in cons.get(n.output[0], []):
            for k, x in enumerate(c.input):
                if x == n.output[0]:
                    c.input[k] = src
        g.node.remove(n)

    stats = {"qkv": 0, "attn_scale": 0, "layer_scale": 0}
    inf = onnx.shape_inference.infer_shapes(m)
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(inf.graph.value_info) + list(inf.graph.input)
    }
    # 1. q/k/v bias + attention scale
    for sp in [
        n for n in g.node if n.op_type == "Split" and "/backbone/" not in n.name
    ]:
        mm = prod.get(sp.input[0])
        if mm is None or mm.op_type != "MatMul" or mm.input[1] not in inits:
            continue
        outs = list(sp.output)
        adds = [cons.get(o, []) for o in outs]
        if len(outs) != 3 or not all(
            len(a) == 1 and a[0].op_type == "Add" for a in adds
        ):
            continue
        adds = [a[0] for a in adds]
        bias_names = [next(x for x in a.input if x in inits) for a in adds]
        if not all("attention" in a.name for a in adds):
            continue
        Wt = arr(mm.input[1]).astype(np.float64)
        bias = np.concatenate([arr(b).astype(np.float64) for b in bias_names])
        C = Wt.shape[1] // 3
        # attention scale: q (0) and k (1) each reach one Mul(scalar const) through Reshape/Transpose
        for i in (0, 1):
            t, mul = adds[i].output[0], None
            for _ in range(3):
                nx = cons.get(t, [])
                if len(nx) != 1:
                    break
                if nx[0].op_type == "Mul":
                    mul = nx[0]
                    break
                if nx[0].op_type not in ("Reshape", "Transpose"):
                    break
                t = nx[0].output[0]
            if mul is None:
                continue
            sc = [x for x in mul.input if x != t]
            if len(sc) != 1:
                continue
            if sc[0] in inits:
                v = arr(sc[0])
            elif sc[0] in prod and prod[sc[0]].op_type == "Constant":
                v = numpy_helper.to_array(prod[sc[0]].attribute[0].t)
            else:
                continue
            if v.size != 1:
                continue
            f = float(v.reshape(()))
            Wt[:, i * C : (i + 1) * C] *= f
            bias[i * C : (i + 1) * C] *= f
            bypass(mul)
            stats["attn_scale"] += 1
        # MatMul + bias as one 2-D Gemm: Reshape(x, [-1, Cin]) -> Gemm(W, b) -> Reshape back
        in_shape = shapes[mm.input[0]]
        base = sp.input[0]
        put(base + "_w", Wt)
        put(base + "_b", bias)
        g.initializer.append(
            numpy_helper.from_array(np.array([-1, Wt.shape[0]], np.int64), base + "_s2")
        )
        g.initializer.append(
            numpy_helper.from_array(
                np.array([*in_shape[:-1], Wt.shape[1]], np.int64), base + "_s3"
            )
        )
        new = [
            helper.make_node(
                "Reshape",
                [mm.input[0], base + "_s2"],
                [base + "_2d"],
                name=base + "/to2d",
            ),
            helper.make_node(
                "Gemm",
                [base + "_2d", base + "_w", base + "_b"],
                [base + "_g"],
                name=base + "/qkv_gemm",
            ),
            helper.make_node(
                "Reshape", [base + "_g", base + "_s3"], [base], name=base + "/to3d"
            ),
        ]
        i = list(g.node).index(mm)
        g.node.remove(mm)
        for k, n in enumerate(new):
            g.node.insert(i + k, n)
        for a in adds:
            bypass(a)
        stats["qkv"] += 1
    # 2. LayerScale into the preceding Gemm (through a Reshape)
    for mul in [n for n in g.node if n.op_type == "Mul" and "layer_scale" in n.name]:
        c_ = [x for x in mul.input if x in inits]
        t = next(x for x in mul.input if x not in inits)
        r = prod.get(t)
        gm = prod.get(r.input[0]) if r is not None and r.op_type == "Reshape" else r
        if (
            len(c_) != 1
            or gm is None
            or gm.op_type != "Gemm"
            or len(cons.get(gm.output[0], [])) != 1
        ):
            continue
        at = {a.name: helper.get_attribute_value(a) for a in gm.attribute}
        if (
            at.get("transB", 0)
            or at.get("alpha", 1.0) != 1.0
            or at.get("beta", 1.0) != 1.0
        ):
            continue
        gamma = arr(c_[0]).astype(np.float64).reshape(-1)
        put(gm.input[1] + "_ls", arr(gm.input[1]).astype(np.float64) * gamma[None, :])
        gm.input[1] = gm.input[1] + "_ls"
        put(gm.input[2] + "_ls", arr(gm.input[2]).astype(np.float64) * gamma)
        gm.input[2] = gm.input[2] + "_ls"
        bypass(mul)
        stats["layer_scale"] += 1
    used = {x for n in g.node for x in n.input}
    for i in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(i)
    return stats


def u8_input(m: onnx.ModelProto, float_input: bool = False) -> onnx.ModelProto:
    """float_input: the same graph but with a float [0, 255] NHWC input (no DequantizeLinear), the
    starting point for quantize.py, which calibrates it with the fixed range (0, 255) so the
    quantized input is again the uint8 RGB bytes (scale 1, zero point 0)."""
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
    dt = onnx.TensorProto.FLOAT if float_input else onnx.TensorProto.UINT8
    new_in = helper.make_tensor_value_info("image", dt, [1, h, w, c])
    nodes = [
        helper.make_node(
            "Transpose",
            ["image_f"],
            ["image_nchw"],
            name="image/to_nchw",
            perm=[0, 3, 1, 2],
        ),
    ]
    if float_input:
        nodes[0].input[0] = "image"
    else:
        nodes.insert(
            0,
            helper.make_node(
                "DequantizeLinear",
                ["image", "u8_scale", "u8_zp"],
                ["image_f"],
                name="image/dq",
            ),
        )
    conv.input[0] = "image_nchw"
    conv.input[1] = "pe_w_u8"
    if len(conv.input) > 2:
        conv.input[2] = "pe_b_u8"
    else:
        conv.input.append("pe_b_u8")
    used = {x for n in g.node for x in n.input}
    for i in [
        i
        for i in g.initializer
        if i.name not in used and i.name not in ("u8_scale", "u8_zp")
    ]:
        g.initializer.remove(i)  # the original patch-embed weight / bias
    if float_input:
        for i in [i for i in g.initializer if i.name in ("u8_scale", "u8_zp")]:
            g.initializer.remove(i)
    g.input.remove(inp)
    g.input.insert(0, new_in)
    for i, n in enumerate(nodes):
        g.node.insert(i, n)
    return m2


def main():
    import onnxsim

    raw, out_sim, out_u8 = sys.argv[1:4]  # also writes <out_u8 with .u8 -> .f255>
    m = onnx.load(raw)
    m, ok = onnxsim.simplify(m)
    assert ok
    n = fold_rank5_transposes(m)
    if os.environ.get("FOLD") == "1":
        print("folded elementwise:", fold_backbone_elementwise(m))
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
    onnx.save(u8_input(m, float_input=True), out_u8.replace(".u8.", ".f255."))


if __name__ == "__main__":
    main()
