#!/usr/bin/env python3
"""QDQ RF-DETR with onnxsim's whole-graph quantizer (onnxsim.full_qdq), region-wise mixed precision,
uint8 NHWC image input (the RGB bytes as-is: scale 1, zero point 0).

usage: quantize.py <variant> <policy> [--n-calib N] [--method minmax|mse|percentile|...]
  -> <work>/<variant>.<policy>.onnx

Starts from export.py's <variant>.f255.onnx (float [0, 255] NHWC input, normalization folded into
the patch embed). Regions, from the exported node names:
  bb    the DINOv2 backbone + projector (everything under /backbone/)
  qsel  two-stage query selection (up to the TopK)
  dec   the decoder layers + heads
Policies (LayerNormalization and Softmax stay fp16 unless noted):
  bb8       backbone int8 (uint8 activations), GELU fp16; qsel + dec fp16
  bb8g16    bb8 with the GELUs quantized too, 16-bit (QNN's quantized elementwise ops are LUTs)
  bb16      backbone W8A16 (uint16 activations), GELU fp16
  bb16g     bb16 with the GELUs quantized too
  g16       fp16 everywhere except the backbone GELUs, quantized 16-bit (SAM's GELU trick)
  bb8x16    backbone int8 with no fp16 islands: its LayerNorm/Softmax/GELU quantized, 16-bit
  bb8r16    bb8x16 plus the residual stream (the Adds / LayerScale Muls feeding Add or LayerNorm)
            16-bit: DINOv2's activation outliers live there
  all8      bb8 plus the decoder's Gemm/MatMul int8 (qsel, GridSample fp16)
Calibration: the deploy pipeline's yolo11n COCO calibration ids (disjoint from the eval ids).
Run with this checkout's onnxsim on PYTHONPATH (plus its built extension).
"""

from __future__ import annotations

import argparse
import sys

import common as C
import numpy as np
import onnx

from onnxsim import full_qdq as F

SENS = {"LayerNormalization", "Softmax"}


def regions(m: onnx.ModelProto):
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}

    def upstream(tensors):
        seen, todo = set(), list(tensors)
        while todo:
            n = producer.get(todo.pop())
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            todo.extend(x for x in n.input if x)
        return seen

    # onnxsim renames the Linear layers (Gemm_NN), so find the backbone structurally: the named
    # /backbone/ nodes whose outputs leave the backbone are its outputs; bb is all their upstream
    named = {n.name for n in g.node if "/backbone/" in n.name}
    outs = [
        o
        for n in g.node
        if n.name in named
        for o in n.output
        if any(o in c.input and c.name not in named for c in g.node)
    ]
    bb = upstream(outs)
    topk = [n for n in g.node if n.op_type == "TopK"]
    assert len(topk) == 1
    qsel = upstream([topk[0].output[1]]) - bb
    dec = {n.name for n in g.node} - bb - qsel
    return bb, qsel, dec


def policy(m: onnx.ModelProto, name: str) -> dict:
    bb, qsel, dec = regions(m)
    ops = {n.name: n.op_type for n in m.graph.node}
    gelu = {n for n in bb if ops[n] == "Gelu"}
    kw: dict = {"exclude_op_types": set(SENS)}
    if name == "bb8":
        kw["exclude_nodes"] = qsel | dec | gelu
    elif name == "bb8g16":
        kw["exclude_nodes"] = qsel | dec
        kw["t16_nodes"] = gelu
    elif name == "bb16":
        kw["exclude_nodes"] = qsel | dec | gelu
        kw["activation_dtype"] = "uint16"
    elif name == "bb16g":
        kw["exclude_nodes"] = qsel | dec
        kw["activation_dtype"] = "uint16"
    elif (
        name == "bb8x16"
    ):  # no fp16 islands: LayerNorm/Softmax/GELU quantized too, 16-bit
        kw["exclude_op_types"] = set()
        kw["exclude_nodes"] = qsel | dec
        kw["t16_nodes"] = {n for n in bb if ops[n] in SENS | {"Gelu"}}
    elif (
        name == "bb8r16"
    ):  # bb8g16 + the residual stream (Adds, LayerScale Muls, norms) 16-bit
        kw["exclude_op_types"] = set()
        kw["exclude_nodes"] = qsel | dec
        g = m.graph
        cons: dict = {}
        for n in g.node:
            for x in n.input:
                cons.setdefault(x, []).append(ops[n.name])
        res = {
            n.name
            for n in g.node
            if n.name in bb
            and ops[n.name] in ("Add", "Mul")
            and any(
                c in ("Add", "LayerNormalization") for c in cons.get(n.output[0], [])
            )
        }
        kw["t16_nodes"] = gelu | res | {n for n in bb if ops[n] in SENS}
    elif (
        name == "g16"
    ):  # only the backbone GELUs quantized (16-bit, a LUT on the HTP); rest fp16
        kw["exclude_op_types"] = set()
        kw["exclude_nodes"] = {n for n in ops if n not in gelu}
        kw["t16_nodes"] = gelu
    elif name == "all8":
        kw["exclude_nodes"] = (
            qsel | gelu | {n for n in dec if ops[n] not in ("Gemm", "MatMul")}
        )
    else:
        raise SystemExit(f"unknown policy {name}")
    t16 = kw.pop("t16_nodes", None)
    tdt = {}
    if t16:  # 16-bit activations for every float tensor these nodes read or write
        for n in m.graph.node:
            if n.name in t16:
                tdt.update({t: "uint16" for t in list(n.input) + list(n.output) if t})
    tdt["image"] = "uint8"  # the camera's RGB bytes as-is
    kw["tensor_dtypes"] = tdt
    return kw


def float_image_to_u8(q: onnx.ModelProto) -> None:
    """A float [0, 255] `image` input becomes uint8 via DequantizeLinear(1, 0) (exact)."""
    from onnx import TensorProto, helper, numpy_helper

    g = q.graph
    for k, v in (
        ("u8_scale", np.array(1.0, np.float32)),
        ("u8_zp", np.array(0, np.uint8)),
    ):
        g.initializer.append(numpy_helper.from_array(v, k))
    for n in g.node:
        n.input[:] = ["image_f" if x == "image" else x for x in n.input]
    g.node.insert(
        0,
        helper.make_node(
            "DequantizeLinear",
            ["image", "u8_scale", "u8_zp"],
            ["image_f"],
            name="image/dq",
        ),
    )
    g.input[0].type.tensor_type.elem_type = TensorProto.UINT8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("policy")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--method", default="minmax")
    a = ap.parse_args()
    m = onnx.load(str(C.WORK / f"{a.variant}.f255.onnx"))
    res = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    kw = policy(m, a.policy)
    data = [
        {"image": C.load_rgb_u8(p, res)[None].astype("float32")}
        for p in C.image_paths("calibration")[: a.n_calib]
    ]
    q = F.quantize_full_qdq(
        m, data, method=a.method, ranges={"image": (0.0, 255.0)}, **kw
    )
    q, info = F.quantized_io(q, inputs=["image"], outputs=[])
    print("io:", info)
    if not info:  # the stem stayed float (e.g. g16)
        float_image_to_u8(q)
    n_q = sum(n.op_type == "QuantizeLinear" for n in q.graph.node)
    tag = a.policy + ("" if a.method == "minmax" else f"-{a.method}")
    out = C.WORK / f"{a.variant}.{tag}.onnx"
    onnx.save(q, str(out))
    print(f"{out}: {len(q.graph.node)} nodes, {n_q} QuantizeLinear")


if __name__ == "__main__":
    sys.exit(main())
