#!/usr/bin/env python3
"""int8 QDQ RT-DETR with onnxsim's whole-graph quantizer (onnxsim.full_qdq), region-wise mixed
precision, uint8 NHWC image input.

usage: quantize.py <policy> [--work <dir>] [--n-calib N]
  -> <work>/full.<policy>.onnx   (input `pixel_values` uint8 [1, 640, 640, 3], the RGB pixels
                                   as-is: scale 1/255, zero point 0)

Regions (found structurally, so they survive onnxsim's node renaming):
  front  backbone + hybrid encoder: every node upstream of the decoder input projections
  qsel   query selection: enc_output/score/box heads, ReduceMax, TopK (nodes upstream of TopK
         that are not front)
  dec    everything after TopK: the 3 decoder layers + heads

Policies (the GELU of the AIFI MLP always stays fp16):
  bb8enc16    backbone uint8 activations, hybrid encoder uint16 (W8A16), qsel + dec fp16 (default)
  front8      front int8, qsel + dec fp16
  front16     front W8A16, qsel + dec fp16
  bb8 / enc8  only the backbone / only the hybrid encoder int8
  front8x:RX  front8 with the front nodes matching regex RX kept fp16 (bisecting)
  front8s     front8 with LayerNorm/Softmax fp16; front8a: with the whole AIFI layer fp16
  front8v     split.py's pre piece: front8 plus the value maps' path int8
  front8lin   front int8, plus the decoder's Gemm/MatMul (Linear layers) int8
  all8        everything int8 except LayerNorm, Softmax, GridSample, qsel (fp16)
  mix8        all8 with GridSample int8 and its sampling coordinates uint16

Calibration: the deploy pipeline's yolo11n COCO calibration ids (disjoint from the eval ids).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # repo root: this checkout's onnxsim

import common as C  # noqa: E402

from onnxsim import full_qdq as F  # noqa: E402


def regions(m: onnx.ModelProto):
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}

    def upstream(tensors):
        seen, todo = set(), list(tensors)
        while todo:
            t = todo.pop()
            n = producer.get(t)
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            todo.extend(x for x in n.input if x)
        return seen

    dproj = [
        n for n in g.node if n.op_type == "Conv" and "decoder_input_proj" in n.name
    ]
    assert len(dproj) == 3, [n.name for n in dproj]
    front = upstream([n.input[0] for n in dproj])
    topk = [n for n in g.node if n.op_type == "TopK"]
    assert len(topk) == 1
    before_topk = upstream([topk[0].output[1]])
    qsel = before_topk - front
    dec = {n.name for n in g.node} - front - qsel
    return front, qsel, dec


def policy(m, name):
    front, qsel, dec = regions(m)
    ops = {n.name: n.op_type for n in m.graph.node}
    sens = {"LayerNormalization", "Softmax", "GridSample"}
    if name == "front8":
        return {"exclude_nodes": qsel | dec}
    if name == "front8v":
        # split.py's pre piece only: front int8 plus the value maps' path (decoder input
        # projections, flatten/concat, the 3 value_proj Linears) int8
        vals = [o.name for o in m.graph.output if o.name.startswith("value")]
        assert vals, "front8v is for split.py's pre piece"
        producer = {o: n for n in m.graph.node for o in n.output}
        up, todo = set(), list(vals)
        while todo:
            n = producer.get(todo.pop())
            if n is None or n.name in up:
                continue
            up.add(n.name)
            todo.extend(x for x in n.input if x)
        return {"exclude_nodes": (qsel | dec) - up}
    if name.startswith(
        "front8x:"
    ):  # front8 with the front nodes matching a regex kept fp16
        import re

        rx = re.compile(name.split(":", 1)[1])
        return {"exclude_nodes": qsel | dec | {n for n in front if rx.search(n)}}
    backbone = {n for n in front if "/backbone/" in n}
    if name == "front16":  # front W8A16 (uint16 activations), qsel + dec fp16
        return {"exclude_nodes": qsel | dec, "activation_dtype": "uint16"}
    if name == "bb8enc16":  # backbone uint8 activations, hybrid encoder uint16
        return {"exclude_nodes": qsel | dec, "t16_nodes": front - backbone}
    if name == "bb16enc8":
        return {"exclude_nodes": qsel | dec, "t16_nodes": backbone}
    if name == "bb8":  # only the ResNet backbone int8
        return {"exclude_nodes": (front - backbone) | qsel | dec}
    if name == "enc8":  # only the hybrid encoder (input projections, AIFI, CCFM) int8
        return {"exclude_nodes": backbone | qsel | dec}
    if name == "front8s":
        # front int8 except LayerNorm/Softmax (the AIFI transformer layer)
        return {
            "exclude_nodes": qsel | dec,
            "exclude_op_types": {"LayerNormalization", "Softmax"},
        }
    if name == "front8a":  # front int8 except the whole AIFI transformer layer
        return {"exclude_nodes": qsel | dec | {n for n in front if "/aifi" in n}}
    if name == "front8lin":
        return {
            "exclude_nodes": qsel | {n for n in dec if ops[n] not in ("Gemm", "MatMul")}
        }
    if name == "all8":
        return {"exclude_nodes": qsel, "exclude_op_types": sens}
    if name == "mix8":
        return {
            "exclude_nodes": qsel,
            "exclude_op_types": sens - {"GridSample"},
            "coords16": True,
        }
    raise SystemExit(f"unknown policy {name}")


def transpose_as_qdq_unit(q: onnx.ModelProto, name: str) -> onnx.ModelProto:
    """quantized_io's NHWC input is `uint8 -> Transpose -> DQ`; QNN refuses that lone uint8
    Transpose here (0xc26). Make it a data-movement QDQ unit instead: `uint8 -> DQ -> Transpose
    -> Q -> DQ` with the same scale/zero point (exact)."""
    from onnx import helper

    g = q.graph
    t = next(n for n in g.node if n.op_type == "Transpose" and n.input[0] == name)
    dq = next(
        n
        for n in g.node
        if n.op_type == "DequantizeLinear" and n.input[0] == t.output[0]
    )
    s_, z_ = dq.input[1], dq.input[2]
    i = list(g.node).index(t)
    new = [
        helper.make_node(
            "DequantizeLinear",
            [name, s_, z_],
            [name + "/dq_nhwc"],
            name=name + "/dq_nhwc",
            domain=dq.domain,
        ),
        helper.make_node(
            "Transpose",
            [name + "/dq_nhwc"],
            [name + "/nchw_f"],
            name=name + "/to_nchw",
            perm=[0, 3, 1, 2],
        ),
        helper.make_node(
            "QuantizeLinear",
            [name + "/nchw_f", s_, z_],
            [t.output[0]],
            name=name + "/q_nchw",
            domain=dq.domain,
        ),
    ]
    g.node.remove(t)
    for k, n in enumerate(new):
        g.node.insert(i + k, n)
    return q


def quant_kwargs(m: onnx.ModelProto, name: str) -> dict:
    """quantize_full_qdq keyword arguments of policy `name` for model m."""
    kw = policy(m, name)
    # The AIFI MLP's exact GELU (Div, Erf, Add, Mul, Mul) stays fp16 as a whole: QNN EP fuses that
    # float pattern into its Gelu, but has no Erf of its own (float or quantized)
    gelu = {n.name for n in m.graph.node if "/activation_fn/" in n.name}
    kw["exclude_nodes"] = set(kw.get("exclude_nodes", ())) | gelu
    tdt = {}
    if kw.pop("coords16", False):
        tdt.update({t: "uint16" for t in F.sampling_coordinate_tensors(m)})
    t16 = kw.pop("t16_nodes", None)
    if t16:  # 16-bit activations for every float tensor these nodes read or write
        for n in m.graph.node:
            if n.name in t16:
                tdt.update({t: "uint16" for t in list(n.input) + list(n.output) if t})
    if kw.get("activation_dtype", "uint8") != "uint8" or tdt.get("pixel_values"):
        tdt["pixel_values"] = "uint8"  # the camera's RGB bytes as-is
    if tdt:
        kw["tensor_dtypes"] = tdt
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-rtdetr/work"))
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    work = Path(a.work)
    m = onnx.load(str(work / "full.sim.onnx"))
    kw = quant_kwargs(m, a.policy)
    data = [
        {"pixel_values": C.to_pixels(C.load_rgb_u8(p))}
        for p in C.image_paths("calibration")[: a.n_calib]
    ]
    q = F.quantize_full_qdq(
        m, data, method=a.method, ranges={"pixel_values": (0.0, 1.0)}, **kw
    )
    q, info = F.quantized_io(
        q, inputs=["pixel_values"], outputs=[], nhwc_inputs=["pixel_values"]
    )
    print("io:", info)
    if info:  # the image input is quantized (the backbone stem is int8)
        q = transpose_as_qdq_unit(q, "pixel_values")
    n_q = sum(n.op_type == "QuantizeLinear" for n in q.graph.node)
    tag = (
        a.policy.replace(":", "_").replace("/", "").replace("|", "+").replace("\\", "")
    )
    out = work / f"full.{tag}{a.suffix}.onnx"
    onnx.save(q, str(out))
    print(f"{out}: {len(q.graph.node)} nodes, {n_q} QuantizeLinear")


if __name__ == "__main__":
    sys.exit(main())
