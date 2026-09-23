"""fp16 graph outputs for the split pieces (what the phone's DFA reads directly).

  python io16.py --work <work> [--outs w] [--suffix w16]

The HTP runs these fp16 graphs anyway; an fp32 graph output makes it convert and ship twice the
bytes across the EP boundary. w is (24, 900, 8, 16) = 11 MB as fp32, and pre0 (one Linear + a
softmax) spent most of its 12.4 ms on it. This appends Cast(to=float16) to the named outputs of
pre0 and every mid piece, writing <piece>.<suffix>.onnx next to <piece>.sim.onnx. s4d_run picks the
variant with S4D_MODELS=<suffix> and tells the DFA skel (flags bit 16) when w arrives as fp16.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

PIECES = ["pre0"] + [f"mid{k}{t}" for k in range(5) for t in "FT"]


def cast_outputs(m: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    g = m.graph
    for o in g.output:
        if o.name not in names:
            continue
        src = o.name + "_f32"
        for n in g.node:
            n.output[:] = [src if x == o.name else x for x in n.output]
            n.input[:] = [src if x == o.name else x for x in n.input]
        g.node.append(helper.make_node("Cast", [src], [o.name], to=TensorProto.FLOAT16, name=o.name + "_to_f16"))
        o.type.tensor_type.elem_type = TensorProto.FLOAT16
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--outs", default="w", help="comma-separated outputs to emit as fp16")
    ap.add_argument("--suffix", default="w16")
    a = ap.parse_args()
    sp = Path(a.work) / "split"
    for p in PIECES:
        m = cast_outputs(onnx.load(str(sp / f"{p}.sim.onnx")), a.outs.split(","))
        onnx.checker.check_model(m)
        onnx.save(m, str(sp / f"{p}.{a.suffix}.onnx"))
        print(f"{p}.{a.suffix}.onnx: {a.outs} -> fp16")


if __name__ == "__main__":
    main()
