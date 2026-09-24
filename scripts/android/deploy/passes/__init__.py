"""Reusable graph rewrites for QDQ models headed to the HTP (the `rewrites:` list in a spec).

Each rewrite is `fn(model, **args) -> (model, info)`; `info` is JSON metadata the pipe stage and
the accuracy stage read back (e.g. the quantization parameters of a uint8 input). They generalize
the Mask R-CNN backbone rewrites in ../../htp_exploration/ceiling/ (#1833), which measured each
of these as lossless and worth 1-3 ms on that backbone:

  uint8_input    graph input becomes the uint8 tensor the first QuantizeLinear would have made
                 (optionally NHWC), so the HTP skips an fp32 quantize (and, NHWC, a transpose)
  uint8_outputs  outputs produced by DequantizeLinear are handed out as their uint8 input instead;
                 the consumer dequantizes only what it needs
  script         run an existing `script.py <in.onnx> <out.onnx> [args...]` rewrite unchanged
                 (how the Mask R-CNN spec reuses ceiling/*.py)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


def _consumers(g):
    c = {}
    for n in g.node:
        for i in n.input:
            c.setdefault(i, []).append(n)
    return c


def _init(g):
    return {i.name: numpy_helper.to_array(i) for i in g.initializer}


def uint8_input(m: onnx.ModelProto, layout: str = "nhwc", input: str | None = None):
    g = m.graph
    init, cons = _init(g), _consumers(g)
    gi = next(i for i in g.input if input in (None, i.name) and i.name not in init)
    x = gi.name
    first = cons.get(x, [])
    unsq = None
    if len(first) == 1 and first[0].op_type == "Unsqueeze":
        unsq = first[0]
        first = cons.get(unsq.output[0], [])
    if len(first) != 1 or first[0].op_type != "QuantizeLinear":
        raise ValueError(f"uint8_input: input {x} is not quantized by exactly one QuantizeLinear")
    q = first[0]
    if init[q.input[2]].dtype != "uint8":
        raise ValueError("uint8_input: input QuantizeLinear is not uint8")
    dqs = cons.get(q.output[0], [])
    if not dqs or any(n.op_type != "DequantizeLinear" for n in dqs):
        raise ValueError("uint8_input: QuantizeLinear output feeds something other than DequantizeLinear")
    dims = [d.dim_value for d in gi.type.tensor_type.shape.dim]
    if unsq is not None:
        dims = [1, *dims]
    n, c, h, w = dims
    scale, zp = float(init[q.input[1]]), int(init[q.input[2]])
    new = f"{x}_u8"
    if layout == "nhwc":
        vi = helper.make_tensor_value_info(new, TensorProto.UINT8, [n, h, w, c])
        g.node.insert(0, helper.make_node("Transpose", [new], [new + "_nchw"], perm=[0, 3, 1, 2],
                                          name=new + "_to_nchw"))
        src = new + "_nchw"
    elif layout == "nchw":
        vi = helper.make_tensor_value_info(new, TensorProto.UINT8, [n, c, h, w])
        src = new
    else:
        raise ValueError(f"uint8_input: layout {layout}")
    for d in dqs:
        d.input[0] = src
    g.node.remove(q)
    if unsq is not None:
        g.node.remove(unsq)
    idx = list(g.input).index(gi)
    g.input.remove(gi)
    g.input.insert(idx, vi)
    return m, {"input": x, "name": new, "layout": layout, "scale": scale, "zero_point": zp,
               "shape": [d.dim_value for d in vi.type.tensor_type.shape.dim]}


def uint8_outputs(m: onnx.ModelProto, outputs: list[str] | None = None):
    g = m.graph
    init = _init(g)
    prod = {o: n for n in g.node for o in n.output}
    cons = _consumers(g)
    info = {}
    for go in list(g.output):
        if outputs is not None and go.name not in outputs:
            continue
        dq = prod.get(go.name)
        if dq is None or dq.op_type != "DequantizeLinear" or cons.get(go.name):
            continue
        zp = init.get(dq.input[2]) if len(dq.input) > 2 else None
        if zp is None or zp.dtype != "uint8" or init[dq.input[1]].size != 1:
            continue
        src = dq.input[0]
        shape = [d.dim_value for d in go.type.tensor_type.shape.dim]
        idx = list(g.output).index(go)
        g.output.remove(go)
        g.output.insert(idx, helper.make_tensor_value_info(src, TensorProto.UINT8, shape))
        if not cons.get(dq.output[0]):
            g.node.remove(dq)
        info[go.name] = {"name": src, "scale": float(init[dq.input[1]]), "zero_point": int(zp)}
    if not info:
        raise ValueError("uint8_outputs: no graph output is a per-tensor uint8 DequantizeLinear")
    return m, info


def script(m: onnx.ModelProto, path: str, args: list | None = None, meta: str | None = None):
    here = Path(__file__).resolve().parents[1]
    p = (here / path).resolve() if not Path(path).is_absolute() else Path(path)
    with tempfile.TemporaryDirectory() as t:
        src, dst = Path(t) / "in.onnx", Path(t) / "out" / "model.onnx"
        dst.parent.mkdir()
        onnx.save(m, str(src))
        subprocess.run([sys.executable, str(p), str(src), str(dst), *map(str, args or [])], check=True)
        out = onnx.load(str(dst))
        info = {"script": str(path)}
        if meta and (dst.parent / meta).exists():
            info.update(json.loads((dst.parent / meta).read_text()))
    return out, info


REWRITES = {"uint8_input": uint8_input, "uint8_outputs": uint8_outputs, "script": script}
