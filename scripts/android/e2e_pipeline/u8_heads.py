#!/usr/bin/env python3
"""Give the HTP heads uint8 graph boundaries, and write pipe_e_u8.txt / pipe_e_u8_ctx.txt.

  python u8_heads.py O        (O = build_models.py --out dir, after it wrote pipe_e_opt.txt)

The box and mask heads that build_models.py emits take fp32 inputs (the RoiAlign rows) and quantize
them as the first thing the HTP graph does; the mask head also ends in DequantizeLinear + Sigmoid
to an fp32 [n,81,28,28] output. Those fp32 boundaries are expensive on the HTP, and much more so
when the graph is loaded from an EP-context model (which is why EP-context looked 1.7x slower on
the box head, PR #1832/#1841; measured per model on the phone with qnn_run, burst + opt mode 3):

  box head, fp32 in -> Reshape -> Quantize      JIT 27.9 ms   EP-context 48.3 ms
  box head, uint8 [1000,12544] in               JIT 24.6 ms   EP-context 25.3 ms
  mask head 100, fp32 in / fp32 sigmoid out     JIT 11.4 ms   EP-context 14.8 ms
  mask head 100, uint8 in / uint8 logits out    JIT  5.3 ms   EP-context  7.8 ms

So here the graph-input QuantizeLinear moves to a CPU `quant` step (the box head's flatten
Reshape goes too: the rows are already contiguous NHWC), and the mask head's final
DequantizeLinear + Sigmoid moves into a `mask_sel` step that also does seg5's per-detection class
gather -- 1 of 81 channels converted instead of all of them, through an exact 256-entry table.
Same quantization parameters as the graph's own nodes, so the uint8 values the HTP sees are the
ones it computed itself before.

pipe_e_u8ra.txt / pipe_e_u8ra_ctx.txt additionally use the merged uint8 RoiAlign skel
(../tinygrad_hexagon_bridge/roialign_fast/roialign_u8_*): per head, one `roialign_u8` call reads the
backbone's uint8 NHWC maps (staged into rpcmem once per frame by `rpc_stage`) and writes the head's
uint8 input rows directly -- replacing the maps' dq, the 4 per-level roialign steps, the ScatterND
merge segment (seg2/seg4) and the `quant` step.
"""

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper


def _qparams(g, node):
    init = {i.name: i for i in g.initializer}
    return float(numpy_helper.to_array(init[node.input[1]])), int(numpy_helper.to_array(init[node.input[2]]))


def _rename_input(g, old, new):
    for n in g.node:
        n.input[:] = [new if t == old else t for t in n.input]


def _drop_value_info(g, names):
    for v in [v for v in g.value_info if v.name in names]:
        g.value_info.remove(v)


def u8_input(m):
    """Remove the graph input's QuantizeLinear (optionally behind a flatten Reshape, which goes too:
    the input is then fed pre-flattened). Returns its (scale, zero point)."""
    g = m.graph
    x = g.input[0].name
    (first,) = [n for n in g.node if x in n.input]
    q = first if first.op_type == "QuantizeLinear" else [n for n in g.node if first.output[0] in n.input][0]
    assert q.op_type == "QuantizeLinear", q.op_type
    s, z = _qparams(g, q)
    g.node.remove(q)
    _rename_input(g, q.output[0], q.input[0])
    if first is not q:
        assert first.op_type in ("Reshape", "Transpose"), first.op_type
        if first.op_type == "Reshape":  # [n,7,7,256] -> [n,12544]: feed the flat shape directly
            dims = g.input[0].type.tensor_type.shape.dim
            flat = 1
            for d in dims[1:]:
                flat *= d.dim_value
            g.node.remove(first)
            _rename_input(g, first.output[0], x)
            del dims[2:]
            dims[1].dim_value = flat
        # a Transpose stays: uint8 NHWC -> NCHW is a layout no-op for the (NHWC) HTP
    g.input[0].type.tensor_type.elem_type = TensorProto.UINT8
    _drop_value_info(g, {q.input[0], first.output[0]})
    return s, z


def u8_logits_output(m):
    """Remove the graph output's DequantizeLinear + Sigmoid: the output becomes the uint8 logits.
    Returns the DequantizeLinear's (scale, zero point)."""
    g = m.graph
    y = g.output[0].name
    (sig,) = [n for n in g.node if y in n.output]
    assert sig.op_type == "Sigmoid", sig.op_type
    (dq,) = [n for n in g.node if sig.input[0] in n.output]
    assert dq.op_type == "DequantizeLinear", dq.op_type
    s, z = _qparams(g, dq)
    g.node.remove(sig)
    g.node.remove(dq)
    for n in g.node:
        n.output[:] = [y if t == dq.input[0] else t for t in n.output]
    g.output[0].type.tensor_type.elem_type = TensorProto.UINT8
    _drop_value_info(g, {dq.input[0], sig.input[0], y})
    return s, z


def main(out):
    O = Path(out)
    lines = (O / "pipe_e_opt.txt").read_text().splitlines()
    res, mask_q = [], None
    for ln in lines:
        f = ln.split()
        if f[0] == "ortpad" and f[1] in ("box_head", "mask_head"):
            buckets, qin = [], None
            for e in f[5].split(","):
                b, src = e.split(":")
                dst = src.replace("_nhwc.onnx", "_u8.onnx")
                m = onnx.load(str(O / src))
                qp = u8_input(m)
                assert qin in (None, qp)
                qin = qp
                if f[1] == "mask_head":
                    lq = u8_logits_output(m)
                    assert mask_q is None or mask_q[0] == lq
                    mask_q = (lq, f[7])
                onnx.checker.check_model(m)
                onnx.save(m, str(O / dst))
                buckets.append(f"{b}:{dst}")
            x = f[4]
            res.append(f"quant {x} {x}_u8 {qin[0]!r} {qin[1]}")
            f[4] = f[6] = f"{x}_u8"
            f[5] = ",".join(buckets)
            res.append(" ".join(f))
        elif f[0] == "ort" and mask_q and mask_q[1] in f[5].split(","):
            # seg5: the per-detection class gather of the mask probabilities
            (s, z), logits = mask_q
            ins, outs = f[5].split(","), f[6].split(",")
            assert len(ins) == 2 and len(outs) == 1, ln
            labels = ins[1] if ins[0] == logits else ins[0]
            res.append(f"mask_sel {logits} {labels} {outs[0]} {s!r} {z}")
        else:
            res.append(ln)
    assert mask_q, "no mask head in pipe_e_opt.txt"
    e = "\n".join(res) + "\n"
    (O / "pipe_e_u8.txt").write_text(e)
    # EP-context variant: ctx=2 = context binary in its own file next to a small .ctx0.onnx (loads
    # faster than the embedded form); the phone compiles it on first use and reuses it after.
    (O / "pipe_e_u8_ctx.txt").write_text("\n".join(_ctx(ln) for ln in res) + "\n")
    ra = u8_roialign(res)
    (O / "pipe_e_u8ra.txt").write_text("\n".join(ra) + "\n")
    (O / "pipe_e_u8ra_ctx.txt").write_text("\n".join(_ctx(ln) for ln in ra) + "\n")
    print("wrote", O / "pipe_e_u8.txt", O / "pipe_e_u8_ctx.txt", O / "pipe_e_u8ra.txt", O / "pipe_e_u8ra_ctx.txt")


def u8_roialign(lines):
    """pipe_e_u8 lines -> the roialign_u8 variant (see the module docstring)."""
    F = [ln.split() for ln in lines]
    dq = {f[2]: f for f in F if f[0] == "dq"}  # map name (e.g. 391@nhwc) -> dq line
    ra = [f for f in F if f[0] == "roialign"]
    maps = []
    for f in ra:
        if f[1] not in maps:
            maps.append(f[1])
    assert len(maps) == 4 and all(m in dq for m in maps), maps
    for f in F:  # the dq outputs feed nothing but the roialign lines
        if f[0] not in ("dq", "roialign"):
            assert not set(",".join(f[1:]).split(",")) & set(dq), f
    staged = {m: dq[m][1] + "@rpc" for m in maps}
    ra_out = {f[3]: f for f in ra}
    out, drop = [], set()
    for f in F:
        if f[0] == "dq":
            if f is F[[g[0] for g in F].index("dq")]:  # first dq line: stage the maps once instead
                out.append(f"rpc_stage maps {','.join(dq[m][1] for m in maps)} {','.join(staged[m] for m in maps)}")
            continue
        if f[0] == "roialign":
            continue
        if f[0] == "ort" and set(f[5].split(",")) & set(ra_out):
            # the merge segment: ins = DATA, (ROWS, ROI_OUT) pairs; its output feeds a quant step
            ins = f[5].split(",")
            pairs = {ins[i + 1]: ins[i] for i in range(1, len(ins) - 1, 2)}
            (q,) = [g for g in F if g[0] == "quant" and g[1] == f[6]]
            spans = [ra_out[o] for o in ins[2::2]]
            spans.sort(key=lambda g: maps.index(g[1]))
            lv = []
            for g in spans:
                m = dq[g[1]]
                lv.append(f"{staged[g[1]]}:{m[3]}:{m[4]}:{g[7]}:{g[2]}:{pairs[g[3]]}")
            oh, ow, sr = spans[0][4], spans[0][5], spans[0][6]
            C = 256  # FPN channels (the kernel checks C % 128 == 0, C <= 256)
            name = {"7": "box_ra", "14": "mask_ra"}.get(oh, "ra_" + oh)
            out.append(f"roialign_u8 {name} {ins[0]} {q[2]} {C} {oh} {ow} {sr} {q[3]} {q[4]} {','.join(lv)}")
            drop.add(id(q))
            continue
        if id(f) in drop:
            continue
        out.append(" ".join(f))
    return out


def _ctx(ln):
    f = ln.split()
    if f[0] not in ("ort", "ortpad"):
        return ln
    k = 3 if f[0] == "ort" else 2  # EP field
    if f[k] != "htp":
        return ln
    f[k + 1] = f[k + 1] + ";ctx=2" if f[k + 1] != "-" else "ctx=2"
    return " ".join(f)


if __name__ == "__main__":
    main(sys.argv[1])
