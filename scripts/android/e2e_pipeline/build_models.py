#!/usr/bin/env python3
"""Build every model piece and one pipeline file per stage for the end-to-end Mask R-CNN run.

Inputs (all produced elsewhere in this repo, nothing downloaded here):
  --work W   backbone.onnx, rest.onnx, maskrcnn_sim.onnx (scripts/android/maskrcnn_e2e/prepare.py)
             and W/opt/ from ../htp_exploration/ceiling/make_optimized.sh (PR #1833's backbone)
  --rpn  R   the fused-RPN model constants: levels.txt, model.txt, l{0..4}_anchors.bin
             (../tinygrad_hexagon_bridge/rpn_fused/capture_rpn_fused.py + rpn_host_check)
  --out  O   everything the phone needs (models + pipe_<stage>.txt)

A pipeline is a list of steps over one tensor store, keyed by the graph's own tensor names. The
CPU parts are *segments* of rest.onnx, cut out automatically around the regions that run
elsewhere (HTP heads, DSP kernels): every CPU node gets the index of the last external region it
depends on, nodes with the same index form one segment, and each segment's inputs/outputs are
exactly the tensors that cross its boundary -- so no piece recomputes another piece's work. Step
lines (one per line, space separated; lists are comma separated, "-" = empty):
  ort     NAME MODEL EP OPTS IN OUT                 EP cpu|htp; OPTS k=v;k=v (QNN options; ctx=1|2:
                                                    EP-context model, binary embedded | own file);
                                                    tensors bind to model inputs/outputs by name
  ortpad  NAME EP OPTS PADIN B:MODEL,B:MODEL IN OUT pad PADIN's dim 0 to the smallest bucket B >= n,
                                                    run that model, slice every OUT back to n
  quant_in SRC DST SCALE ZP                         fp32 [3,H,W] -> uint8 [1,H,W,3] (QuantizeLinear)
  dq      SRC DST SCALE ZP                          uint8 -> fp32, same layout, DSP-shared memory
  quant   SRC DST SCALE ZP                          fp32 -> uint8, same shape (u8_heads.py)
  mask_sel LOGITS LABELS DST SCALE ZP               DST[i,0] = sigmoid(dequant(LOGITS[i,LABELS[i]]))
                                                    (u8_heads.py: the mask head's uint8 logits)
  rpn     SRC SCORES DELTAS OUT                     fused RPN span on the DSP (SRC 0: fp32 per-anchor
                                                    deltas, 1: the backbone's raw uint8 [12,H,W] convs)
  roialign X ROIS OUT OH OW SR SCALE                RoiAlign on the DSP over an NHWC fp32 map X; OUT
                                                    holds (R,OH,OW,C) rows under the graph's
                                                    (R,C,OH,OW) label (the heads read them as such)
"""

import argparse
import collections
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import extract_model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "htp_exploration" / "qnn_shell"))
import rest_split  # noqa: E402
import scatter_rewrite  # noqa: E402

FPN = ["487", "455", "423", "391"]  # P2..P5 (rest.onnx input names)
SCORES = ["881", "1214", "1547", "1880", "2213"]  # P2..P6 per-anchor objectness (fp32)
DELTAS = ["898_910_dequantized", "1231_1236_dequantized", "1564_1562_dequantized",
          "1897_1888_dequantized", "2230_2214_dequantized"]
PROPOSALS = "2527"
BOX_IN, BOX_OUT = rest_split.BOX_IN, rest_split.BOX_OUT
MASK_IN, MASK_OUT = rest_split.MASK_IN, rest_split.MASK_OUT
FINAL = rest_split.FINAL
HEAD_QNN = "htp_performance_mode=burst;htp_graph_finalization_optimization_mode=3"
BB_QNN = "htp_performance_mode=burst"  # mode 3 is 3.6 ms *slower* on the backbone (PR #1833)


class Graph:
    def __init__(self, model):
        self.m = model
        self.g = model.graph
        self.prod = {o: n for n in self.g.node for o in n.output}
        self.cons = collections.defaultdict(list)
        for n in self.g.node:
            for i in n.input:
                self.cons[i].append(n)
        self.index = {id(n): k for k, n in enumerate(self.g.node)}

    def between(self, starts, ends):
        back, stack = set(), list(ends)
        while stack:
            t = stack.pop()
            n = self.prod.get(t)
            if n is None or id(n) in back:
                continue
            back.add(id(n))
            stack.extend(n.input)
        fwd, stack = set(), list(starts)
        while stack:
            t = stack.pop()
            for c in self.cons[t]:
                if id(c) not in fwd:
                    fwd.add(id(c))
                    stack.extend(c.output)
        return back & fwd


def static_nodes(model):
    """ids of nodes whose inputs all derive from initializers/Constants only (weight DQs etc.)."""
    inits = {i.name for i in model.graph.initializer}
    static_t, ids = set(inits), set()
    for n in model.graph.node:
        if n.op_type == "Constant" or all((not t) or t in static_t for t in n.input):
            ids.add(id(n))
            static_t.update(n.output)
    return ids


def partition(model, regions):
    """regions: ordered [(name, set of node ids)]. Returns [segment node lists] (len(regions)+1)."""
    G = Graph(model)
    region_of = {}
    for k, (_, ids) in enumerate(regions, 1):
        for i in ids:
            region_of[i] = k
    level = {}  # tensor -> level
    seg = collections.defaultdict(list)
    static = static_nodes(model)
    for n in G.g.node:
        if id(n) in static:
            continue  # pulled into whichever piece needs it (make_segment / the head models)
        k = region_of.get(id(n))
        lv = 0
        for t in n.input:
            if not t:
                continue
            p = G.prod.get(t)
            if k is not None and p is not None and region_of.get(id(p)) == k:
                continue  # produced inside the same region
            lv = max(lv, level.get(t, 0))
        if k is not None:
            assert lv < k, f"region {regions[k - 1][0]} needs a tensor computed after it ({n.name})"
            for o in n.output:
                level[o] = k
            continue
        for o in n.output:
            level[o] = lv
        seg[lv].append(n)
    crossing = []  # (region, tensor) produced inside a region and read outside it
    for name, ids in regions:
        for n in G.g.node:
            if id(n) in ids:
                crossing += [(name, o) for o in n.output
                             if any(id(c) not in ids for c in G.cons[o])]
    return [seg[k] for k in range(len(regions) + 1)], crossing


def infer_types(model):
    inf = onnx.shape_inference.infer_shapes(model)
    vi = {}
    for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        vi[v.name] = v
    return vi


def make_segment(model, nodes, vi, path):
    """A standalone model for `nodes`; returns (inputs, outputs) tensor names, or None if empty."""
    if not nodes:
        return None
    G = Graph(model)
    inits = {i.name: i for i in model.graph.initializer}
    ids = {id(n) for n in nodes}
    produced = {o for n in nodes for o in n.output}
    graph_outs = {o.name for o in model.graph.output}
    static = static_nodes(model)
    const_nodes, ins, seen_c = [], [], set()

    def pull(t):  # a static tensor: include its producer chain
        p = G.prod.get(t)
        if p is None or id(p) in seen_c:
            return
        seen_c.add(id(p))
        for u in p.input:
            if u and u not in inits:
                pull(u)
        const_nodes.append(p)

    for n in nodes:
        for t in n.input:
            if not t or t in produced or t in inits or t in ins:
                continue
            p = G.prod.get(t)
            if p is not None and id(p) in static:
                pull(t)
                continue
            ins.append(t)
    outs = []
    for n in nodes:
        for o in n.output:
            if o in graph_outs or any(id(c) not in ids for c in G.cons[o]):
                outs.append(o)
    used_inits = [inits[t] for n in nodes + const_nodes for t in n.input if t in inits]
    uniq = {i.name: i for i in used_inits}
    order = sorted(nodes, key=lambda n: G.index[id(n)])
    const_nodes.sort(key=lambda n: G.index[id(n)])
    g = helper.make_graph(const_nodes + order, Path(path).stem, [vi[t] for t in ins],
                          [vi[t] for t in outs], initializer=list(uniq.values()))
    m = helper.make_model(g, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.save(m, path)
    return ins, outs


def pin_first_dim(src, dst, name, dims):
    rest_split._pin(src, dst, {name: dims})


def nhwc_box_head(src, dst):
    """Box head taking (R,7,7,256) rows: same graph, fc6's int8 weight rows permuted to match."""
    m = onnx.load(src)
    g = m.graph
    G = Graph(m)
    inp = g.input[0]
    n, c, h, w = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    for d, v in zip(inp.type.tensor_type.shape.dim, [n, h, w, c]):
        d.dim_value = v
    # follow the flatten -> (Q,DQ) -> MatMul chain to fc6's weight initializer
    t = inp.name
    while True:
        (nx,) = G.cons[t]
        if nx.op_type == "MatMul":
            break
        t = nx.output[0]
    wname = nx.input[1]
    wp = G.prod.get(wname)
    inits = {i.name: i for i in g.initializer}
    if wp is not None and wp.op_type == "DequantizeLinear":
        sc = numpy_helper.to_array(inits[wp.input[1]])
        assert sc.size == 1, "per-tensor weight scale expected"
        wname = wp.input[0]
    W = numpy_helper.to_array(inits[wname])
    assert W.shape[0] == c * h * w, W.shape
    k = np.arange(c * h * w)
    ci, hw = k % c, k // c  # new (NHWC) flat index k = hw*c + ci  <-  old index ci*h*w + hw
    Wn = W[ci * h * w + hw]
    inits[wname].CopyFrom(numpy_helper.from_array(Wn, wname))
    onnx.save(m, dst)


def nhwc_mask_head(src, dst):
    """Mask head taking (R,14,14,256) rows: a leading Transpose back to the graph's NCHW."""
    m = onnx.load(src)
    g = m.graph
    inp = g.input[0]
    n, c, h, w = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    new = inp.name + "_nhwc"
    for node in g.node:
        for k, t in enumerate(node.input):
            if t == inp.name:
                node.input[k] = inp.name + "_nchw"
    g.node.insert(0, helper.make_node("Transpose", [new], [inp.name + "_nchw"], perm=[0, 3, 1, 2]))
    del g.input[:]
    g.input.append(helper.make_tensor_value_info(new, onnx.TensorProto.FLOAT, [n, h, w, c]))
    onnx.save(m, dst)
    return new


def adapter(step2, step3_outs, want, dst, fpn_nhwc=None):
    """uint8 raw backbone outputs -> rest.onnx's fp32 inputs, cut out of the step-2 backbone (the
    chains PR #1833's raw_rpn_outputs.py removed) + a DequantizeLinear per output."""
    import json
    q = json.loads((Path(step2).parent / "outputs_qparams.json").read_text())
    byfloat = {v["float_output"]: (k, v) for k, v in q.items()}
    rpn_want = [f for f in want if f not in FPN]
    if rpn_want:
        tmp = str(dst) + ".cut.onnx"
        extract_model(step2, tmp, step3_outs, [byfloat[f][0] for f in rpn_want])
        m = onnx.load(tmp)
        Path(tmp).unlink()
    else:
        ref = onnx.load(step2, load_external_data=False)
        m = helper.make_model(helper.make_graph([], "adapter", [], []), opset_imports=ref.opset_import,
                              ir_version=ref.ir_version)
    g = m.graph
    for f in want:
        qname, v = byfloat[f]
        s, z = f + "_adapt_s", f + "_adapt_z"
        g.initializer.extend([numpy_helper.from_array(np.array(v["scale"], np.float32), s),
                              numpy_helper.from_array(np.array(v["zero_point"], np.uint8), z)])
        src = qname
        if f in FPN:
            src = qname + "_nhwc"  # the optimized backbone's NHWC uint8 FPN output
            g.input.append(helper.make_tensor_value_info(src, onnx.TensorProto.UINT8, fpn_nhwc[f]))
            g.node.append(helper.make_node("Transpose", [src], [qname], perm=[0, 3, 1, 2]))
        g.node.append(helper.make_node("DequantizeLinear", [qname, s, z], [f]))
    del g.output[:]
    g.output.extend(helper.make_tensor_value_info(f, onnx.TensorProto.FLOAT, None) for f in want)
    onnx.save(m, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rpn", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    O = a.out
    O.mkdir(parents=True, exist_ok=True)
    for f in ["maskrcnn_sim.onnx", "backbone.onnx", "rest.onnx"]:
        shutil.copy(a.work / f, O / f)
    shutil.copy(a.work / "opt/5_final/backbone.onnx", O / "backbone_opt.onnx")
    for f in ["levels.txt", "model.txt", "rpn_region.onnx"] + [f"l{i}_anchors.bin" for i in range(5)]:
        shutil.copy(a.rpn / f, O / f)
    scatter_rewrite.rewrite(str(O / "rest.onnx"), str(O / "rest_snd.onnx"))
    rest = onnx.load(str(O / "rest_snd.onnx"))
    # one Shape node outside the mask head reads its pre-sigmoid logits 6856; it only needs the
    # shape, which the head's output 6857 shares (same fix as rest_split.py's cpu pieces)
    for n in rest.graph.node:
        if n.op_type == "Shape" and list(n.input) == ["6856"]:
            n.input[0] = "6857"
    onnx.save(rest, str(O / "rest_snd.onnx"))
    vi = infer_types(rest)
    G = Graph(rest)

    # heads (the same cut as rest_split.py), pinned to static batches, plus NHWC-row variants
    extract_model(str(O / "rest_snd.onnx"), str(O / "box_head_dyn.onnx"), [BOX_IN], BOX_OUT)
    pin_first_dim(str(O / "box_head_dyn.onnx"), str(O / "box_head_1000.onnx"), BOX_IN, [1000, 256, 7, 7])
    nhwc_box_head(str(O / "box_head_1000.onnx"), str(O / "box_head_1000_nhwc.onnx"))
    extract_model(str(O / "rest_snd.onnx"), str(O / "mask_head_dyn.onnx"), [MASK_IN], MASK_OUT)
    mask_nhwc_in = None
    for b in (32, 100):
        pin_first_dim(str(O / "mask_head_dyn.onnx"), str(O / f"mask_head_{b}.onnx"), MASK_IN, [b, 256, 14, 14])
        mask_nhwc_in = nhwc_mask_head(str(O / f"mask_head_{b}.onnx"), str(O / f"mask_head_{b}_nhwc.onnx"))

    # adapters for the optimized backbone (PR #1833): raw uint8 outputs -> rest.onnx inputs
    opt = a.work / "opt"
    s3 = onnx.load(str(opt / "3_qraw/backbone.onnx"), load_external_data=False)
    raw = [o.name for o in s3.graph.output if o.type.tensor_type.shape.dim[1].dim_value != 256]
    raw_scores = [o.name for o in s3.graph.output if o.type.tensor_type.shape.dim[1].dim_value == 3]
    raw_deltas = [o.name for o in s3.graph.output if o.type.tensor_type.shape.dim[1].dim_value == 12]
    hw = lambda n: [d.dim_value for d in next(o for o in s3.graph.output if o.name == n).type.tensor_type.shape.dim][2:]  # noqa: E731
    raw_scores.sort(key=lambda n: -np.prod(hw(n)))
    raw_deltas.sort(key=lambda n: -np.prod(hw(n)))
    fpn_nhwc = {}
    for f in FPN:
        s, c, h, w = [d.dim_value for d in vi[f].type.tensor_type.shape.dim]
        fpn_nhwc[f] = [s, h, w, c]
    step2 = str(opt / "2_qout/backbone.onnx")
    rest_inputs = [i.name for i in rest.graph.input]
    adapter(step2, raw, rest_inputs, O / "adapt_all.onnx", fpn_nhwc)
    adapter(step2, raw_scores, SCORES, O / "adapt_scores.onnx")
    adapter(step2, raw_deltas, DELTAS, O / "adapt_deltas.onnx")  # host emulation of rpn SRC 1 only
    adapter(step2, [], FPN, O / "adapt_fpn.onnx", fpn_nhwc)
    import json
    q2 = json.loads((opt / "2_qout/outputs_qparams.json").read_text())
    fpn_q = {v["float_output"]: (k + "_nhwc", v["scale"], v["zero_point"]) for k, v in q2.items() if v["float_output"] in FPN}

    # regions of rest_snd.onnx
    rpn_ids = G.between(SCORES + DELTAS, [PROPOSALS])
    box_ids = G.between([BOX_IN], BOX_OUT)
    mask_ids = G.between([MASK_IN], MASK_OUT)
    ra = [n for n in rest.graph.node if n.op_type == "RoiAlign"]
    box_ra = [n for n in ra if BOX_IN in _downstream(G, n.output[0])]
    mask_ra = [n for n in ra if n not in box_ra]
    assert len(box_ra) == 4 and len(mask_ra) == 4

    def ra_steps(nodes):
        out = []
        for n in nodes:
            at = {x.name: helper.get_attribute_value(x) for x in n.attribute}
            out.append(f"roialign {n.input[0]}@nhwc {n.input[1]} {n.output[0]} {at['output_height']} "
                       f"{at['output_width']} {at['sampling_ratio']} {at['spatial_scale']!r}")
        return out

    def box_head(nhwc):
        m = "box_head_1000_nhwc.onnx" if nhwc else "box_head_1000.onnx"
        return f"ortpad box_head htp {HEAD_QNN} {BOX_IN} 1000:{m} {BOX_IN} {','.join(BOX_OUT)}"

    def mask_head(nhwc):
        sfx = "_nhwc" if nhwc else ""
        return (f"ortpad mask_head htp {HEAD_QNN} {MASK_IN} 32:mask_head_32{sfx}.onnx,"
                f"100:mask_head_100{sfx}.onnx {MASK_IN} {','.join(MASK_OUT)}")

    def cpu_rest(stage, regions, region_steps, pre):
        segs, leaks = partition(rest, regions)
        declared = {o for st in region_steps for line in st for o in _step_outputs(line)}
        bad = [(r, t) for r, t in leaks if t not in declared]
        assert not bad, f"{stage}: region-internal tensors used outside their region: {bad}"
        lines = list(pre)
        for k, nodes in enumerate(segs):
            path = O / f"seg_{stage}_{k}.onnx"
            io = make_segment(rest, nodes, vi, str(path))
            if io is not None:
                lines.append(f"ort seg{k} {path.name} cpu - {','.join(io[0]) or '-'} {','.join(io[1])}")
            if k < len(region_steps):
                lines.extend(region_steps[k])
        (O / f"pipe_{stage}.txt").write_text("\n".join(lines) + "\n")
        print(stage, [len(s) for s in segs])

    bb = f"ort backbone backbone.onnx htp {BB_QNN} image {','.join(rest_inputs)}"
    bbo = ["quant_in image image_u8 1.0776140689849854 114",
           f"ort backbone backbone_opt.onnx htp {BB_QNN} image_u8 {','.join(o.name for o in onnx.load(str(O / 'backbone_opt.onnx'), load_external_data=False).graph.output)}"]
    fpn_q_lines = [f"dq {fpn_q[f][0]} {f}@nhwc {fpn_q[f][1]!r} {fpn_q[f][2]}" for f in FPN]
    (O / "pipe_ref.txt").write_text(f"ort full maskrcnn_sim.onnx cpu - image {','.join(FINAL)}\n")
    (O / "pipe_a.txt").write_text(bb + f"\nort rest rest.onnx cpu - {','.join(rest_inputs)} {','.join(FINAL)}\n")
    (O / "pipe_b.txt").write_text(bb + f"\nort rest rest_snd.onnx cpu - {','.join(rest_inputs)} {','.join(FINAL)}\n")
    heads = [("box_head", box_ids), ("mask_head", mask_ids)]
    cpu_rest("c", heads, [[box_head(False)], [mask_head(False)]], [bb])
    rpn0 = f"rpn 0 {','.join(SCORES)} {','.join(DELTAS)} {PROPOSALS}"
    rpn1 = f"rpn 1 {','.join(SCORES)} {','.join(raw_deltas)} {PROPOSALS}"
    cpu_rest("d", [("rpn", rpn_ids)] + heads, [[rpn0], [box_head(False)], [mask_head(False)]], [bb])
    ain = [i.name for i in onnx.load(str(O / "adapt_all.onnx"), load_external_data=False).graph.input]
    adapt_all = f"ort adapt adapt_all.onnx cpu - {','.join(ain)} {','.join(rest_inputs)}"
    cpu_rest("c_opt", heads, [[box_head(False)], [mask_head(False)]], bbo + [adapt_all])
    adapt_sc = f"ort adapt_scores adapt_scores.onnx cpu - {','.join(raw_scores)} {','.join(SCORES)}"
    adapt_fpn = f"ort adapt_fpn adapt_fpn.onnx cpu - {','.join(fpn_q[f][0] for f in FPN)} {','.join(FPN)}"
    cpu_rest("d_opt", [("rpn", rpn_ids)] + heads, [[rpn1], [box_head(False)], [mask_head(False)]],
             bbo + [adapt_sc, adapt_fpn])
    ids = lambda ns: {id(n) for n in ns}  # noqa: E731
    cpu_rest("e_opt", [("rpn", rpn_ids), ("box_ra", ids(box_ra)), ("box_head", box_ids),
                       ("mask_ra", ids(mask_ra)), ("mask_head", mask_ids)],
             [[rpn1], ra_steps(box_ra), [box_head(True)], ra_steps(mask_ra), [mask_head(True)]],
             bbo + [adapt_sc] + fpn_q_lines)
    # same final pipeline with every HTP session loaded from an EP-context model (Ort::CompileModel,
    # embed mode): PR #1832 saw the box head run ~2x slower that way, PR #1829 not the backbone
    e = (O / "pipe_e_opt.txt").read_text()
    (O / "pipe_e_opt_ctx.txt").write_text(e.replace(f" htp {HEAD_QNN} ", f" htp {HEAD_QNN};ctx=1 ")
                                           .replace(f" htp {BB_QNN} ", f" htp {BB_QNN};ctx=1 "))
    (O / "mask_nhwc_input.txt").write_text(mask_nhwc_in + "\n")
    print("wrote", O)


def _step_outputs(line):
    f = line.split()
    if f[0] in ("ort",):
        return f[6].split(",")
    if f[0] == "ortpad":
        return f[7].split(",")
    if f[0] == "rpn":
        return [f[4]]
    if f[0] == "roialign":
        return [f[3]]
    return []


def _downstream(G, t):
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for c in G.cons[x]:
            stack.extend(c.output)
    return seen


if __name__ == "__main__":
    main()
