"""Split Mask R-CNN's rest.onnx into its two static heads (run on the HTP) and the dynamic
remainder (run on the CPU), and stitch them back together end to end.

The heads are cut at fixed tensor names in the real graph (see ../rest_htp_findings.md):
  box head : 2788 [nroi,256,7,7]  -> 2811 (softmax scores) [nroi,81], 2810 (box deltas) [nroi,324]
  mask head: 6833 [ndet,256,14,14] -> 6857 (mask probs) [ndet,81,28,28]
Inside the full graph their batch dims are dynamic, so the QNN EP rejects every head node with
"Cannot get shape"; cut out and pinned to a static batch, both run entirely on the HTP.

usage:
  python rest_split.py make  <rest.onnx> <outdir>             # write the head/remainder models
  python rest_split.py stitch <outdir> <backbone_outputs.npz> <host|phone> [workdir]
The stitch step runs the three CPU pieces with ORT on the host and the two heads either on host
ORT ("host", a self-check that the cut is exact) or on the phone's HTP via run_multi.sh ("phone").
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import extract_model

BOX_IN, BOX_OUT = "2788", ["2811", "2810"]
MASK_IN, MASK_OUT = "6833", ["6857"]
MASK_BATCH = 100  # detections_per_img; the mask head is padded to this static batch
FINAL = ["6568", "6570", "6572", "6887"]
SHAPES = {"2811": ["nroi", 81], "2810": ["nroi", 324], "6857": ["ndet", 81, 28, 28]}


def _between(g, start, ends):
    """Node indices that lie on a path start -> any of ends."""
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}
    back, stack = set(), list(ends)
    while stack:
        t = stack.pop()
        i = prod.get(t)
        if i is None or i in back:
            continue
        back.add(i)
        stack.extend(g.node[i].input)
    cons = {}
    for i, n in enumerate(g.node):
        for t in n.input:
            cons.setdefault(t, []).append(i)
    fwd, stack = set(), [start]
    while stack:
        t = stack.pop()
        for i in cons.get(t, []):
            if i not in fwd:
                fwd.add(i)
                stack.extend(g.node[i].output)
    return back & fwd


def _pin(src, dst, dims):
    m = onnx.load(src)
    for i in m.graph.input:
        for k, v in enumerate(dims[i.name]):
            d = i.type.tensor_type.shape.dim[k]
            d.ClearField("dim_param")
            d.dim_value = v
    for o in m.graph.output:
        o.type.tensor_type.ClearField("shape")
    onnx.save(onnx.shape_inference.infer_shapes(m), dst)


def _cut(rest, dst, removed, new_inputs, outputs):
    """rest.onnx minus the given nodes, with the removed nodes' outputs turned into inputs."""
    m = onnx.load(rest)
    g = m.graph
    keep = [n for i, n in enumerate(g.node) if i not in removed]
    # one Shape node outside the mask head reads the pre-sigmoid logits (6856); it only needs the
    # shape, which 6857 (the head's real output) shares, so point it there
    for n in keep:
        if n.op_type == "Shape" and list(n.input) == ["6856"] and "6857" in new_inputs:
            n.input[0] = "6857"
    del g.node[:]
    g.node.extend(keep)
    for name in new_inputs:
        g.input.append(
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, SHAPES[name]
            )
        )
    tmp = dst + ".tmp.onnx"
    onnx.save(m, tmp)
    extract_model(tmp, dst, [i.name for i in g.input], outputs)
    Path(tmp).unlink()


def make(rest, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = onnx.load(rest).graph
    nroi = 1000  # post-NMS proposals per image at this input size (rpn post_nms_top_n)
    extract_model(rest, str(out / "box_head_dyn.onnx"), [BOX_IN], BOX_OUT)
    _pin(
        str(out / "box_head_dyn.onnx"),
        str(out / f"box_head_{nroi}.onnx"),
        {BOX_IN: [nroi, 256, 7, 7]},
    )
    extract_model(rest, str(out / "mask_head_dyn.onnx"), [MASK_IN], MASK_OUT)
    _pin(
        str(out / "mask_head_dyn.onnx"),
        str(out / f"mask_head_{MASK_BATCH}.onnx"),
        {MASK_IN: [MASK_BATCH, 256, 14, 14]},
    )
    box_nodes = _between(g, BOX_IN, BOX_OUT)
    mask_nodes = _between(g, MASK_IN, MASK_OUT)
    ins = [i.name for i in g.input]
    extract_model(
        rest, str(out / "cpu_a.onnx"), ins, [BOX_IN]
    )  # proposals -> box RoiAlign
    _cut(
        rest, str(out / "cpu_b.onnx"), box_nodes, BOX_OUT, [MASK_IN]
    )  # box post-proc -> mask RoiAlign
    _cut(
        rest, str(out / "cpu_c.onnx"), box_nodes | mask_nodes, BOX_OUT + MASK_OUT, FINAL
    )
    print(
        f"box head {len(box_nodes)} nodes, mask head {len(mask_nodes)} nodes -> {out}"
    )


def _run(model, feeds):
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    return s.run(None, {n: feeds[n] for n in names})


def _host_head(model, inp, x):
    return _run(model, {inp: x})


def _phone_head(model, inp, x, work):
    """Run a head model strictly on the phone's HTP via run_multi.sh; returns its outputs."""
    import subprocess

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    xb = work / f"{inp}.bin"
    x.astype(np.float32).tofile(xb)
    man = work / f"{inp}.man"
    man.write_text(f"{inp} f32 {xb} {','.join(map(str, x.shape))}\n")
    here = Path(__file__).resolve().parent
    log = subprocess.run(
        [str(here / "run_multi.sh"), model, str(man), "htp", "1"],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "QNN_EXTRA": "htp_graph_finalization_optimization_mode=3",
        },
    )
    outs = [line.split() for line in log.stdout.splitlines() if line.startswith("out ")]
    if "PASS" not in log.stdout:
        raise RuntimeError(log.stdout + log.stderr)
    res = []
    for _, i, _name, _dt, shape in outs:
        f = work / f"{Path(model).stem}_o{i}.bin"
        subprocess.run(
            [
                "adb",
                "-s",
                "239dbd8f",
                "pull",
                f"/data/local/tmp/qnn_rest/out_htp_o{i}.bin",
                str(f),
            ],
            check=True,
            capture_output=True,
        )
        res.append(
            np.fromfile(f, np.float32).reshape(
                [int(d) for d in shape.strip(",").split(",")]
            )
        )
    return res


def stitch(out, feats_npz, where="host", work=None):
    """CPU pieces on host ORT; heads on host ORT (where="host") or the phone's HTP ("phone")."""
    out = Path(out)
    z = np.load(feats_npz)
    head = (
        _host_head if where == "host" else (lambda m, i, x: _phone_head(m, i, x, work))
    )
    rest_inputs = [i.name for i in onnx.load(str(out / "cpu_a.onnx")).graph.input]
    feeds = {n: z[f"o{k}"] for k, n in enumerate(rest_inputs)}
    (roi,) = _run(str(out / "cpu_a.onnx"), feeds)
    scores, deltas = head(str(out / "box_head_1000.onnx"), BOX_IN, roi)
    feeds.update({"2811": scores, "2810": deltas})
    (mroi,) = _run(str(out / "cpu_b.onnx"), feeds)
    ndet = mroi.shape[0]
    (mask,) = head(
        str(out / f"mask_head_{MASK_BATCH}.onnx"),
        MASK_IN,
        np.pad(mroi, ((0, MASK_BATCH - ndet), (0, 0), (0, 0), (0, 0))),
    )
    feeds["6857"] = mask[:ndet]
    return _run(str(out / "cpu_c.onnx"), feeds)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(*sys.argv[2:4])
    else:
        res = stitch(*sys.argv[2:6])
        print([r.shape for r in res])
