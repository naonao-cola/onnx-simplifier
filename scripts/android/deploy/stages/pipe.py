"""Write the pipeline description runtime/pipe_run reads (the `pipe` stage).

Output: work/<name>/pipe/
  pipe.txt            header (`input`, `output`) + one step per line, same grammar as
                      e2e_pipeline's pipe_<stage>.txt files (see runtime/pipe_run.cpp)
  <name>.onnx         the HTP model (rewrite stage output)
  <name>_post.onnx    CPU post-processing, if the spec has `postprocess:`
  inputs/<stem>.bin   eval images, preprocessed exactly as the runtime expects them
  inputs/<stem>.json  preprocessing meta (letterbox scale/pad) for mapping results back
  pipe_meta.json      everything the accuracy stage needs to rebuild the fp32 reference

spec `pipeline:` keys:
  qnn: {k: v}        QNN EP provider options for the HTP session (htp_performance_mode, ...)
  finalize_mode: N   htp_graph_finalization_optimization_mode for this model: set per session,
                     because #1833 measured 3 as slower for the Mask R-CNN backbone while #1832
                     needed it for the box head
  ctx: true          load (and compile once on device) an EP-context model
  host_quantize: true  inputs are handed over already quantized to uint8 NHWC (what a camera
                     pipeline that emits uint8 would do); default false quantizes on the phone
                     with a `quant_in` step, which is then part of the timed total
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import images as imglib


def prebuilt_files(ctx) -> list:
    """The pipe file of a prebuilt pipeline + every file in its dir the steps name (models, and the
    DSP RPN's constants: levels.txt, model.txt, l*_anchors.bin, read by the runtime from its cwd)."""
    pb = ctx.prebuilt
    src = Path(pb["dir"])
    text = (src / pb["pipe"]).read_text()
    names = {w for line in text.splitlines() for w in line.split()}
    names |= {w.split(":", 1)[1] for n in list(names) for w in n.split(",") if ":" in w}  # ortpad buckets
    names |= set(pb.get("extra_files", []))
    files = [src / pb["pipe"], *sorted(src / n for n in names if (src / n).is_file() and n != pb["pipe"])]
    if pb.get("dsp"):
        files += sorted(src.glob("l*_anchors.bin")) + [src / "levels.txt", src / "model.txt"]
    return list(dict.fromkeys(files))


def write_prebuilt(ctx, d) -> None:
    pb = ctx.prebuilt
    for f in d.glob("*"):
        if f.is_symlink() or f.is_file():
            f.unlink()
    files = prebuilt_files(ctx)
    for f in files[1:]:
        (d / f.name).symlink_to(f.resolve())
    shutil.copyfile(files[0], d / "pipe.txt")
    inp = d / "inputs"
    shutil.rmtree(inp, ignore_errors=True)
    inp.mkdir()
    stems = []
    for f in sorted(Path(pb["inputs"]).glob("*.bin")):
        (inp / f.name).symlink_to(f.resolve())
        stems.append(f.stem)
    (d / "pipe_meta.json").write_text(json.dumps({
        "prebuilt": True, "inputs": stems, "files": [f.name for f in files[1:]], "dsp": bool(pb.get("dsp")),
        "ref": str(Path(pb["inputs"]) / "ref")}, indent=1))
    print(f"  pipe.txt = {pb['pipe']} ({len(files) - 1} files), {len(stems)} inputs")


def write(ctx, d) -> None:
    if ctx.prebuilt:
        return write_prebuilt(ctx, d)
    sp = ctx.spec
    p = sp.get("pipeline", {}) or {}
    rw_dir = ctx.work / "rewrite"
    rw = json.loads((rw_dir / "rewrite_meta.json").read_text())
    post_meta = json.loads((ctx.work / "post" / "post_meta.json").read_text())
    name = ctx.name
    shutil.copyfile(rw_dir / "model.onnx", d / f"{name}.onnx")
    import onnx

    m = onnx.load(str(d / f"{name}.onnx"), load_external_data=False)
    model_outs = [o.name for o in m.graph.output]
    (in_name,) = list(sp["inputs"])
    shape = sp["inputs"][in_name]["shape"]
    u8 = rw.get("uint8_input")

    qnn = dict(p.get("qnn", {}) or {})
    if "finalize_mode" in p:
        qnn["htp_graph_finalization_optimization_mode"] = str(p["finalize_mode"])
    if p.get("ctx"):
        qnn["ctx"] = "1"
    qopts = ";".join(f"{k}={v}" for k, v in qnn.items()) or "-"

    lines, host_q = [], bool(p.get("host_quantize")) and u8 is not None
    c, h, w = shape[1:]
    if u8 is None:
        lines.append(f"input {in_name} f32 {c},{h},{w}")
        # the runtime reads a CHW tensor; the model takes NCHW with batch 1: same bytes
        net_in = in_name
    elif host_q:
        lines.append(f"input {u8['name']} u8 {','.join(map(str, u8['shape']))}")
        net_in = u8["name"]
    else:
        if u8["layout"] != "nhwc":
            raise SystemExit("pipe: on-device quant_in produces NHWC; use uint8_input layout nhwc")
        lines.append(f"input image f32 {c},{h},{w}")
        lines.append(f"quant_in image {u8['name']} {u8['scale']!r} {u8['zero_point']}")
        net_in = u8["name"]
    outs = post_meta.get("outputs") or model_outs
    lines.insert(1 if not lines[0].startswith("input") else 1, f"output {','.join(outs)}")
    engine = p.get("engine", "htp")
    lines.append(f"ort {name} {name}.onnx {engine} {qopts if engine == 'htp' else '-'} {net_in} {','.join(model_outs)}")
    if post_meta:
        shutil.copyfile(ctx.work / "post" / "post.onnx", d / f"{name}_post.onnx")
        lines.append(f"ort post {name}_post.onnx cpu - {post_meta['input']} {','.join(outs)}")
    (d / "pipe.txt").write_text("\n".join(lines) + "\n")

    inp = d / "inputs"
    shutil.rmtree(inp, ignore_errors=True)
    inp.mkdir()
    stems = []
    for f in imglib.list_images(sp.get("eval", {}), ctx.images):
        x, meta = imglib.preprocess(f, sp["preprocess"])
        stem = f.stem
        if host_q:
            q = np.clip(np.rint(x / u8["scale"]) + u8["zero_point"], 0, 255).astype(np.uint8)
            (q.transpose(1, 2, 0) if u8["layout"] == "nhwc" else q).tofile(inp / f"{stem}.bin")
        else:
            x.astype(np.float32).tofile(inp / f"{stem}.bin")
        (inp / f"{stem}.json").write_text(json.dumps({**meta, "file": str(f)}))
        stems.append(stem)
    (d / "pipe_meta.json").write_text(json.dumps({
        "model": f"{name}.onnx", "post": f"{name}_post.onnx" if post_meta else None, "outputs": outs,
        "inputs": stems, "host_quantize": host_q, "uint8_input": u8}, indent=1))
    print("  pipe.txt:\n    " + "\n    ".join(lines))
