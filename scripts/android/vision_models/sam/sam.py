#!/usr/bin/env python3
"""Segment-Anything variants on the phone's HTP: export, quantize, host accuracy, phone runs.

  sam.py export   <variant>   torch -> enc.onnx / dec.onnx (fp32) -> onnxsim; torch-vs-ORT check;
                               enc.fp16.onnx (uint8 NHWC pixels in, the HTP runs the rest in fp16)
  sam.py ref      <variant>   fp32 ORT CPU over the calibration + eval images and prompts
  sam.py quantize <variant> [--policy int8|mix]   enc/dec int8 QDQ (onnxsim.full_qdq), uint8 I/O
  sam.py host     <variant>   host int8 accuracy vs fp32 (ORT CPU, graph optimizations off)
  sam.py phone    <variant>   strict all-HTP runs (fp16 + int8 encoder/decoder, CPU decoder) under
                               the phone lock: medians + outputs of the eval set -> phone accuracy
  sam.py report               every variant's results.json -> the README table (markdown)

Work dir: $SAM_WORK (default ~/.cache/onnxsim-sam)/<variant>/. Images: the deploy pipeline's
COCO val2017 cache (the first 16 yolo11n calibration ids, the first 10 eval ids). Prompts per
eval image: three foreground points and one box, fixed fractions of the image (see prompts()).
Mask accuracy = IoU of the thresholded 256x256 low-res mask inside the valid (unpadded) area,
same mask slot as the fp32 reference picks (box: slot 0; point: best fp32 iou among 1..3).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANDROID = HERE.parents[1]
REPO = ANDROID.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ANDROID / "deploy"))
WORK = Path(os.environ.get("SAM_WORK", Path.home() / ".cache/onnxsim-sam"))
IMAGES = Path(os.environ.get("SAM_IMAGES", Path.home() / ".cache/onnxsim-deploy/_images"))
CALIB_IDS = [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503]
EVAL_IDS = [7088, 7108, 7278, 7281, 7386, 7511, 7574, 7784, 7795, 7816]
PROMPT_SIZE = 1024
PAD = [124, 116, 104]  # round(SAM pixel mean): pads to ~0 after normalization, like SAM's zero pad


def wdir(v):
    d = WORK / v
    d.mkdir(parents=True, exist_ok=True)
    return d


def ort_sess(path, opt=True, threads=0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if not opt:  # fused int8 kernels saturate on non-VNNI x86; run the reference QDQ math
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- images and prompts
def image(img_id, size):
    """-> uint8 [size, size, 3] (resize longest side, pad bottom/right), valid (h, w) in the
    1024 prompt frame."""
    from PIL import Image
    from stages import images as imglib

    imglib.fetch_images({"coco_val2017": [img_id]}, IMAGES)
    im = Image.open(IMAGES / f"coco_{img_id:012d}.jpg").convert("RGB")
    w0, h0 = im.size
    r = size / max(h0, w0)
    nh, nw = int(h0 * r + 0.5), int(w0 * r + 0.5)
    out = np.empty((size, size, 3), np.uint8)
    out[:] = PAD
    out[:nh, :nw] = np.asarray(im.resize((nw, nh), Image.BILINEAR))
    r2 = PROMPT_SIZE / max(h0, w0)
    return out, (int(h0 * r2 + 0.5), int(w0 * r2 + 0.5))


def prompts(valid):
    h, w = valid
    ps = []
    for fx, fy in ((0.5, 0.5), (0.3, 0.35), (0.7, 0.65)):
        ps.append(("point", np.array([[[fx * w, fy * h], [0, 0]]], np.float32),
                   np.array([[1, -1]], np.float32)))
    ps.append(("box", np.array([[[0.25 * w, 0.25 * h], [0.75 * w, 0.75 * h]]], np.float32),
               np.array([[2, 3]], np.float32)))
    return ps


def nchw(u8):
    return np.ascontiguousarray(u8.transpose(2, 0, 1)[None].astype(np.float32))


def mask_slot(kind, iou):
    return 0 if kind == "box" else 1 + int(np.argmax(iou[0, 1:]))


def mask_iou(ref_masks, got_masks, slot, valid):
    h, w = (valid[0] + 3) // 4, (valid[1] + 3) // 4
    a = ref_masks[0, slot, :h, :w] > 0
    b = got_masks[0, slot, :h, :w] > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------- export
def u8_nhwc_input(src, dst, size):
    """fp32 NCHW pixels input -> uint8 NHWC `pixels_u8` + DequantizeLinear(scale 1, zero point 0)
    + Transpose in the graph. Not Cast: on the HTP (QNN 2.50, fp16 graph) a uint8 -> float Cast
    as the first node gives wrong values (MobileSAM's stem conv cos 0.81 vs fp32, the embedding
    0.45-0.82); the DequantizeLinear form matches fp32 to cos 0.99999 at every probed tensor."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(src)
    g = m.graph
    (x,) = [i for i in g.input if i.name == "pixels"]
    g.input.remove(x)
    g.input.insert(0, helper.make_tensor_value_info("pixels_u8", TensorProto.UINT8,
                                                    [1, size, size, 3]))
    g.initializer.extend([numpy_helper.from_array(np.array(1.0, np.float32), "pixels_u8_scale"),
                          numpy_helper.from_array(np.array(0, np.uint8), "pixels_u8_zp")])
    g.node.insert(0, helper.make_node("Transpose", ["pixels_f"], ["pixels"], perm=[0, 3, 1, 2]))
    g.node.insert(0, helper.make_node("DequantizeLinear",
                                      ["pixels_u8", "pixels_u8_scale", "pixels_u8_zp"],
                                      ["pixels_f"]))
    onnx.save(m, dst)


def simplify(src, dst):
    import onnx

    import onnxsim

    # no com.microsoft::Attention: the QNN EP does not take it (the node would run on the CPU)
    m, ok = onnxsim.simplify(onnx.load(src), skipped_optimizers=["fuse_attention"])
    assert ok, src
    odd = {(x.domain, x.op_type) for x in m.graph.node if x.domain not in ("", "ai.onnx")}
    assert not odd, f"{src}: non-standard ops after onnxsim: {odd}"
    onnx.save(m, dst)
    Path(src).unlink()


def cmd_export(a):
    import torch
    import variants

    d = wdir(a.variant)
    p = variants.load(a.variant)
    if a.gelu != "exact" or a.upsample:  # an approximated encoder next to the exact export
        if a.gelu != "exact":
            n, tag = variants.set_gelu(p.encoder, a.gelu), f"gelu_{a.gelu}"
        else:
            n, tag = variants.set_upsample(p.encoder, a.upsample), f"up_{a.upsample}"
        enc = variants.Encoder(p).eval()
        x = torch.from_numpy(nchw(image(EVAL_IDS[0], p.size)[0]))
        torch.onnx.export(enc, (x,), d / "enc_g.onnx", input_names=["pixels"],
                          output_names=["image_embeddings"], opset_version=20, dynamo=False)
        simplify(d / "enc_g.onnx", d / f"enc.{tag}.onnx")
        fp16 = f"fp16_{tag.split('_', 1)[1]}"
        u8_nhwc_input(d / f"enc.{tag}.onnx", d / f"enc.{fp16}.onnx", p.size)
        print(f"{n} modules -> {tag}: enc.{fp16}.onnx")
        return
    enc, dec = variants.Encoder(p).eval(), variants.SamDecoder(p).eval()
    u8, valid = image(EVAL_IDS[0], p.size)
    x = torch.from_numpy(nchw(u8))
    kind, pc, pl = prompts(valid)[0]
    with torch.no_grad():
        emb = enc(x)
        iou, masks = dec(emb, torch.from_numpy(pc), torch.from_numpy(pl))
        with variants.upstream():  # our rank <= 4 rewrites vs the unpatched model
            emb_up = enc(x)
        iou_up, masks_up = dec.forward_upstream(emb, torch.from_numpy(pc), torch.from_numpy(pl))
    rewrite_check = {"enc_maxabs": float((emb - emb_up).abs().max()),
                     "dec_masks_maxabs": float((masks - masks_up).abs().max()),
                     "dec_iou_maxabs": float((iou - iou_up).abs().max())}
    print("rewrite vs upstream", rewrite_check)
    t = time.time()
    torch.onnx.export(enc, (x,), d / "enc.onnx", input_names=["pixels"],
                      output_names=["image_embeddings"], opset_version=20, dynamo=False)
    torch.onnx.export(dec, (emb, torch.from_numpy(pc), torch.from_numpy(pl)), d / "dec.onnx",
                      input_names=["image_embeddings", "point_coords", "point_labels"],
                      output_names=["iou_predictions", "low_res_masks"], opset_version=20,
                      dynamo=False)
    print(f"export {time.time() - t:.1f} s")
    for n in ("enc", "dec"):
        simplify(d / f"{n}.onnx", d / f"{n}.sim.onnx")
    e = ort_sess(d / "enc.sim.onnx").run(None, {"pixels": x.numpy()})[0]
    i2, m2 = ort_sess(d / "dec.sim.onnx").run(None, {"image_embeddings": emb.numpy(),
                                                      "point_coords": pc, "point_labels": pl})
    u8_nhwc_input(d / "enc.sim.onnx", d / "enc.fp16.onnx", p.size)
    info = {"size": p.size, "source": p.source,
            "enc_mb": round((d / "enc.sim.onnx").stat().st_size / 2**20, 1),
            "dec_mb": round((d / "dec.sim.onnx").stat().st_size / 2**20, 1),
            "rewrite_check": rewrite_check,
            "export_check": {"enc_cos": cos(emb.numpy(), e), "dec_masks_cos": cos(masks.numpy(), m2),
                             "dec_iou_maxabs": float(np.abs(iou.numpy() - i2).max())}}
    (d / "export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


# ---------------------------------------------------------------- fp32 reference
def cmd_ref(a):
    d = wdir(a.variant)
    size = json.loads((d / "export.json").read_text())["size"]
    enc, dec = ort_sess(d / "enc.sim.onnx"), ort_sess(d / "dec.sim.onnx")
    r = d / "ref"
    r.mkdir(exist_ok=True)
    for tag, ids in (("calib", CALIB_IDS), ("eval", EVAL_IDS)):
        for i in ids:
            u8, valid = image(i, size)
            np.save(r / f"{tag}_{i}_img.npy", u8)
            emb = enc.run(None, {"pixels": nchw(u8)})[0]
            np.save(r / f"{tag}_{i}_emb.npy", emb)
            for k, (kind, pc, pl) in enumerate(prompts(valid)):
                iou, masks = dec.run(None, {"image_embeddings": emb, "point_coords": pc,
                                            "point_labels": pl})
                if tag == "eval":
                    np.savez(r / f"eval_{i}_p{k}.npz", iou=iou, masks=masks, pc=pc, pl=pl,
                             valid=np.array(valid))
        print(tag, "done")


# ---------------------------------------------------------------- quantization
MIX_FLOAT = ["LayerNormalization", "Softmax", "Erf", "Gelu", "Div", "Pow", "Sqrt", "ReduceMean"]


def cmd_quantize(a):
    import onnx

    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    d = wdir(a.variant)
    r = d / "ref"
    tag = a.policy + ("" if a.method == "minmax" else "_" + a.method)
    kw = {"method": a.method}
    if a.policy == "mix":
        kw["exclude_op_types"] = MIX_FLOAT
    if a.policy == "a16":
        kw["activation_dtype"] = "uint16"
    if a.policy in ("dw8", "dw16"):  # depthwise convs (reparameterized kernels) stay fp16
        enc_m = onnx.load(d / "enc.sim.onnx")
        kw["exclude_nodes"] = [n.name for n in enc_m.graph.node if n.op_type == "Conv" and any(
            at.name == "group" and at.i > 1 for at in n.attribute)]
        if a.policy == "dw16":
            kw["activation_dtype"] = "uint16"
        print(f"{len(kw['exclude_nodes'])} depthwise/grouped convs kept in float")
    if a.policy == "stem8":  # int8 only for the conv stem + first stage, the rest stays fp16
        kw["exclude_nodes"] = [n.name for n in onnx.load(d / "enc.sim.onnx").graph.node
                               if not re.match(r"/(Sub|Mul|enc/patch_embed/|enc/layers\.0/)",
                                               n.name)]
    t = time.time()
    data = [{"pixels": nchw(np.load(r / f"calib_{i}_img.npy"))} for i in CALIB_IDS[: a.calib]]
    q = quantize_full_qdq(onnx.load(d / "enc.sim.onnx"), data, ranges={"pixels": (0.0, 255.0)},
                          **kw)
    q, io = quantized_io(q, inputs=["pixels"], outputs=["image_embeddings"],
                         nhwc_inputs=["pixels"])
    onnx.save(q, d / f"enc.{tag}.onnx")
    io_all = {"enc": io}
    print(f"enc {tag}: {time.time() - t:.0f} s {io}")
    if a.policy in ("dw8", "dw16"):  # depthwise convs (reparameterized kernels) stay fp16
        enc_m = onnx.load(d / "enc.sim.onnx")
        kw["exclude_nodes"] = [n.name for n in enc_m.graph.node if n.op_type == "Conv" and any(
            at.name == "group" and at.i > 1 for at in n.attribute)]
        if a.policy == "dw16":
            kw["activation_dtype"] = "uint16"
        print(f"{len(kw['exclude_nodes'])} depthwise/grouped convs kept in float")
    if a.policy in ("stem8", "dw8", "dw16"):  # encoder-only policies
        (d / f"quant_{tag}.json").write_text(json.dumps({**io_all, "dec": None}, indent=1,
                                                       default=str))
        return
    ddata = []
    for i in CALIB_IDS[: a.calib]:
        emb = np.load(r / f"calib_{i}_emb.npy")
        valid = (PROMPT_SIZE, PROMPT_SIZE)  # prompts spread over the frame for calibration
        for kind, pc, pl in prompts(valid):
            ddata.append({"image_embeddings": emb, "point_coords": pc, "point_labels": pl})
    # coordinates stay float: their range sets the positional encoding's frequency phase
    q = quantize_full_qdq(onnx.load(d / "dec.sim.onnx"), ddata, **kw)
    q, io = quantized_io(q, inputs=["image_embeddings"], outputs=[])
    onnx.save(q, d / f"dec.{tag}.onnx")
    io_all["dec"] = io
    (d / f"quant_{tag}.json").write_text(json.dumps(io_all, indent=1, default=str))
    print(f"dec {tag}: {io}")


def q_u8(x, io):
    dt = np.dtype(io.get("dtype", "uint8"))
    q = np.round(x / io["scale"]) + io["zero_point"]
    return np.clip(q, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)


def dq_u8(x, io):
    return (x.astype(np.float32) - io["zero_point"]) * io["scale"]


# ---------------------------------------------------------------- host accuracy
def eval_masks(d, get_emb, run_dec):
    """-> per-prompt IoUs and embedding cosines vs the fp32 reference."""
    r = d / "ref"
    ious, coss = [], []
    for i in EVAL_IDS:
        emb_ref = np.load(r / f"eval_{i}_emb.npy")
        emb = get_emb(i, emb_ref)
        coss.append(cos(emb_ref, emb))
        for k in range(4):
            z = np.load(r / f"eval_{i}_p{k}.npz")
            iou, masks = run_dec(emb, z["pc"], z["pl"], i, k)
            kind = "box" if k == 3 else "point"
            ious.append(mask_iou(z["masks"], masks, mask_slot(kind, z["iou"]), tuple(z["valid"])))
    return {"emb_cos_min": min(coss), "emb_cos_mean": float(np.mean(coss)),
            "mask_iou_mean": float(np.mean(ious)), "mask_iou_min": min(ious), "n": len(ious)}


def cmd_host(a):
    d = wdir(a.variant)
    res = {}
    dec_f = ort_sess(d / "dec.sim.onnx")

    def dec_fp32(emb, pc, pl, i, k):
        return dec_f.run(None, {"image_embeddings": emb, "point_coords": pc, "point_labels": pl})

    # approximated encoders (GELU form, upsample mode), fp32: the approximation's own cost
    for g in sorted([*d.glob("enc.gelu_*.onnx"), *d.glob("enc.up_*.onnx")]):
        enc_g = ort_sess(g)
        def enc_gelu(i, _ref, s=enc_g):
            return s.run(None, {"pixels": nchw(np.load(d / "ref" / f"eval_{i}_img.npy"))})[0]

        res[g.stem[len("enc."):]] = eval_masks(d, enc_gelu, dec_fp32)
        print(g.stem, res[g.stem[len("enc."):]])
    for pol in tags(d):
        io = json.loads((d / f"quant_{pol}.json").read_text())
        enc_q = ort_sess(d / f"enc.{pol}.onnx", opt=False)
        eio, pio = io["enc"].get("image_embeddings"), io["enc"]["pixels"]

        def enc_int8(i, _ref):
            u8 = np.load(d / "ref" / f"eval_{i}_img.npy")[None]
            out = enc_q.run(None, {"pixels": q_u8(u8.astype(np.float32), pio)})[0]
            return dq_u8(out, eio) if eio else out

        res[pol] = {"enc_int8+dec_fp32": eval_masks(d, enc_int8, dec_fp32)}
        if io["dec"]:
            dec_q, dio = ort_sess(d / f"dec.{pol}.onnx", opt=False), io["dec"]["image_embeddings"]

            def dec_int8(emb, pc, pl, i, k):
                return dec_q.run(None, {"image_embeddings": q_u8(emb, dio), "point_coords": pc,
                                        "point_labels": pl})

            res[pol]["enc_fp32+dec_int8"] = eval_masks(d, lambda i, e: e, dec_int8)
            res[pol]["enc_int8+dec_int8"] = eval_masks(d, enc_int8, dec_int8)
        print(pol, json.dumps(res[pol], indent=1))
    upd(d, {"host": res})


def tags(d):
    return sorted(p.stem[len("quant_"):] for p in d.glob("quant_*.json"))


def upd(d, new):
    p = d / "results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(new)
    p.write_text(json.dumps(cur, indent=1))


# ---------------------------------------------------------------- phone
def phone_run(d, name, model, mode, sets, iters, extra_env=""):
    """Run `model` on the phone for each input set (list of [(name, dtype, array)]); the first set
    is timed over `iters` runs, the rest once. -> (log text, [[outputs] per set])."""
    loc = d / "phone" / name
    loc.mkdir(parents=True, exist_ok=True)
    mans = []
    for s, inputs in enumerate(sets):
        lines = []
        for n, dt, arr in inputs:
            f = loc / f"s{s}_{n}.bin"
            np.ascontiguousarray(arr).tofile(f)
            lines.append(f"{n} {dt} {f.name} {','.join(map(str, arr.shape))}")
        (loc / f"m{s}.txt").write_text("\n".join(lines) + "\n")
        mans.append(f"m{s}.txt")
    env = dict(os.environ, PHONE_LOCK_OWNER="codex/android-sam-variants", EXTRA_ENV=extra_env)
    t = time.time()
    out = subprocess.run([str(Path.home() / ".cache/android-phone/phone-run"),
                          str(HERE / "phone.sh"), str(model), mode, str(iters), str(loc), name,
                          *mans], env=env, capture_output=True, text=True)
    log = out.stdout + out.stderr
    (loc / "log.txt").write_text(log)
    print(f"  phone {name} {mode}: {time.time() - t:.0f} s wall")
    return log, loc


def outs_of(loc, s, log_part):
    outs = re.findall(r"^out (\d+) (\S+) (\S+) (\S+)", log_part, re.M)
    res = []
    for i, n, dt, shape in outs:
        arr = np.fromfile(loc / f"out{s}_o{i}.bin",
                          {"f32": np.float32, "u8": np.uint8, "u16": np.uint16}[dt])
        res.append(arr.reshape([int(x) for x in shape.split(",") if x] or [-1]))
    return res


def parse(log):
    parts = re.split(r"^=== set (\d+)$", log, flags=re.M)
    sets = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    return sets


def med(txt):
    m = re.search(r"median_ms ([\d.]+)", txt)
    return float(m.group(1)) if m else None


def cmd_phone(a):
    d = wdir(a.variant)
    r = d / "ref"
    res = {}
    size = json.loads((d / "export.json").read_text())["size"]
    imgs = [np.load(r / f"eval_{i}_img.npy")[None] for i in EVAL_IDS[: a.images]]
    pols = [p for p in tags(d) if not a.tags or p in a.tags.split(",")]
    # --- encoder: fp16 and int8 policies, strict all-HTP
    embs = {}
    fp16s = sorted(p.stem[len("enc."):] for p in d.glob("enc.fp16*.onnx"))
    if a.tags:
        fp16s = [p for p in fp16s if p in a.tags.split(",") or p == "fp16"]
    for prec in [*fp16s, *pols] if a.pieces in ("all", "enc") else []:
        model = d / f"enc.{prec}.onnx"
        eio = None
        if prec.startswith("fp16"):
            sets = [[("pixels_u8", "u8", im)] for im in imgs]
        else:  # quantized input: scale 1 / zero point 0 for uint8 (the pixels as-is), not uint16
            qio = json.loads((d / f"quant_{prec}.json").read_text())["enc"]
            eio, pio = qio.get("image_embeddings"), qio["pixels"]
            sets = [[("pixels", "u16" if pio["dtype"] == "uint16" else "u8",
                      q_u8(im.astype(np.float32), pio))] for im in imgs]
        log, loc = phone_run(d, f"enc_{prec}", model, "htp", sets, a.iters)
        ps = parse(log)
        ok = all("PASS" in ps.get(s, "") for s in range(len(sets)))
        ent = {"median_ms": med(ps.get(0, "")), "strict_htp": ok,
               "compile_ms": float(m.group(1)) if (m := re.search(r"compile_ms ([\d.]+)", log))
               else None}
        if ok:
            got = [outs_of(loc, s, ps[s])[0] for s in range(len(sets))]
            if eio:
                got = [dq_u8(g, eio) for g in got]
            embs[prec] = [g.reshape(1, 256, 64, 64) for g in got]
            ent["emb_cos_min"] = min(cos(np.load(r / f"eval_{i}_emb.npy"), g)
                                     for i, g in zip(EVAL_IDS, embs[prec]))
        else:
            ent["error"] = "\n".join(log.splitlines()[-5:])
        res[f"enc_{prec}"] = ent
        print(prec, ent)
    # --- decoder: fp16 / int8 on the HTP, fp32 on the CPU (4 threads); inputs = fp32 ref embeddings
    dsets = []
    for i in EVAL_IDS[: a.images]:
        for k in range(4):
            z = np.load(r / f"eval_{i}_p{k}.npz")
            dsets.append((i, k, np.load(r / f"eval_{i}_emb.npy"), z["pc"], z["pl"]))
    decs = (("fp16", d / "dec.sim.onnx", "htp", ""),
            *[(p, d / f"dec.{p}.onnx", "htp", "") for p in pols if (d / f"dec.{p}.onnx").exists()],
            ("cpu4", d / "dec.sim.onnx", "cpu", "ORT_THREADS=4"))
    for prec, model, mode, env in decs if a.pieces in ("all", "dec") else ():
        io = (json.loads((d / f"quant_{prec}.json").read_text())["dec"]["image_embeddings"]
              if prec in pols else None)
        sets = [[("image_embeddings", ("u16" if "a16" in prec else "u8") if io else "f32",
                  q_u8(e, io) if io else e),
                 ("point_coords", "f32", pc), ("point_labels", "f32", pl)]
                for _, _, e, pc, pl in dsets]
        log, loc = phone_run(d, f"dec_{prec}", model, mode, sets, a.iters, env)
        ps = parse(log)
        ok = all("PASS" in ps.get(s, "") for s in range(len(sets)))
        ent = {"median_ms": med(ps.get(0, "")), "strict_htp" if mode == "htp" else "ok": ok}
        if ok:
            ious = []
            for s, (i, k, *_rest) in enumerate(dsets):
                iou, masks = outs_of(loc, s, ps[s])
                z = np.load(r / f"eval_{i}_p{k}.npz")
                kind = "box" if k == 3 else "point"
                ious.append(mask_iou(z["masks"], masks.reshape(1, 4, 256, 256),
                                     mask_slot(kind, z["iou"]), tuple(z["valid"])))
            ent["mask_iou_mean"], ent["mask_iou_min"] = float(np.mean(ious)), min(ious)
        else:
            ent["error"] = "\n".join(log.splitlines()[-5:])
        res[f"dec_{prec}"] = ent
        print(prec, ent)
    # --- chain: phone encoder embeddings -> fp32 host decoder (the encoder's mask cost)
    dec_f = ort_sess(d / "dec.sim.onnx")
    for prec, el in embs.items():
        ious = []
        for i, emb in zip(EVAL_IDS, el):
            for k in range(4):
                z = np.load(r / f"eval_{i}_p{k}.npz")
                iou, masks = dec_f.run(None, {"image_embeddings": emb.astype(np.float32),
                                              "point_coords": z["pc"], "point_labels": z["pl"]})
                kind = "box" if k == 3 else "point"
                ious.append(mask_iou(z["masks"], masks, mask_slot(kind, z["iou"]),
                                     tuple(z["valid"])))
        res[f"enc_{prec}"]["mask_iou_mean"] = float(np.mean(ious))
        res[f"enc_{prec}"]["mask_iou_min"] = min(ious)
    res["size"] = size
    p = d / "results.json"
    upd(d, {"phone": {**(json.loads(p.read_text()).get("phone", {}) if p.exists() else {}),
                      **res}})
    print(json.dumps(res, indent=1))


def cmd_report(a):
    """Markdown tables from every variant's results.json (phone medians, accuracy vs fp32)."""
    rows, qrows = [], []
    for d in sorted(p for p in WORK.iterdir() if (p / "results.json").exists()):
        r = json.loads((d / "results.json").read_text())
        ph, ex = r.get("phone", {}), json.loads((d / "export.json").read_text())
        e16, d16, dc = ph.get("enc_fp16", {}), ph.get("dec_fp16", {}), ph.get("dec_cpu4", {})
        first = (e16.get("median_ms") or 0) + (d16.get("median_ms") or 0)
        rows.append(f"| {d.name} | {ex['size']} | {ex['enc_mb']} | {_ms(e16)} | "
                    f"{_f(e16.get('emb_cos_min'), 5)} / {_f(e16.get('mask_iou_mean'), 3)} | "
                    f"{_ms(d16)} | {_ms(dc)} | {first:.0f} ms |")
        for k, v in ph.items():
            if k.startswith("enc_") and k != "enc_fp16" and isinstance(v, dict):
                qrows.append(f"| {d.name} | {k[4:]} | {_ms(v)} | {_f(v.get('emb_cos_min'), 4)} | "
                             f"{_f(v.get('mask_iou_mean'), 3)} / {_f(v.get('mask_iou_min'), 3)} |")
    print("| variant | input | enc MB | enc fp16 HTP | emb cos min / mask IoU | dec fp16 HTP "
          "| dec CPU x4 | first mask |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n| variant | encoder | HTP | emb cos min | mask IoU mean / min |")
    print("|---|---|---|---|---|")
    print("\n".join(qrows))


def _ms(e):
    return f"{e['median_ms']:.1f} ms" if e.get("median_ms") else ("fails" if e else "-")


def _f(x, n):
    return "-" if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "ref", "quantize", "host", "phone", "report"])
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--policy", default="int8", choices=["int8", "mix", "a16", "stem8", "dw8", "dw16"])
    ap.add_argument("--method", default="minmax", choices=["minmax", "mse", "percentile",
                                                            "entropy"])
    ap.add_argument("--pieces", default="all", choices=["all", "enc", "dec"])
    ap.add_argument("--gelu", default="exact", choices=["exact", "tanh", "tanh_ops", "sigmoid"],
                    help="export: also write an approximate-GELU encoder (enc.fp16_<gelu>.onnx)")
    ap.add_argument("--upsample", default="", choices=["", "bilinear", "polyphase"],
                    help="export: also write an encoder with bilinear neck upsampling")
    ap.add_argument("--tags", default="", help="phone: only these quantized tags (comma list)")
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--images", type=int, default=len(EVAL_IDS))
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    {"export": cmd_export, "ref": cmd_ref, "quantize": cmd_quantize, "host": cmd_host,
     "phone": cmd_phone, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
