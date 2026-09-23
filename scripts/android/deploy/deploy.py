#!/usr/bin/env python3
"""Deploy a vision model to the phone's Hexagon from a spec file, in cached stages.

    deploy.py models/<name>.yaml [--device 239dbd8f] [--stages a,b,...] [--force a,b,...]

Stages, each cached under work/<name>/<stage>/ and re-run only when its inputs change (or with
--force). Any stage can be run on its own with --stages; earlier stages must already be cached.

  fetch      download the model (pinned URL + sha256) and the real images the spec names
  simplify   onnxsim with the spec's fixed input shapes (QNN needs static shapes)
  quantize   static int8 QDQ PTQ on real calibration images (per-channel int8 weights, uint8
             activations -- both measured free/right on the HTP, see ../htp_exploration)
  rewrite    graph rewrites from passes/ (e.g. uint8 NHWC input, raw uint8 outputs)
  post       build the CPU post-processing graph the spec asks for (e.g. YOLO decode + NMS)
  pipe       write the pipeline file (pipe.txt) that runtime/pipe_run reads, + the host inputs
  partition  run the HTP model alone through pipe_run with CPU fallback allowed and strict, and
             report every op QNN refused (with tensor rank; see ../vision_models_plan.md)
  push       build runtime/pipe_run and push it, the libs, models, pipe.txt and inputs
  bench      run pipe.txt on the phone over the eval images: per-step and total latency, FPS
  accuracy   compare the phone's outputs with fp32 ORT on the host (metric set by the spec)

Heavy host steps (onnxsim, quantization, calibration) run in a child process under
`systemd-run --user -p MemoryMax=<--mem, default 16G>` when systemd-run is available, one at a
time. Nothing runs in the background.

A spec with `prebuilt:` (see models/maskrcnn.yaml) points at a pipeline another script built;
simplify..post are skipped for it and pipe..accuracy run as usual. See README.md.
"""
from __future__ import annotations

import argparse
import collections.abc
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from passes import REWRITES  # noqa: E402
from stages import images as imglib  # noqa: E402

STAGES = ["fetch", "simplify", "quantize", "rewrite", "post", "pipe", "partition", "push", "bench", "accuracy"]
HEAVY = {"simplify", "quantize", "rewrite", "post", "pipe"}  # run in a memory-capped child


# ---------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, spec_path: Path, device: str, work: Path, mem: str):
        self.spec_path = spec_path.resolve()
        self.spec = yaml.safe_load(spec_path.read_text())
        self.name = self.spec["name"]
        self.device = device
        self.work = (work / self.name).resolve()
        self.images = (work / "_images").resolve()
        self.mem = mem
        # `prebuilt:` = a pipeline some other script already built (models + pipe file), e.g.
        # Mask R-CNN from ../e2e_pipeline/build_models.py; only pipe..accuracy apply to it
        pb = self.spec.get("prebuilt")
        self.prebuilt = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in pb.items()} if pb else None
        if self.prebuilt:
            miss = [k for k, v in self.prebuilt.items() if isinstance(v, str) and "$" in v]
            if miss:
                raise SystemExit(f"prebuilt: set the environment variables in {miss}")

    def d(self, stage: str) -> Path:
        p = self.work / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def section(self, stage: str):
        return self.spec.get(stage, {})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stamp_of(ctx: Ctx, stage: str) -> str:
    """A stage's cache key: the spec keys it reads, its code (the st_* function + the modules it
    uses), and the previous stage's stamp -- so editing, say, the bench code re-runs only bench."""
    i = STAGES.index(stage)
    prev = (ctx.work / STAGES[i - 1] / "stamp.json") if i else None
    prev_s = json.loads(prev.read_text())["key"] if prev and prev.exists() else ""
    code = hashlib.sha256(inspect.getsource(FUNCS[stage]).encode())
    depends, code_files = list(DEPENDS[stage]), list(CODE[stage])
    if stage == "quantize" and _auto_calibration(ctx):  # the picker scores with post + accuracy
        depends += ["accuracy", "postprocess"]
        code_files += ["stages/accuracy.py", "stages/post.py"]
    for f in code_files:
        for g in sorted(HERE.glob(f)):
            code.update(g.read_bytes())
    if ONNXSIM_CODE.get(stage):  # the onnxsim the child will import (PYTHONPATH included)
        spec = importlib.util.find_spec("onnxsim")
        for f in ONNXSIM_CODE[stage]:
            code.update((Path(spec.origin).parent / f).read_bytes())
    keys = {"stage": stage, "spec": {k: ctx.spec.get(k) for k in depends}, "device": ctx.device
            if stage in ("partition", "push", "bench", "accuracy") else "", "prev": prev_s,
            "code": code.hexdigest()}
    return hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()


DEPENDS = {
    "fetch": ["fetch", "calibration", "eval", "prebuilt"],
    "simplify": ["inputs"],
    "quantize": ["quantize", "preprocess", "calibration", "inputs"],
    "rewrite": ["rewrites"],
    "post": ["postprocess"],
    "pipe": ["pipeline", "preprocess", "eval", "inputs", "prebuilt"],
    "partition": ["pipeline"],
    "push": ["pipeline"],
    "bench": ["bench"],
    "accuracy": ["accuracy", "postprocess"],
}


RUNTIME = ["runtime/*.cpp", "runtime/*.sh"]
CODE = {
    "fetch": ["stages/images.py"],
    "simplify": [],
    "quantize": ["stages/images.py"],
    "rewrite": ["passes/*.py"],
    "post": ["stages/post.py"],
    "pipe": ["stages/pipe.py", "stages/images.py"],
    "partition": ["stages/device.py", "stages/partition.py", *RUNTIME],
    "push": ["stages/device.py", *RUNTIME],
    "bench": ["stages/device.py", *RUNTIME],
    "accuracy": ["stages/accuracy.py", "stages/post.py", "stages/images.py"],
}
# onnxsim modules a stage runs: a quantizer change must invalidate quantize and what follows
ONNXSIM_CODE = {"quantize": ["calibration.py", "calibration_pick.py", "qdq_full_graph.py"]}


def _auto_calibration(ctx: Ctx) -> bool:
    q = ctx.spec.get("quantize") or {}
    return q.get("enabled", True) and str(q.get("calibration_method", "")).lower() == "auto"


# what `calibration_method: auto` compares (onnxsim.pick_calibration candidates); `auto` itself is
# onnxsim's per-tensor choice
AUTO_CANDIDATES = ["minmax", "mse", "percentile:99.999", "percentile:99.99", "entropy", "auto"]


# ---------------------------------------------------------------------------------------------
# stages (each takes ctx and writes into ctx.d(stage))

PREBUILT_SKIP = {"simplify", "quantize", "rewrite", "post"}


def st_fetch(ctx: Ctx) -> None:
    if ctx.prebuilt:
        from stages import pipe

        files = pipe.prebuilt_files(ctx)
        man = {f.name: f.stat().st_size for f in files}
        (ctx.d("fetch") / "prebuilt.json").write_text(json.dumps(man, indent=1))
        print(f"  prebuilt pipeline {ctx.prebuilt['pipe']}: {len(files)} files, "
              f"{sum(man.values()) / 1e6:.0f} MB")
        return
    f = ctx.section("fetch")
    out = ctx.d("fetch") / "model.onnx"
    if "url" in f:
        if not out.exists() or (f.get("sha256") and sha256(out) != f["sha256"]):
            print(f"  downloading {f['url']}")
            urllib.request.urlretrieve(f["url"], out)
    elif "script" in f:  # a pinned export recipe: script writes model.onnx into the stage dir
        subprocess.run([sys.executable, str((ctx.spec_path.parent / f["script"]).resolve()), str(out),
                        *map(str, f.get("args", []))], check=True)
    elif "path" in f:  # an already-built local model (e.g. produced by another repo script)
        shutil.copyfile(os.path.expandvars(f["path"]), out)
    else:
        raise SystemExit("fetch: need url, script or path")
    got = sha256(out)
    if f.get("sha256") and got != f["sha256"]:
        raise SystemExit(f"fetch: sha256 mismatch {got} != {f['sha256']}")
    print(f"  model.onnx sha256 {got}")
    for key in ("calibration", "eval"):
        imglib.fetch_images(ctx.spec.get(key, {}), ctx.images)


def st_simplify(ctx: Ctx) -> None:
    import onnx
    import onnxsim

    src = ctx.work / "fetch" / "model.onnx"
    shapes = {k: v["shape"] for k, v in ctx.spec["inputs"].items()}
    m, ok = onnxsim.simplify(onnx.load(str(src)), overwrite_input_shapes=shapes)
    if not ok:
        raise SystemExit("simplify: onnxsim could not validate the simplified model")
    onnx.save(m, str(ctx.d("simplify") / "model.onnx"))
    print(f"  {len(m.graph.node)} nodes after onnxsim, inputs fixed to {shapes}")


def st_quantize(ctx: Ctx) -> None:
    q = ctx.section("quantize")
    src = ctx.work / "simplify" / "model.onnx"
    dst = ctx.d("quantize") / "model.onnx"
    if not q.get("enabled", True):
        shutil.copyfile(src, dst)
        print("  quantization disabled: fp32 graph (the HTP runs it as fp16)")
        return
    import fnmatch
    import resource

    import onnx
    import onnxsim

    for k in ("op_types", "extra_options"):  # ORT-quantizer keys this stage no longer reads
        if k in q:
            raise SystemExit(f"quantize.{k} is not supported (onnxsim.quantize_static); "
                             "use exclude_nodes / exclude_op_types")
    files = imglib.list_images(ctx.spec.get("calibration", {}), ctx.images)
    method = q.get("calibration_method", "minmax").lower()
    model = onnx.load(str(src))
    tensors = {o for n in model.graph.node for o in n.output}
    keep = sorted(t for t in tensors if any(fnmatch.fnmatchcase(t, p) for p in q.get("minmax_tensors", [])))
    opts = dict(minmax_tensor_names=keep, full_graph=True, per_channel=q.get("per_channel", True),
                activation_type=q.get("activation", "uint8"), nodes_to_exclude=q.get("exclude_nodes", []),
                op_types_to_exclude=q.get("exclude_op_types", []))
    t = time.time()
    extra = {}
    if method == "auto":
        m, extra = _pick_calibration(ctx, q, model, files, opts)
    else:
        m = onnxsim.quantize_static(model, _ImageBatches(ctx, files), method=method,
                                    percentile=float(q.get("percentile", 99.999)), **opts)
    onnx.save(m, str(dst))
    meta = {"method": method, "percentile": q.get("percentile", 99.999) if method == "percentile" else None,
            **extra, "minmax_tensors": len(keep), "images": len(files), "seconds": round(time.time() - t, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
    (ctx.d("quantize") / "quantize_meta.json").write_text(json.dumps(meta, indent=1))
    shown = f"auto -> {extra['picked']}" if extra else method
    print(f"  QDQ int8 ({shown}) on {len(files)} calibration images in {meta['seconds']} s, "
          f"peak RSS {meta['peak_rss_mb']} MB")


class _ImageBatches:
    """Calibration batches: re-iterable (calibration runs twice), preprocessed on demand."""

    def __init__(self, ctx: Ctx, files: list):
        (self.name,) = list(ctx.spec["inputs"])
        self.files, self.pre = files, ctx.spec["preprocess"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if i >= len(self.files):
            raise IndexError(i)
        return {self.name: imglib.preprocess(self.files[i], self.pre)[0][None]}


collections.abc.Sequence.register(_ImageBatches)


def _pick_calibration(ctx: Ctx, q: dict, model, files: list, opts: dict):
    """`calibration_method: auto`: onnxsim.pick_calibration over AUTO_CANDIDATES (or
    `auto_candidates`), cross-fitted over the calibration images (`auto_folds`, default 4): each
    fold is scored by candidates calibrated on the other folds, on host ORT against the fp32
    model, with the spec's own accuracy kind -- detection_match runs the spec's postprocess on both
    sides and scores matched/ref detections (worst-output SQNR only breaks ties); any other kind
    scores by worst-output SQNR. The winner is then calibrated on all the images. The eval ids are
    never used. (A single held-out quarter -- 16 of YOLO11n's 64 -- ranked percentile 99.99 first;
    4 folds rank mse first, like 128 separate images do.)"""
    import onnx
    import onnxsim
    from onnxsim.calibration_pick import run_outputs, worst_output_sqnr

    from stages import accuracy, post

    folds = int(q.get("auto_folds", 4))
    acc = ctx.spec.get("accuracy", {}) or {}
    metric, kind = worst_output_sqnr, "worst_output_sqnr"
    if acc.get("kind") == "detection_match" and ctx.spec.get("postprocess"):
        pdir = ctx.d("quantize") / "auto_post"
        pdir.mkdir(exist_ok=True)
        info = post.build(ctx.spec["postprocess"], ctx.work / "simplify" / "model.onnx", pdir)
        pmodel = onnx.load(str(pdir / "post.onnx"))
        head, outs = info["input"], info["outputs"][:3]  # boxes, scores, classes

        def dets(net_outputs):
            return [[o[k] for k in outs] for o in run_outputs(pmodel, [{head: n[head]} for n in net_outputs])]

        def det_metric(f_out, q_out):
            matched = ref = 0
            for a, b in zip(dets(f_out), dets(q_out)):
                r = accuracy.det_match(a, b, acc.get("iou", 0.5), acc.get("score", 0.25))
                matched, ref = matched + r["matched"], ref + r["ref"]
            return matched / max(ref, 1) + 1e-6 * worst_output_sqnr(f_out, q_out)

        metric, kind = det_metric, "detection_match"
    cands = [str(c) for c in q.get("auto_candidates", AUTO_CANDIDATES)]
    pick = onnxsim.pick_calibration(model, _ImageBatches(ctx, files), metric=metric, candidates=cands,
                                    folds=folds, verbose=True, **opts)
    per_tensor: dict = {}
    for c in pick.auto_choices.values():
        per_tensor[c] = per_tensor.get(c, 0) + 1
    return pick.model, {"picked": pick.method, "score_kind": kind,
                        "scores": {k: round(v, 6) for k, v in pick.scores.items()},
                        "auto_per_tensor": dict(sorted(per_tensor.items())), "folds": folds}


def st_rewrite(ctx: Ctx) -> None:
    import onnx

    cur = ctx.work / "quantize" / "model.onnx"
    d = ctx.d("rewrite")
    meta = {}
    m = onnx.load(str(cur))
    for r in ctx.spec.get("rewrites", []) or []:
        name, args = (r, {}) if isinstance(r, str) else (r["name"], {k: v for k, v in r.items() if k != "name"})
        m, info = REWRITES[name](m, **args)
        meta[name] = info
        print(f"  rewrite {name}: {info}")
    onnx.checker.check_model(m)
    onnx.save(m, str(d / "model.onnx"))
    (d / "rewrite_meta.json").write_text(json.dumps(meta, indent=1))


def st_post(ctx: Ctx) -> None:
    from stages import post

    d = ctx.d("post")
    p = ctx.spec.get("postprocess")
    info = post.build(p, ctx.work / "rewrite" / "model.onnx", d) if p else {}
    (d / "post_meta.json").write_text(json.dumps(info, indent=1))


def st_pipe(ctx: Ctx) -> None:
    from stages import pipe

    pipe.write(ctx, ctx.d("pipe"))


def st_partition(ctx: Ctx) -> None:
    from stages import device

    device.partition(ctx, ctx.d("partition"))


def st_push(ctx: Ctx) -> None:
    from stages import device

    device.push(ctx, ctx.d("push"))


def st_bench(ctx: Ctx) -> None:
    from stages import device

    device.bench(ctx, ctx.d("bench"))


def st_accuracy(ctx: Ctx) -> None:
    from stages import accuracy

    accuracy.run(ctx, ctx.d("accuracy"))


FUNCS = {s: globals()[f"st_{s}"] for s in STAGES}


# ---------------------------------------------------------------------------------------------
def run_stage(ctx: Ctx, stage: str, force: bool, capped: bool) -> None:
    sd = ctx.work / stage
    key = stamp_of(ctx, stage)
    stamp = sd / "stamp.json"
    if not force and stamp.exists() and json.loads(stamp.read_text())["key"] == key:
        print(f"[{stage}] cached")
        return
    t = time.time()
    if ctx.prebuilt and stage in PREBUILT_SKIP:
        print(f"[{stage}] prebuilt pipeline: skipped")
        sd.mkdir(parents=True, exist_ok=True)
    elif (stage in HEAVY and not ctx.prebuilt) and capped and shutil.which("systemd-run"):
        print(f"[{stage}]", flush=True)
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "-q", "-p", f"MemoryMax={ctx.mem}",
               "-p", "MemorySwapMax=0", "-d", "-E", f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
               sys.executable, __file__, str(ctx.spec_path), "--device", ctx.device, "--work",
               str(ctx.work.parent), "--mem", ctx.mem, "--in-child", stage]
        r = subprocess.run(cmd)
        if r.returncode:
            raise SystemExit(f"[{stage}] failed (exit {r.returncode}; OOM under MemoryMax={ctx.mem}?)")
    else:
        print(f"[{stage}]", flush=True)
        FUNCS[stage](ctx)
    stamp.write_text(json.dumps({"key": key, "seconds": round(time.time() - t, 1)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--device", default=os.environ.get("DEVICE_SERIAL", "239dbd8f"))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", default="", help="comma-separated stages to re-run even if cached")
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--mem", default="16G", help="MemoryMax for heavy host stages")
    ap.add_argument("--no-cap", action="store_true", help="run heavy stages in-process")
    ap.add_argument("--in-child", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    ctx = Ctx(a.spec, a.device, a.work, a.mem)
    if a.in_child:
        FUNCS[a.in_child](ctx)
        return
    force = set(filter(None, a.force.split(",")))
    for s in a.stages.split(","):
        if s not in FUNCS:
            raise SystemExit(f"unknown stage {s}; stages: {','.join(STAGES)}")
        run_stage(ctx, s, s in force, not a.no_cap)


if __name__ == "__main__":
    main()
