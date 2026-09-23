"""Phone-side stages: partition report, push, bench.

Device layout: /data/local/tmp/deploy/ holds the shared runtime (pipe_run, qnn_run_multi, ORT +
QNN libraries, pushed only when their size changes); /data/local/tmp/deploy/<name>/ holds one
model's pipe.txt, models and inputs. Libraries come from
../../htp_exploration/qnn_shell/fetch_libs.sh (Maven Central, not committed).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import partition as part

HERE = Path(__file__).resolve().parents[1]
QS = HERE.parent / "htp_exploration" / "qnn_shell"
ROOT = "/data/local/tmp/deploy"
ADSP = f"{ROOT};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"


def adb(ctx, *args, check=True, capture=True, timeout=600):
    r = subprocess.run(["adb", "-s", ctx.device, *args], capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode:
        raise SystemExit(f"adb {' '.join(args[:3])}: {r.stderr or r.stdout}")
    return r.stdout if capture else ""


def push_if_changed(ctx, src: Path, dst: str) -> None:
    size = str(src.stat().st_size)
    got = adb(ctx, "shell", f"stat -c %s {dst} 2>/dev/null", check=False).strip()
    if got != size:
        adb(ctx, "push", str(src), dst)


def runtime(ctx) -> Path:
    """Build runtime/pipe_run and qnn_run_multi (once; rebuilt when sources change)."""
    subprocess.run([str(HERE / "runtime" / "build.sh")], check=True)
    return HERE / "runtime" / "build"


def push_runtime(ctx) -> None:
    b = runtime(ctx)
    adb(ctx, "shell", f"mkdir -p {ROOT}")
    for f in [b / "pipe_run", b / "qnn_run_multi", *sorted((QS / "libs").glob("*.so"))]:
        push_if_changed(ctx, f, f"{ROOT}/{f.name}")
    adb(ctx, "shell", f"chmod 755 {ROOT}/pipe_run {ROOT}/qnn_run_multi")


def dev_dir(ctx) -> str:
    return f"{ROOT}/{ctx.name}"


def push_model(ctx) -> None:
    pd = ctx.work / "pipe"
    dd = dev_dir(ctx)
    adb(ctx, "shell", f"mkdir -p {dd}/inputs")
    for f in [*pd.glob("*.onnx"), pd / "pipe.txt"]:
        push_if_changed(ctx, f, f"{dd}/{f.name}")
    for f in sorted((pd / "inputs").glob("*.bin")):
        push_if_changed(ctx, f, f"{dd}/inputs/{f.name}")


def partition(ctx, d: Path) -> None:
    """Run the HTP model alone, fallback allowed (verbose log -> logcat) then strict."""
    push_runtime(ctx)
    push_model(ctx)
    meta = json.loads((ctx.work / "pipe" / "pipe_meta.json").read_text())
    import onnx

    m = onnx.load(str(ctx.work / "pipe" / meta["model"]), load_external_data=False)
    gi = m.graph.input[0]
    ty = {1: "f32", 2: "u8"}[gi.type.tensor_type.elem_type]
    dims = ",".join(str(x.dim_value) for x in gi.type.tensor_type.shape.dim)
    first = meta["inputs"][0]
    # the network input is either the raw input file (f32 CHW = NCHW bytes, or host-quantized u8)
    # or, when the phone quantizes, a uint8 NHWC tensor made here once from the same image
    src = f"inputs/{first}.bin"
    if meta["uint8_input"] and not meta["host_quantize"]:
        import numpy as np

        u = meta["uint8_input"]
        x = np.fromfile(ctx.work / "pipe" / "inputs" / f"{first}.bin", np.float32).reshape(u["shape"][3], *u["shape"][1:3])
        q = np.clip(np.rint(x / u["scale"]) + u["zero_point"], 0, 255).astype(np.uint8).transpose(1, 2, 0)
        tmp = d / "net_input.bin"
        q.tofile(tmp)
        push_if_changed(ctx, tmp, f"{dev_dir(ctx)}/net_input.bin")
        src = "net_input.bin"
    man = d / "manifest.txt"
    man.write_text(f"{gi.name} {ty} {src} {dims}\n")
    push_if_changed(ctx, man, f"{dev_dir(ctx)}/manifest.txt")
    env = f"cd {dev_dir(ctx)} && LD_LIBRARY_PATH={ROOT} ADSP_LIBRARY_PATH='{ADSP}' QNN_PERF=burst"
    adb(ctx, "logcat", "-c")
    fb = adb(ctx, "shell", f"{env} ORT_LOG=0 {ROOT}/qnn_run_multi {meta['model']} manifest.txt htp-fallback 3 pr 2>&1",
             check=False)
    (d / "htp-fallback.out").write_text(fb)
    (d / "logcat.txt").write_text(adb(ctx, "logcat", "-d", check=False))
    st = adb(ctx, "shell", f"{env} {ROOT}/qnn_run_multi {meta['model']} manifest.txt htp 10 pr 2>&1", check=False)
    (d / "htp.out").write_text(st)
    rep = part.summarize(ctx.work / "pipe" / meta["model"], d / "logcat.txt", fb, st)
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    print(part.render(rep))


def push(ctx, d: Path) -> None:
    push_runtime(ctx)
    push_model(ctx)
    (d / "pushed.txt").write_text(dev_dir(ctx) + "\n")


def parse_bench(log: str) -> dict:
    res = {"inputs": {}}
    cur = None
    for line in log.splitlines():
        if m := re.match(r"(\S+) total_ms median ([0-9.]+) min ([0-9.]+) \(n=(\d+)\) fps ([0-9.]+)", line):
            cur = m.group(1)
            res["inputs"][cur] = {"median_ms": float(m.group(2)), "min_ms": float(m.group(3)),
                                  "fps": float(m.group(5)), "steps": {}}
        elif (m := re.match(r"\s+step (\S+)\s+([0-9.]+) ms", line)) and cur:
            res["inputs"][cur]["steps"][m.group(1)] = float(m.group(2))
        elif m := re.match(r"overall total_ms median ([0-9.]+) fps ([0-9.]+)", line):
            res["median_ms"], res["fps"] = float(m.group(1)), float(m.group(2))
        elif m := re.match(r"(\S+) cold_ms ([0-9.]+)", line):
            res["cold_ms"] = float(m.group(2))
        elif m := re.match(r"session (\S+) create_ms ([0-9.]+)", line):
            res.setdefault("session_create_ms", {})[m.group(1)] = float(m.group(2))
    res["pass"] = "PASS" in log
    return res


def bench(ctx, d: Path) -> None:
    b = ctx.spec.get("bench", {}) or {}
    push_runtime(ctx)
    push_model(ctx)
    dd = dev_dir(ctx)
    meta = json.loads((ctx.work / "pipe" / "pipe_meta.json").read_text())
    ins = " ".join(f"inputs/{s}.bin" for s in meta["inputs"])
    adb(ctx, "shell", f"rm -rf {dd}/res && mkdir -p {dd}/res")
    log = adb(ctx, "shell", f"cd {dd} && ORT_THREADS={b.get('cpu_threads', 4)} LD_LIBRARY_PATH={ROOT} "
                            f"ADSP_LIBRARY_PATH='{ADSP}' {ROOT}/pipe_run pipe.txt {b.get('warmup', 2)} "
                            f"{b.get('reps', 10)} res {ins} 2>&1", check=False, timeout=3600)
    (d / "bench.log").write_text(log)
    res = parse_bench(log)
    if not res["pass"]:
        raise SystemExit(f"bench failed:\n{log[-2000:]}")
    out = d / "outputs"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    out.mkdir()
    adb(ctx, "pull", f"{dd}/res/.", str(out))
    (d / "bench.json").write_text(json.dumps(res, indent=1))
    print(f"  median {res['median_ms']:.2f} ms = {res['fps']:.1f} FPS over {len(res['inputs'])} inputs"
          f" (cold first run {res.get('cold_ms', 0):.1f} ms)")
    for k, v in next(iter(res["inputs"].values()))["steps"].items():
        print(f"    step {k:14s} {v:8.2f} ms")
