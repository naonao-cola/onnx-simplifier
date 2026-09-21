"""Time the ``modelopt_pipeline.py`` variants with ``trtexec`` and print a table.

Stdlib only. Each ``*.onnx`` in DIR is built and run with the precision flags implied
by its name (``*.int8*.onnx`` -> ``--int8 --fp16``, i.e. ModelOpt's mixed INT8/FP16
Q/DQ output; everything else runs at both fp32 and ``--fp16``). Reports mean/median
GPU compute time (trtexec's own CUDA-event timing, transfers excluded) and engine
build time.

    python bench_trtexec.py DIR [--duration 5] [--glob '*.sim*.onnx'] [--json out.json]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def run(onnx, flags, duration):
    cmd = [TRTEXEC, f"--onnx={onnx}", "--noDataTransfers", f"--duration={duration}",
           "--warmUp=500", "--avgRuns=20", *flags]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
                  r"median = ([\d.]+) ms", out)
    b = re.search(r"Engine built in ([\d.]+) sec", out)
    if not m or "&&&& PASSED" not in out:
        err = next((l for l in out.splitlines() if "[E]" in l), "no result")
        return {"error": err.strip()[:160]}
    return {"mean_ms": float(m.group(3)), "median_ms": float(m.group(4)),
            "min_ms": float(m.group(1)), "build_s": float(b.group(1)) if b else None}


def configs(path):
    if ".int8" in path.name:
        return [("int8+fp16", ["--int8", "--fp16"])]
    return [("fp32", []), ("fp16", ["--fp16"])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--glob", default="*.onnx", help="e.g. '*.sim*.onnx' to skip the un-simplified controls")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rows = []
    for path in sorted(Path(args.dir).glob(args.glob)):
        for prec, flags in configs(path):
            r = run(path, flags, args.duration)
            variant = path.name[: -len(".onnx")]
            rows.append({"model": variant, "precision": prec, **r})
            print(rows[-1], flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    print(f"\n{'model':<22}{'precision':<11}{'mean ms':>9}{'median ms':>11}{'build s':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<22}{r['precision']:<11}  ERROR {r['error']}")
        else:
            print(f"{r['model']:<22}{r['precision']:<11}{r['mean_ms']:>9.3f}"
                  f"{r['median_ms']:>11.3f}{r['build_s'] or 0:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
