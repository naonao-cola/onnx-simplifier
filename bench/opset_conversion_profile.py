#!/usr/bin/env python3
"""Profile onnxsim's opset version converter path (``target_opset_version``).

Follow-up to ``RESULTS_profiling_survey.md``, which found the Python<->C++
marshalling boundary dominating simplify()'s *unprofiled* time. That report's
fix (passing paths through, avoiding a double SerializeToString) closed one
"invisible to ONNXSIM_PROFILE" gap. This script targets a second one:
``ConvertOpsetVersion`` (the C++ wrapper around onnx's version converter,
``onnx::version_conversion::ConvertVersion``) used to run *before* the
profiled "Simplify" root span in onnxsim.cpp, so its cost never showed up in
``ONNXSIM_PROFILE`` output at all -- see onnxsim.cpp's ConvertOpsetVersion
call site. It is now wrapped in its own "ConvertOpsetVersion" span, sibling to
"Simplify", so this script can measure it directly.

Synthetic models (not a downloaded regression model) so this is
self-contained and reproducible: a low-opset (9) CNN-shaped graph, built at a
few sizes, with real numpy-backed initializer weights -- large enough for the
Import/Export "back to protobuf" (ModelProto <-> onnx IR Graph) tensor-copy
cost highlighted in issue discussion to be visible, small enough to run
quickly and stay within sandbox disk/memory limits.

Usage:
    python bench/opset_conversion_profile.py [--trials N] [--target latest]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
from onnx import numpy_helper, parser

WORK_DIR = os.environ.get("OPSET_PROFILE_DIR", tempfile.mkdtemp(prefix="onnxsim_opset_profile_"))


def _conv_bn_relu_block(idx: int, channels: int) -> tuple[str, list[onnx.TensorProto]]:
    """One Conv+BatchNormalization+Relu block as an onnx-text fragment, plus its
    initializers (weights sized to give the model real tensor bytes to copy)."""
    w_name, b_name = f"w{idx}", f"b{idx}"
    scale, bias, mean, var = f"s{idx}", f"bs{idx}", f"m{idx}", f"v{idx}"
    w = numpy_helper.from_array(
        np.random.randn(channels, channels, 3, 3).astype(np.float32) * 0.01, name=w_name
    )
    b = numpy_helper.from_array(np.zeros(channels, dtype=np.float32), name=b_name)
    s = numpy_helper.from_array(np.ones(channels, dtype=np.float32), name=scale)
    bs = numpy_helper.from_array(np.zeros(channels, dtype=np.float32), name=bias)
    m = numpy_helper.from_array(np.zeros(channels, dtype=np.float32), name=mean)
    v = numpy_helper.from_array(np.ones(channels, dtype=np.float32), name=var)
    prev = "x" if idx == 0 else f"y{idx - 1}"
    body = f"""
        c{idx} = Conv<pads=[1,1,1,1]>({prev}, {w_name}, {b_name})
        n{idx} = BatchNormalization(c{idx}, {scale}, {bias}, {mean}, {var})
        y{idx} = Relu(n{idx})
    """
    return body, [w, b, s, bs, m, v]


def build_model(num_blocks: int, channels: int = 64, opset: int = 9) -> onnx.ModelProto:
    """A num_blocks-deep Conv/BN/Relu stack at a low opset, so
    target_opset_version="latest" has real adapter work to do (BatchNormalization
    alone changed shape at opsets 9, 14; Conv/Relu are stable but still walked)."""
    bodies, inits = [], []
    for i in range(num_blocks):
        body, block_inits = _conv_bn_relu_block(i, channels)
        bodies.append(body)
        inits.extend(block_inits)

    last = f"y{num_blocks - 1}"
    text = f"""
        <ir_version: 8, opset_import: ["": {opset}]>
        graph (float[1,{channels},32,32] x) => (float[1,{channels},32,32] {last})
        {{
            {"".join(bodies)}
        }}
    """
    model = parser.parse_model(text)
    model.graph.initializer.extend(inits)
    return model


CHILD_SRC = r'''
import json, resource, sys, time
import onnx
from onnxsim import simplify

model_path = sys.argv[1]
profile_path = sys.argv[2]
target = sys.argv[3]
target = "latest" if target == "latest" else int(target)

model = onnx.load(model_path)
t0 = time.time()
sim, ok = simplify(model, target_opset_version=target, profile=profile_path)
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
opset = next(o.version for o in sim.opset_import if o.domain in ("", "ai.onnx"))
print("__CHILDRESULT__" + json.dumps({
    "seconds": round(dt, 4),
    "peak_rss_mb": round(peak, 1),
    "valid": bool(ok),
    "result_opset": opset,
}))
'''


def run_child(model_path: str, profile_path: str, target: str):
    args = [sys.executable, "-c", CHILD_SRC, model_path, profile_path, target]
    proc = subprocess.run(args, capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__CHILDRESULT__"):
            result = json.loads(line[len("__CHILDRESULT__"):])
    return result, proc.stdout, proc.stderr


def parse_summary_table(stdout: str) -> dict:
    lines = stdout.splitlines()
    rows = {}
    in_table = False
    for line in lines:
        if "onnxsim profiling summary" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("function") or line.startswith("---"):
            continue
        if not line.strip():
            break
        m = re.match(r"^(\s*)(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if not m:
            continue
        indent, name, calls, wall, cpu, max_wall, peak = m.groups()
        rows[name.strip()] = {
            "calls": int(calls),
            "wall_ms": float(wall),
            "peak_mib": float(peak),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--target", default="latest")
    ap.add_argument(
        "--configs",
        default="tiny:1:16,large:40:96",
        help="comma-separated name:num_blocks:channels configs",
    )
    args = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"work dir: {WORK_DIR}\n")

    for cfg in args.configs.split(","):
        name, num_blocks, channels = cfg.split(":")
        num_blocks, channels = int(num_blocks), int(channels)
        model = build_model(num_blocks, channels)
        model_path = os.path.join(WORK_DIR, f"{name}.onnx")
        onnx.save(model, model_path)
        total_init_bytes = sum(len(t.raw_data) for t in model.graph.initializer)
        print(f"=== {name}: {num_blocks} blocks, {channels} ch, "
              f"{len(model.graph.node)} nodes, "
              f"{total_init_bytes / 1e6:.1f} MB initializers, opset 9 -> {args.target} ===")

        trial_results = []
        for t in range(args.trials):
            profile_path = os.path.join(WORK_DIR, f"{name}_trial{t}.json")
            result, out, err = run_child(model_path, profile_path, args.target)
            if result is None:
                print(f"  trial {t}: CRASHED\n--- stdout ---\n{out[-2000:]}\n--- stderr ---\n{err[-2000:]}")
                continue
            table = parse_summary_table(out)
            convert = table.get("ConvertOpsetVersion")
            simplify_span = table.get("Simplify")
            trial_results.append({
                "seconds": result["seconds"],
                "peak_rss_mb": result["peak_rss_mb"],
                "convert_ms": convert["wall_ms"] if convert else None,
                "simplify_ms": simplify_span["wall_ms"] if simplify_span else None,
            })
            conv_str = f"{convert['wall_ms']:.2f}ms" if convert else "n/a (not profiled?)"
            simp_str = f"{simplify_span['wall_ms']:.2f}ms" if simplify_span else "n/a"
            print(f"  trial {t}: total={result['seconds']:.3f}s peak={result['peak_rss_mb']:.1f}MiB "
                  f"ConvertOpsetVersion={conv_str} Simplify={simp_str} "
                  f"result_opset={result['result_opset']} valid={result['valid']}")

        if trial_results:
            conv_vals = [r["convert_ms"] for r in trial_results if r["convert_ms"] is not None]
            if conv_vals:
                print(f"  -> ConvertOpsetVersion median: {sorted(conv_vals)[len(conv_vals)//2]:.2f}ms "
                      f"(min={min(conv_vals):.2f}ms, max={max(conv_vals):.2f}ms)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
