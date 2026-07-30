#!/usr/bin/env python3
"""Export the latest Ultralytics YOLO models to ONNX and simplify with onnxsim.

Ultralytics YOLO (https://github.com/ultralytics/ultralytics) is the most
widely deployed real-time object detector family. Ultralytics' ONNX export
path historically shelled out to ``onnxsim.simplify`` (the ``simplify=True``
export flag); onnxsim is therefore a long-standing part of the YOLO
deployment pipeline. This standalone harness reproduces that step against the
*current* onnxsim on the newest YOLO generations and every task head.

Each spec is built from its Ultralytics **YAML config** (random weights, no
checkpoint download -- the graph structure onnxsim rewrites is identical to
the pretrained ``.pt`` model), exported with ``YOLO.export(format="onnx")``,
and run through ``onnxsim.simplify(check_n=3)``. It reports node counts
before/after and whether onnxsim's numerical equivalence check passed.

Requirements (all from PyPI)::

    pip install ultralytics onnxruntime onnxsim

``onnxruntime`` is not optional: onnxsim uses it to numerically compare the
original and simplified graphs (``check_n``). Without it onnxsim falls back
to onnx's slower Python reference evaluator.

Usage::

    python simplify_yolo.py                    # a small default set
    python simplify_yolo.py yolo26n yolo12n-seg
    python simplify_yolo.py --all              # newest gens x every task head
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
import traceback

import onnx

import onnxsim

os.environ.setdefault("YOLO_OFFLINE", "True")

# Newest YOLO generations (latest first). YOLO26 is the current flagship;
# YOLO12 and YOLO11 are the two prior generations, all actively shipped by
# Ultralytics. Task suffix -> head: (none)=detect, -seg, -pose, -obb, -cls.
GENERATIONS = ["yolo26", "yolo12", "yolo11"]
TASK_SUFFIXES = ["", "-seg", "-pose", "-obb", "-cls"]

# Nano scale keeps the harness fast; the graph structure (what onnxsim
# simplifies) is identical across n/s/m/l/x scales of a given config.
DEFAULT_SPECS = ["yolo26n", "yolo26n-seg", "yolo26n-pose"]
ALL_SPECS = [f"{gen}n{suffix}" for gen in GENERATIONS for suffix in TASK_SUFFIXES]


# Classification heads take 224x224; everything else uses the canonical 640.
def _imgsz_for(spec: str) -> int:
    return 224 if spec.endswith("-cls") else 640


def run_spec(spec: str, output_dir: str, opset: int) -> dict:
    """Build one YOLO spec from its YAML config, export to ONNX, simplify."""
    from ultralytics import YOLO

    result: dict = {
        "spec": spec,
        "onnxsim_version": onnxsim.__version__,
        "onnx_version": onnx.__version__,
    }
    try:
        imgsz = _imgsz_for(spec)
        result["imgsz"] = imgsz
        model = YOLO(f"{spec}.yaml")
        result["task"] = model.task

        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()
        # Ultralytics prints a lot; keep the batch summary readable.
        with contextlib.redirect_stdout(io.StringIO()):
            path = str(
                model.export(
                    format="onnx",
                    opset=opset,
                    imgsz=imgsz,
                    simplify=False,
                    dynamic=False,
                    verbose=False,
                )
            )
        # Move the export next to the other artifacts.
        dst = os.path.join(output_dir, os.path.basename(path))
        if os.path.abspath(dst) != os.path.abspath(path):
            os.replace(path, dst)
            path = dst
        result["export_s"] = round(time.time() - t0, 1)
        result["onnx_path"] = path

        graph = onnx.load(path)
        result["nodes_before"] = len(graph.graph.node)
        inp = graph.graph.input[0]
        result["input_shape"] = [d.dim_value for d in inp.type.tensor_type.shape.dim]

        t0 = time.time()
        model_opt, check_ok = onnxsim.simplify(path, check_n=3)
        result["simplify_s"] = round(time.time() - t0, 1)
        result["check_ok"] = bool(check_ok)
        result["nodes_after"] = len(model_opt.graph.node)
        onnx.save(model_opt, path.replace(".onnx", ".sim.onnx"))
        result["status"] = "OK" if check_ok else "SIMPLIFY_CHECK_FAILED"
    except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="*", help="YOLO spec names, e.g. yolo26n-seg")
    parser.add_argument(
        "--all", action="store_true", help="newest gens x every task head"
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset for export")
    parser.add_argument(
        "--output-dir", default="yolo_onnx", help="ONNX output directory"
    )
    args = parser.parse_args()

    if args.all:
        specs = ALL_SPECS
    elif args.specs:
        specs = args.specs
    else:
        specs = DEFAULT_SPECS

    results = []
    for spec in specs:
        print(f"=== {spec} ===", flush=True)
        result = run_spec(spec, args.output_dir, args.opset)
        print(json.dumps(result), flush=True)
        results.append(result)

    print("\n===== SUMMARY =====")
    ok = 0
    for r in results:
        status = r.get("status")
        ok += status == "OK"
        b, a = r.get("nodes_before", "?"), r.get("nodes_after", "?")
        red = (
            f"{100 * (b - a) // b}%"
            if isinstance(b, int) and isinstance(a, int) and b
            else "?"
        )
        print(
            f"{r['spec']:16s} {str(r.get('task', '?')):8s} {status:22s} "
            f"nodes {b}->{a} ({red}) check_ok={r.get('check_ok', '?')} {r.get('error', '')}"
        )
    print(f"\n{ok}/{len(results)} specs simplified with onnxsim {onnxsim.__version__}")


if __name__ == "__main__":
    main()
