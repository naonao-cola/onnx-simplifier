#!/usr/bin/env python3
"""Regression: can onnxsim replace onnxslim in ultralytics' ONNX export?

ultralytics (``ultralytics/engine/exporter.py``) exports a YOLO model to ONNX
and then, when ``simplify=True``, slims the graph with::

    model_onnx = onnxslim.slim(model_onnx)

This script tests swapping that call for onnxsim::

    model_onnx, ok = onnxsim.simplify(model_onnx)

For each YOLO task it exports the *raw* ONNX graph once (``simplify=False``),
then runs BOTH simplifiers on that identical graph and compares:

* status  - did the simplifier run without raising?
* nodes   - node count before / after (optimization strength)
* valid   - does the simplified graph load + run in onnxruntime?
* parity  - max abs diff of every output vs the raw graph on a fixed random
            input (correctness of the simplification)

onnxslim is the incumbent baseline; onnxsim is the candidate. A task "passes"
for onnxsim when it produces a graph that is valid and numerically matches the
raw export within tolerance, i.e. onnxsim is at least as safe a drop-in as the
onnxslim the exporter ships today.
"""

from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import onnx

DEFAULT_MODELS = [
    "yolo11n.pt",       # detect
    "yolo11n-seg.pt",   # segment
    "yolo11n-cls.pt",   # classify
    "yolo11n-pose.pt",  # pose
    "yolo11n-obb.pt",   # oriented bounding boxes
]

# numeric parity tolerance for raw-vs-simplified outputs
RTOL = 1e-3
ATOL = 1e-4


def _node_count(model: onnx.ModelProto) -> int:
    return len(model.graph.node)


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]):
    """Run a model in onnxruntime and return the list of output arrays."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _feeds(model: onnx.ModelProto, spatial: int = 640) -> dict[str, np.ndarray]:
    """Build a deterministic random input matching the model's input shapes.

    Dynamic axes have no fixed size, so we pick concrete ones: axis 0 (batch)
    -> 1, and any other dynamic axis -> ``spatial`` (a stride-32-aligned size
    the detector's downsampling path is valid for; feeding 1 there produces a
    degenerate 1x1 map that breaks the network before any simplifier runs).
    """
    rng = np.random.default_rng(0)
    feeds = {}
    for inp in model.graph.input:
        shape = []
        for axis, d in enumerate(inp.type.tensor_type.shape.dim):
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(1 if axis == 0 else spatial)
        feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def _parity(ref_outs, test_outs) -> float:
    """Max abs diff across all outputs; inf if shapes/counts mismatch."""
    if len(ref_outs) != len(test_outs):
        return float("inf")
    worst = 0.0
    for a, b in zip(ref_outs, test_outs):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            return float("inf")
        worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
    return worst


def _slim_with_onnxsim(model: onnx.ModelProto):
    import onnxsim

    sim, ok = onnxsim.simplify(model)
    return sim, ok


def _slim_with_onnxslim(model: onnx.ModelProto):
    import onnxslim

    return onnxslim.slim(model), True


def export_raw(weight: str, imgsz: int = 640, dynamic: bool = False):
    """Export a YOLO model to a raw (un-simplified) ONNX ModelProto."""
    from ultralytics import YOLO

    model = YOLO(weight)
    path = model.export(
        format="onnx", simplify=False, imgsz=imgsz, dynamic=dynamic, verbose=False
    )
    return onnx.load(str(path))


def evaluate(weight: str, dynamic: bool = False) -> dict:
    label = f"{weight}{' [dyn]' if dynamic else ''}"
    row: dict = {"model": label}
    try:
        raw = export_raw(weight, dynamic=dynamic)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"export failed: {e}"
        traceback.print_exc()
        return row

    row["raw_nodes"] = _node_count(raw)
    feeds = _feeds(raw)
    try:
        ref = _run(raw, feeds)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"raw model failed to run: {e}"
        return row

    for tool, fn in (("sim", _slim_with_onnxsim), ("slim", _slim_with_onnxslim)):
        try:
            t = time.time()
            out, ok = fn(onnx.ModelProto.FromString(raw.SerializeToString()))
            row[f"{tool}_seconds"] = round(time.time() - t, 2)
            row[f"{tool}_nodes"] = _node_count(out)
            row[f"{tool}_check_ok"] = bool(ok)
            try:
                test = _run(out, feeds)
                row[f"{tool}_valid"] = True
                row[f"{tool}_parity"] = _parity(ref, test)
            except Exception as e:  # noqa: BLE001
                row[f"{tool}_valid"] = False
                row[f"{tool}_parity"] = float("inf")
                row[f"{tool}_run_error"] = str(e)
            row[f"{tool}_status"] = "ok"
        except Exception as e:  # noqa: BLE001
            row[f"{tool}_status"] = "crash"
            row[f"{tool}_error"] = str(e)
            traceback.print_exc()
    return row


def _fmt_parity(v) -> str:
    if v is None:
        return "-"
    if v == float("inf"):
        return "MISMATCH"
    return f"{v:.2e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--dynamic-detect",
        action="store_true",
        help="also export yolo11n.pt with dynamic axes (stresses dynamic-shape passes)",
    )
    args = ap.parse_args()

    models = args.models or DEFAULT_MODELS
    rows = [evaluate(m) for m in models]
    if args.dynamic_detect:
        rows.append(evaluate("yolo11n.pt", dynamic=True))

    print("\n\n================ onnxsim vs onnxslim in ultralytics ================\n")
    hdr = (
        f"{'model':<18} {'raw':>5} | {'onnxsim nodes/valid/parity/ok':<34} | "
        f"{'onnxslim nodes/valid/parity':<26}"
    )
    print(hdr)
    print("-" * len(hdr))
    regressions = []
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<18} EXPORT ERROR: {r['error']}")
            regressions.append(r["model"])
            continue
        sim_ok = r.get("sim_valid") and r.get("sim_parity", float("inf")) <= (ATOL + RTOL)
        slim_ok = r.get("slim_valid") and r.get("slim_parity", float("inf")) <= (ATOL + RTOL)
        sim_cell = (
            f"{r.get('sim_nodes','-'):>5} "
            f"{'Y' if r.get('sim_valid') else 'N'} "
            f"{_fmt_parity(r.get('sim_parity')):>9} "
            f"{'ok' if r.get('sim_check_ok') else 'no':>3}"
        )
        slim_cell = (
            f"{r.get('slim_nodes','-'):>5} "
            f"{'Y' if r.get('slim_valid') else 'N'} "
            f"{_fmt_parity(r.get('slim_parity')):>9}"
        )
        print(f"{r['model']:<18} {r.get('raw_nodes','-'):>5} | {sim_cell:<34} | {slim_cell:<26}")
        if not sim_ok:
            regressions.append(r["model"])

    print("\nLegend: nodes=node count after simplify, valid=loads+runs in ORT,")
    print("        parity=max abs diff vs raw export, ok=onnxsim's own check_ok.")
    print(f"        parity tolerance = atol {ATOL} + rtol {RTOL}\n")

    if regressions:
        print(f"RESULT: onnxsim NOT a clean drop-in for: {', '.join(regressions)}")
        return 1
    print("RESULT: onnxsim is a valid drop-in replacement for onnxslim on all tested models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
