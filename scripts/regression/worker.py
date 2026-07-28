#!/usr/bin/env python3
"""Download one onnxmodelzoo model and run onnxsim + onnxslim over it.

Prints one JSON result line (``__RESULT__<json>``) carrying both tools'
outcomes side by side, so the regression can compare them on the same real-world
graphs (robustness, optimization strength, speed / peak memory).

Isolation model
---------------
The orchestrator (default invocation) downloads the model once, counts its
nodes, then runs **each tool in its own child subprocess** (``--run-tool``) with
its own wall-clock timeout. An OOM / segfault / C++ abort / hang in one tool is
therefore contained and reported as that tool's failure, and it cannot corrupt
the other tool's result. Peak RSS is captured per tool from the child.

onnxsim remains the *authoritative* arm: its status is what the surrounding
harness gates the regression on. onnxslim is measured for comparison only and
never fails the run.

If onnxsim raises a C++ assertion from a specific optimizer pass (some passes
still abort on dynamic-shape graphs), the offending pass is detected from the
message, added to ``skipped_optimizers``, and simplification is retried -- so a
newly-fragile pass shows up in the summary instead of silently passing.

onnxslim is measured in two parts: an optimization call (robustness + speed +
node reduction) and, unless ``SLIM_CHECK=0``, a best-effort equivalence check
(``slim(model_check=True)`` returns ``None`` when the slimmed graph does not
match). A check that errors out (e.g. the model needs explicit input shapes to
run) leaves ``valid`` unknown rather than counting as an onnxslim crash.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = ("onnxsim", "onnxslim")


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def peak_rss_mb():
    # ru_maxrss is kilobytes on Linux; report MiB.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


# --------------------------------------------------------------------------- #
# Child mode: run exactly one tool on an already-downloaded .onnx and print
# __TOOLRESULT__<json>. Kept crash-safe so an import error or abort still yields
# a parseable line where possible (a hard segfault won't, and the orchestrator
# treats a missing line as a crash).
# --------------------------------------------------------------------------- #
def run_onnxsim(onnx_path):
    import onnx

    from onnxsim import simplify

    model = onnx.load(onnx_path)
    skipped = []
    while True:
        try:
            model_simp, check = simplify(model, skipped_optimizers=skipped or None)
            break
        except RuntimeError as e:
            m = re.search(r"passes/([A-Za-z0-9_]+)\.h", str(e))
            if not m or m.group(1) in skipped:
                raise
            skipped.append(m.group(1))
    return {
        "simp_nodes": len(model_simp.graph.node),
        "valid": bool(check),
        "status": "ok" if check else "unvalidated",
        "skipped_optimizers": skipped,
    }


def run_onnxslim(onnx_path):
    import onnx
    import onnxslim

    # Optimization pass: robustness + speed + node reduction. slim() mutates its
    # input in place, so it gets its own freshly-loaded model.
    model_simp = onnxslim.slim(onnx.load(onnx_path))
    out = {
        "simp_nodes": len(model_simp.graph.node),
        "valid": None,
        "status": "ok",
    }
    del model_simp

    # Best-effort equivalence check on a pristine reload. slim(model_check=True)
    # returns None when the slimmed graph fails onnxslim's own onnxruntime check;
    # a check that errors (model needs input shapes, unsupported op, ...) leaves
    # validity unknown rather than blaming onnxslim for a crash.
    if os.environ.get("SLIM_CHECK", "1") != "0":
        try:
            checked = onnxslim.slim(onnx.load(onnx_path), model_check=True)
            out["valid"] = checked is not None
            if checked is None:
                out["status"] = "unvalidated"
        except Exception:
            out["valid"] = None
    return out


def child(tool, onnx_path):
    res = {"tool": tool, "status": "error", "error": None, "simp_nodes": None,
           "valid": None, "seconds": None, "peak_rss_mb": None,
           "skipped_optimizers": []}
    t0 = time.time()
    try:
        runner = {"onnxsim": run_onnxsim, "onnxslim": run_onnxslim}[tool]
        res.update(runner(onnx_path))
    except Exception as e:
        res["status"] = "crash"
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-600:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
        res["peak_rss_mb"] = peak_rss_mb()
    print("__TOOLRESULT__" + json.dumps(res))
    return 0


def run_tool_subprocess(tool, onnx_path, timeout):
    """Run one tool child, returning its parsed result (never raises)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py"),
             "--run-tool", tool, onnx_path],
            capture_output=True, text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        return {"tool": tool, "status": "timeout", "error": f">{timeout}s",
                "simp_nodes": None, "valid": None,
                "seconds": float(timeout), "peak_rss_mb": None,
                "skipped_optimizers": []}
    for line in proc.stdout.splitlines():
        if line.startswith("__TOOLRESULT__"):
            return json.loads(line[len("__TOOLRESULT__"):])
    # No result line: the child died before it could report (segfault, OOM kill,
    # onnxslim not installed, ...).
    return {"tool": tool, "status": "crash",
            "error": (proc.stderr or proc.stdout or "no tool result line")[-300:],
            "simp_nodes": None, "valid": None,
            "seconds": round(time.time() - t0, 1), "peak_rss_mb": None,
            "skipped_optimizers": []}


# --------------------------------------------------------------------------- #
# Orchestrator mode: download once, then run each tool in its own child.
# --------------------------------------------------------------------------- #
def orchestrate(model_id, dl_dir, per_tool_timeout):
    res = {
        "model": model_id,
        "status": "error", "error": None,
        "orig_nodes": None, "orig_bytes": None,
        # onnxsim (authoritative arm -- these top-level names gate the run)
        "simp_nodes": None, "valid": None, "seconds": None,
        "skipped_optimizers": [], "peak_rss_mb": None,
        # onnxslim (comparison arm)
        "slim_status": None, "slim_simp_nodes": None, "slim_valid": None,
        "slim_seconds": None, "slim_error": None, "slim_peak_rss_mb": None,
    }
    local = os.path.join(dl_dir, model_id.replace("/", "__"))
    try:
        import onnx
        from huggingface_hub import snapshot_download

        # Grab the onnx graph + any external-data blobs; skip docs/metadata.
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local,
            allow_patterns=[
                "*.onnx", "*.onnx_data", "*.onnx.data", "*.data",
                "*.pb", "*.bin", "*.weight", "*.weights",
            ],
        )
        onnx_files = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".onnx"):
                    onnx_files.append(os.path.join(root, f))
        if not onnx_files:
            res["error"] = "no .onnx file in repo"
            res["status"] = "error"
            return res
        # If several graphs are present, the largest is the main model.
        onnx_path = max(onnx_files, key=os.path.getsize)
        res["orig_bytes"] = dir_size(path)
        res["orig_nodes"] = len(onnx.load(onnx_path).graph.node)

        sim = run_tool_subprocess("onnxsim", onnx_path, per_tool_timeout)
        slim = run_tool_subprocess("onnxslim", onnx_path, per_tool_timeout)

        res["status"] = sim["status"]
        res["error"] = sim.get("error")
        res["simp_nodes"] = sim.get("simp_nodes")
        res["valid"] = sim.get("valid")
        res["seconds"] = sim.get("seconds")
        res["skipped_optimizers"] = sim.get("skipped_optimizers", [])
        res["peak_rss_mb"] = sim.get("peak_rss_mb")

        res["slim_status"] = slim["status"]
        res["slim_error"] = slim.get("error")
        res["slim_simp_nodes"] = slim.get("simp_nodes")
        res["slim_valid"] = slim.get("valid")
        res["slim_seconds"] = slim.get("seconds")
        res["slim_peak_rss_mb"] = slim.get("peak_rss_mb")
    except Exception as e:
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        shutil.rmtree(local, ignore_errors=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--run-tool":
        sys.exit(child(sys.argv[2], sys.argv[3]))

    model_id = sys.argv[1]
    dl_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "dl")
    per_tool_timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = orchestrate(model_id, dl_dir, per_tool_timeout)
    print("__RESULT__" + json.dumps(out))
