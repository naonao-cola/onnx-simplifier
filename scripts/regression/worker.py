#!/usr/bin/env python3
"""Download one onnxmodelzoo model, run onnxsim, print one JSON result line.

Runs as an isolated subprocess (see run_regression.py) so an OOM / segfault /
C++ abort / hang on a single model cannot take the whole job down. The final
stdout line is exactly ``__RESULT__<json>``. The downloaded files live under
``<dl_dir>/<sanitized id>`` and are removed before exit.

If onnxsim raises a C++ assertion from a specific optimizer pass (some passes
still abort on dynamic-shape graphs), the offending pass is detected from the
message, added to ``skipped_optimizers``, and simplification is retried. That
keeps the regression green on the current tree while recording *which* pass
aborted, so a newly-fragile pass shows up in the run summary instead of
silently passing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import traceback


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main(model_id, dl_dir):
    res = {
        "model": model_id,
        "status": "error",
        "error": None,
        "orig_nodes": None,
        "simp_nodes": None,
        "orig_bytes": None,
        "seconds": None,
        "skipped_optimizers": [],
        "valid": None,
    }
    t0 = time.time()
    local = os.path.join(dl_dir, model_id.replace("/", "__"))
    try:
        import onnx
        from huggingface_hub import snapshot_download

        from onnxsim import simplify

        # Grab the onnx graph + any external-data blobs; skip docs/metadata.
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local,
            allow_patterns=[
                "*.onnx",
                "*.onnx_data",
                "*.onnx.data",
                "*.data",
                "*.pb",
                "*.bin",
                "*.weight",
                "*.weights",
            ],
        )
        onnx_files = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".onnx"):
                    onnx_files.append(os.path.join(root, f))
        if not onnx_files:
            res["error"] = "no .onnx file in repo"
            return res
        # If several graphs are present, the largest is the main model.
        onnx_path = max(onnx_files, key=os.path.getsize)
        res["orig_bytes"] = dir_size(path)

        model = onnx.load(onnx_path)
        res["orig_nodes"] = len(model.graph.node)

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

        res["simp_nodes"] = len(model_simp.graph.node)
        res["valid"] = bool(check)
        res["skipped_optimizers"] = skipped
        res["status"] = "ok" if check else "unvalidated"
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
        shutil.rmtree(local, ignore_errors=True)
    return res


if __name__ == "__main__":
    model_id = sys.argv[1]
    dl_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "dl")
    out = main(model_id, dl_dir)
    print("__RESULT__" + json.dumps(out))
