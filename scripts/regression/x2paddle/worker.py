#!/usr/bin/env python3
"""Convert one ONNX model to PaddlePaddle with X2Paddle, twice, and compare.

Why this exists
---------------
`onnxsim` is not just a peer of X2Paddle -- it is a dependency *inside* it.
X2Paddle's ONNX front-end (`x2paddle.convert.onnx2paddle`, the default
`enable_optim=True` path) runs ``onnxsim.simplify`` on the graph before handing
it to the op-mapper. So an onnxsim regression can break X2Paddle conversions
directly: a crash, or a graph onnxsim rewrites into something the op-mapper can
no longer translate.

This worker measures exactly that. For one model it runs two arms:

* **baseline** -- the *original* ONNX graph straight into X2Paddle's
  decoder -> op-mapper -> optimizer -> ``gen_model`` (onnxsim OFF).
* **onnxsim** -- ``onnxsim.simplify`` first, then the *simplified* graph
  through the same X2Paddle stages.

The regression signal is the difference between the arms: if X2Paddle converts
the original but *fails* on the onnxsim-simplified graph, onnxsim introduced a
downstream regression. If X2Paddle can't convert the original either, that's an
X2Paddle limitation on that model (an unsupported op, say) and onnxsim is not to
blame -- recorded, but not a failure.

Isolation model
---------------
Every heavy step runs in its own child subprocess with its own wall-clock
timeout (``--onnxsim`` / ``--convert``), for two reasons:

1. onnxsim's optimizer passes are C++ and some still *abort* on odd graphs; a
   segfault/abort/hang in one step is contained and reported as that step's
   failure instead of taking the model (or the other arm) down.
2. **Paddle's parameter registry is process-global.** Converting two graphs in
   one interpreter makes the second collide with the first
   (``parameter name [...] have be been used``). Separate processes are the only
   clean way to convert both arms of the same model.

If onnxsim raises a C++ assertion from a specific optimizer pass, that pass is
detected from the message, added to ``skipped_optimizers``, and simplification
is retried -- so a newly-fragile pass shows up in the summary (matching the
main onnxmodelzoo regression's behaviour).

Environment note
----------------
X2Paddle 1.6.0 targets ONNX opset <= 15 and needs an ONNX build that still ships
``onnx.mapping`` (i.e. ``onnx<1.16``); its own ``onnx2paddle`` entry point also
mis-detects ``onnx>=1.16`` as "onnx not installed". This worker therefore drives
X2Paddle's conversion stages directly rather than through ``onnx2paddle``, which
both dodges that version gate and lets us turn onnxsim on/off per arm.
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
# Child: onnxsim step. Simplify <in> -> <out>, print __STEPRESULT__<json>.
# --------------------------------------------------------------------------- #
def child_onnxsim(in_path, out_path):
    res = {
        "step": "onnxsim",
        "status": "error",
        "error": None,
        "simp_nodes": None,
        "valid": None,
        "skipped_optimizers": [],
        "seconds": None,
        "peak_rss_mb": None,
    }
    t0 = time.time()
    try:
        import onnx

        from onnxsim import simplify

        model = onnx.load(in_path)
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
        onnx.save(model_simp, out_path)
        res["simp_nodes"] = len(model_simp.graph.node)
        res["valid"] = bool(check)
        res["skipped_optimizers"] = skipped
        res["status"] = "ok" if check else "unvalidated"
    except Exception as e:
        res["status"] = "crash"
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-600:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
        res["peak_rss_mb"] = peak_rss_mb()
    print("__STEPRESULT__" + json.dumps(res))
    return 0


# --------------------------------------------------------------------------- #
# Child: X2Paddle conversion of one graph. Drives the same stages that
# x2paddle.convert.onnx2paddle runs after its (broken-on-new-onnx) version gate.
# --------------------------------------------------------------------------- #
def child_convert(arm, onnx_path, save_dir):
    res = {
        "step": "convert",
        "arm": arm,
        "status": "error",
        "error": None,
        "paddle_ops": None,
        "produced_model": False,
        "seconds": None,
        "peak_rss_mb": None,
    }
    t0 = time.time()
    try:
        import warnings

        warnings.filterwarnings("ignore")

        from x2paddle.decoder.onnx_decoder import ONNXDecoder
        from x2paddle.op_mapper.onnx2paddle.onnx_op_mapper import ONNXOpMapper
        from x2paddle.optimizer.optimizer import GraphOptimizer

        decoder = ONNXDecoder(onnx_path, enable_onnx_checker=True)
        mapper = ONNXOpMapper(decoder)
        mapper.paddle_graph.build()
        res["paddle_ops"] = len(mapper.paddle_graph.layers)
        GraphOptimizer(source_frame="onnx").optimize(mapper.paddle_graph)
        mapper.paddle_graph.gen_model(save_dir)
        res["produced_model"] = os.path.isdir(os.path.join(save_dir, "inference_model"))
        res["status"] = "ok" if res["produced_model"] else "unvalidated"
    except Exception as e:
        res["status"] = "crash"
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-600:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
        res["peak_rss_mb"] = peak_rss_mb()
    print("__STEPRESULT__" + json.dumps(res))
    return 0


def run_step(args, timeout):
    """Run one child step, returning its parsed result (never raises)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py"), *args],
            capture_output=True,
            text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f">{timeout}s",
            "seconds": float(timeout),
            "peak_rss_mb": None,
        }
    for line in proc.stdout.splitlines():
        if line.startswith("__STEPRESULT__"):
            return json.loads(line[len("__STEPRESULT__") :])
    # No result line: the child died before it could report (segfault, OOM, ...).
    return {
        "status": "crash",
        "error": (proc.stderr or proc.stdout or "no step result line")[-300:],
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": None,
    }


def verdict(onnxsim, baseline, simp):
    """Classify one model from its three step results.

    Gating verdicts (fail the regression):
      onnxsim_fail -- onnxsim crashed, timed out, or failed its own check.
      regression   -- X2Paddle converted the original but NOT the
                      onnxsim-simplified graph: onnxsim broke a working
                      downstream conversion.
    Non-gating verdicts:
      pass                  -- onnxsim ok and the simplified graph converts.
      baseline_unsupported  -- X2Paddle can't convert the original either;
                               onnxsim is not implicated.
      improved              -- original failed, simplified converts (onnxsim
                               actually unblocked X2Paddle).
    """
    if onnxsim.get("status") not in ("ok",):
        return "onnxsim_fail"
    base_ok = baseline.get("status") == "ok"
    simp_ok = (simp or {}).get("status") == "ok"
    if base_ok and simp_ok:
        return "pass"
    if base_ok and not simp_ok:
        return "regression"
    if not base_ok and simp_ok:
        return "improved"
    return "baseline_unsupported"


# --------------------------------------------------------------------------- #
# Orchestrator: download once, run onnxsim + both conversions in child procs.
# --------------------------------------------------------------------------- #
def orchestrate(model_id, dl_dir, per_step_timeout):
    res = {
        "model": model_id,
        "status": "error",
        "verdict": "error",
        "error": None,
        "orig_nodes": None,
        "orig_bytes": None,
        # onnxsim arm
        "onnxsim_status": None,
        "simp_nodes": None,
        "onnxsim_valid": None,
        "onnxsim_seconds": None,
        "skipped_optimizers": [],
        "onnxsim_peak_rss_mb": None,
        # baseline conversion (original graph -> X2Paddle)
        "baseline_conv_status": None,
        "baseline_ops": None,
        "baseline_conv_seconds": None,
        "baseline_peak_rss_mb": None,
        "baseline_error": None,
        # onnxsim-arm conversion (simplified graph -> X2Paddle)
        "simp_conv_status": None,
        "simp_ops": None,
        "simp_conv_seconds": None,
        "simp_peak_rss_mb": None,
        "simp_error": None,
    }
    local = os.path.join(dl_dir, model_id.replace("/", "__"))
    save_root = local + "__pd"
    try:
        import onnx
        from huggingface_hub import snapshot_download

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
        onnx_files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(path)
            for f in fs
            if f.endswith(".onnx")
        ]
        if not onnx_files:
            res["error"] = "no .onnx file in repo"
            return res
        onnx_path = max(onnx_files, key=os.path.getsize)
        res["orig_bytes"] = dir_size(path)
        res["orig_nodes"] = len(onnx.load(onnx_path).graph.node)

        simp_path = onnx_path[:-5] + ".onnxsim.onnx"
        onnxsim = run_step(["--onnxsim", onnx_path, simp_path], per_step_timeout)
        res["onnxsim_status"] = onnxsim.get("status")
        res["simp_nodes"] = onnxsim.get("simp_nodes")
        res["onnxsim_valid"] = onnxsim.get("valid")
        res["onnxsim_seconds"] = onnxsim.get("seconds")
        res["skipped_optimizers"] = onnxsim.get("skipped_optimizers", []) or []
        res["onnxsim_peak_rss_mb"] = onnxsim.get("peak_rss_mb")
        res["error"] = onnxsim.get("error")

        baseline = run_step(
            ["--convert", "baseline", onnx_path, save_root + "_base"], per_step_timeout
        )
        res["baseline_conv_status"] = baseline.get("status")
        res["baseline_ops"] = baseline.get("paddle_ops")
        res["baseline_conv_seconds"] = baseline.get("seconds")
        res["baseline_peak_rss_mb"] = baseline.get("peak_rss_mb")
        res["baseline_error"] = baseline.get("error")

        simp = None
        if onnxsim.get("status") == "ok" and os.path.exists(simp_path):
            simp = run_step(
                ["--convert", "onnxsim", simp_path, save_root + "_simp"],
                per_step_timeout,
            )
            res["simp_conv_status"] = simp.get("status")
            res["simp_ops"] = simp.get("paddle_ops")
            res["simp_conv_seconds"] = simp.get("seconds")
            res["simp_peak_rss_mb"] = simp.get("peak_rss_mb")
            res["simp_error"] = simp.get("error")

        res["verdict"] = verdict(onnxsim, baseline, simp)
        res["status"] = (
            "ok"
            if res["verdict"] in ("pass", "baseline_unsupported", "improved")
            else "fail"
        )
    except Exception as e:
        res["status"] = "error"
        res["verdict"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        shutil.rmtree(local, ignore_errors=True)
        shutil.rmtree(save_root + "_base", ignore_errors=True)
        shutil.rmtree(save_root + "_simp", ignore_errors=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--onnxsim":
        sys.exit(child_onnxsim(sys.argv[2], sys.argv[3]))
    if len(sys.argv) >= 5 and sys.argv[1] == "--convert":
        sys.exit(child_convert(sys.argv[2], sys.argv[3], sys.argv[4]))

    model_id = sys.argv[1]
    dl_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "dl")
    per_step_timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = orchestrate(model_id, dl_dir, per_step_timeout)
    print("__RESULT__" + json.dumps(out))
