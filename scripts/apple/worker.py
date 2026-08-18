#!/usr/bin/env python3
"""Check one model against the Core ML execution provider, in an isolated subprocess.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model
comes from the shared ``common/synthetic_models.py`` suite; with an
``onnx_path`` the graph is loaded from disk. Each model runs in its own
process (mirrors the QNN harness) and the final stdout line is exactly
``__RESULT__<json>``.

The check is framed as *original vs. simplified on the same Core ML backend*,
so a backend limitation (an op Core ML declines to place) cancels out and only
an onnxsim-introduced change can fail the run. For one model:

1. ``simplify`` it with onnxsim.
2. Run the **original** graph on the Core ML EP.
   * If that already fails to compile/run, the backend simply does not
     support this graph -- status ``unsupported`` (reported, not a failure).
3. Run the **simplified** graph on the Core ML EP.
   * If the original ran but the simplified does not, simplification broke
     Core ML compatibility -- status ``coreml_regression`` (a failure).
4. Compare the two Core ML outputs. Divergence beyond tolerance means
   simplification changed the on-device result -- ``coreml_regression``.
5. Also record the ONNX Runtime CPU reference diff and the Core ML coverage
   (full vs. partial) for both graphs, as information.

Status values:

* ``ok``                 - original & simplified both ran on Core ML and agreed.
* ``coreml_regression``  - the simplified graph broke Core ML compat or changed results.
* ``simplify_error``     - onnxsim raised on this model.
* ``unsupported``        - the Core ML backend could not handle even the original graph.
* ``skipped``            - the Core ML EP is not available on this host.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def check(model_name: str, onnx_path: str | None) -> dict:
    res = {
        "model": model_name,
        "status": "error",
        "orig_nodes": None,
        "simp_nodes": None,
        "coverage_orig": None,
        "coverage_simp": None,
        "diff_vs_coreml_orig": None,
        "diff_vs_cpu_ref": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import coreml_backend as coreml
        import onnx

        if not coreml.COREML_AVAILABLE:
            res["status"] = "skipped"
            res["error"] = coreml.unavailable_reason()
            return res

        from onnxsim import simplify

        if onnx_path:
            model = onnx.load(onnx_path)
        else:
            import models

            model = models.build(model_name)
        res["orig_nodes"] = len(model.graph.node)

        try:
            simp, _check_ok = simplify(model)
        except Exception as exc:
            res["status"] = "simplify_error"
            res["error"] = f"{type(exc).__name__}: {exc}"
            return res
        res["simp_nodes"] = len(simp.graph.node)

        feeds = coreml.random_feeds(model, seed=0)

        # ORT CPU reference (informational end-to-end check).
        cpu_ref = coreml.run_with_cpu(model, feeds)

        # Original graph on Core ML. If this fails, the backend can't handle
        # the graph at all -- not something onnxsim did.
        try:
            res["coverage_orig"] = coreml.coverage(model)
            coreml_orig = coreml.run_with_coreml(model, feeds)
        except Exception as exc:
            res["status"] = "unsupported"
            res["error"] = f"original graph not supported by Core ML: {exc}"
            return res

        # Simplified graph on Core ML. The original worked, so a failure here
        # is an onnxsim-introduced compatibility regression.
        try:
            res["coverage_simp"] = coreml.coverage(simp)
            coreml_simp = coreml.run_with_coreml(simp, feeds)
        except Exception as exc:
            res["status"] = "coreml_regression"
            res["error"] = f"simplified graph broke Core ML compile/run: {exc}"
            return res

        # Simplification must not change the on-device result.
        agree, diff_backend = coreml.compare(coreml_orig, coreml_simp)
        _, diff_ref = coreml.compare(cpu_ref, coreml_simp)
        res["diff_vs_coreml_orig"] = diff_backend
        res["diff_vs_cpu_ref"] = diff_ref
        if agree:
            res["status"] = "ok"
        else:
            res["status"] = "coreml_regression"
            res["error"] = (
                f"simplified Core ML output diverged from original "
                f"(max_abs_diff={diff_backend:.3g})"
            )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    name = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else None
    print("__RESULT__" + json.dumps(check(name, path)))
