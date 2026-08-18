#!/usr/bin/env python3
"""Check one model against the OpenVINO execution provider, in an isolated subprocess.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model
comes from the shared ``common/synthetic_models.py`` suite; with an
``onnx_path`` the graph is loaded from disk. Each model runs in its own
process (mirrors the QNN/Core ML harnesses) and the final stdout line is
exactly ``__RESULT__<json>``.

The check is framed as *original vs. simplified on the same OpenVINO
backend*, so a backend limitation (an op OpenVINO declines to place) cancels
out and only an onnxsim-introduced change can fail the run. For one model:

1. ``simplify`` it with onnxsim.
2. Run the **original** graph on the OpenVINO EP.
   * If that already fails to compile/run, the backend simply does not
     support this graph -- status ``unsupported`` (reported, not a failure).
3. Run the **simplified** graph on the OpenVINO EP.
   * If the original ran but the simplified does not, simplification broke
     OpenVINO compatibility -- status ``openvino_regression`` (a failure).
4. Compare the two OpenVINO outputs. Divergence beyond tolerance means
   simplification changed the on-device result -- ``openvino_regression``.
5. Also record the ONNX Runtime CPU reference diff and the OpenVINO coverage
   (full vs. partial) for both graphs, as information.

Status values:

* ``ok``                   - original & simplified both ran on OpenVINO and agreed.
* ``openvino_regression``  - the simplified graph broke OpenVINO compat or changed results.
* ``simplify_error``       - onnxsim raised on this model.
* ``unsupported``          - the OpenVINO backend could not handle even the original graph.
* ``skipped``              - the OpenVINO EP is not available on this host.
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
        "diff_vs_openvino_orig": None,
        "diff_vs_cpu_ref": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import onnx
        import openvino_backend as openvino

        if not openvino.OPENVINO_AVAILABLE:
            res["status"] = "skipped"
            res["error"] = openvino.unavailable_reason()
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

        feeds = openvino.random_feeds(model, seed=0)

        # ORT CPU reference (informational end-to-end check).
        cpu_ref = openvino.run_with_cpu(model, feeds)

        # Original graph on OpenVINO. If this fails, the backend can't handle
        # the graph at all -- not something onnxsim did.
        try:
            res["coverage_orig"] = openvino.coverage(model)
            openvino_orig = openvino.run_with_openvino(model, feeds)
        except Exception as exc:
            res["status"] = "unsupported"
            res["error"] = f"original graph not supported by OpenVINO: {exc}"
            return res

        # Simplified graph on OpenVINO. The original worked, so a failure
        # here is an onnxsim-introduced compatibility regression.
        try:
            res["coverage_simp"] = openvino.coverage(simp)
            openvino_simp = openvino.run_with_openvino(simp, feeds)
        except Exception as exc:
            res["status"] = "openvino_regression"
            res["error"] = f"simplified graph broke OpenVINO compile/run: {exc}"
            return res

        # Simplification must not change the on-device result.
        agree, diff_backend = openvino.compare(openvino_orig, openvino_simp)
        _, diff_ref = openvino.compare(cpu_ref, openvino_simp)
        res["diff_vs_openvino_orig"] = diff_backend
        res["diff_vs_cpu_ref"] = diff_ref
        if agree:
            res["status"] = "ok"
        else:
            res["status"] = "openvino_regression"
            res["error"] = (
                f"simplified OpenVINO output diverged from original "
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
