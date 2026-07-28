#!/usr/bin/env python3
"""Check one model against the QNN execution provider, in an isolated subprocess.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model comes
from the built-in ``models.py`` suite; with an ``onnx_path`` the graph is loaded
from disk (so the same worker handles real onnxmodelzoo models later). The QNN
HTP compiler can be heavy and, on some ops, abort at the C++ level, so each model
runs in its own process and the final stdout line is exactly ``__RESULT__<json>``
— the same protocol as the large-model regression harness.

The check is deliberately framed as *original vs. simplified on the same QNN
backend*, so that a backend limitation (an op the HTP x86 emulator will not
finalize in fp32, say) cancels out and only an onnxsim-introduced change can
fail the run. For one model:

1. ``simplify`` it with onnxsim.
2. Build deterministic random inputs.
3. Run the **original** graph on the QNN EP.
   * If that already fails to compile/run, the backend simply does not support
     this graph — status ``unsupported`` (reported, not a failure).
4. Run the **simplified** graph on the QNN EP.
   * If the original compiled but the simplified does not, simplification broke
     QNN compatibility — status ``qnn_regression`` (a failure).
5. Compare the two QNN outputs. Divergence beyond tolerance means simplification
   changed the on-device result — ``qnn_regression`` (a failure).
6. Also record the ONNX Runtime CPU reference diff and the QNN partition coverage
   (full vs. partial) for both graphs, as information.

Status values:

* ``ok``             - original & simplified both ran on QNN and agreed.
* ``qnn_regression`` - the simplified graph broke QNN compat or changed results.
* ``simplify_error`` - onnxsim raised on this model.
* ``unsupported``    - the QNN backend could not handle even the original graph.
* ``skipped``        - the QNN EP is not available on this host.
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
        "diff_vs_qnn_orig": None,
        "diff_vs_cpu_ref": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import onnx
        import qnn_backend as qnn

        if not qnn.QNN_AVAILABLE:
            res["status"] = "skipped"
            res["error"] = qnn.unavailable_reason()
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

        feeds = qnn.random_feeds(model, seed=0)

        # ORT CPU reference (informational end-to-end check).
        cpu_ref = qnn.run_with_cpu(model, feeds)

        # Original graph on QNN. If this fails, the backend can't handle the
        # graph at all — not something onnxsim did.
        try:
            res["coverage_orig"] = qnn.coverage(model)
            qnn_orig = qnn.run_with_qnn(model, feeds)
        except Exception as exc:
            res["status"] = "unsupported"
            res["error"] = f"original graph not supported by QNN: {exc}"
            return res

        # Simplified graph on QNN. The original worked, so a failure here is an
        # onnxsim-introduced compatibility regression.
        try:
            res["coverage_simp"] = qnn.coverage(simp)
            qnn_simp = qnn.run_with_qnn(simp, feeds)
        except Exception as exc:
            res["status"] = "qnn_regression"
            res["error"] = f"simplified graph broke QNN compile/run: {exc}"
            return res

        # Simplification must not change the on-device result.
        agree, diff_backend = qnn.compare(qnn_orig, qnn_simp)
        _, diff_ref = qnn.compare(cpu_ref, qnn_simp)
        res["diff_vs_qnn_orig"] = diff_backend
        res["diff_vs_cpu_ref"] = diff_ref
        if agree:
            res["status"] = "ok"
        else:
            res["status"] = "qnn_regression"
            res["error"] = (
                f"simplified QNN output diverged from original "
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
