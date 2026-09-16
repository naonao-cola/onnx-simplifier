#!/usr/bin/env python3
"""Check one model against the static TIDL (TI edgeai) coverage heuristic.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model
comes from ``models.py`` (the shared ``common/synthetic_models.py`` suite
plus the edgeai-specific dynamic-batch fixture); with an ``onnx_path`` the
graph is loaded from disk. The final stdout line is exactly
``__RESULT__<json>``.

Unlike the QNN/OpenVINO/MIGraphX workers, there is no real compiler to run
the graph through (see ``tidl_backend.py``), so there is nothing to crash at
the C++ level -- this still runs each model in its own process to match the
sibling harnesses' shape and keep ``run_tidl_compat.py`` generic. For one
model:

1. ``simplify`` it with onnxsim (``check_n=0``, the default -- this harness
   checks static op/shape coverage, not onnxsim's own numeric correctness).
2. Compute the static TIDL-blocker set for the original and the simplified
   graph (``tidl_ops.blocking_ops``), plus the static-shape risk
   (``tidl_backend.dynamic_shape_risks``).
3. If simplification introduced a blocking op type that wasn't already
   present -- ``tidl_regression`` (a failure): simplification likely pushed
   part of the graph off TIDL's accelerator path.
4. If onnxsim raised -- ``simplify_error``.

Status values:

* ``ok``              - simplified cleanly, no new TIDL blocker.
* ``tidl_regression`` - simplification introduced a new blocking op type.
* ``simplify_error``  - onnxsim raised on this model.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# `_local_import.fresh()` is a generic, vendor-agnostic utility that happens
# to live under scripts/axera (it was written there first, to fix the same
# bare-name "models"/"worker" sys.modules collision this directory's own
# models.py would otherwise hit -- see that module's docstring). Reused here
# rather than duplicated.
_AXERA_DIR = os.path.join(os.path.dirname(HERE), "axera")
for _dir in (HERE, _AXERA_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from _local_import import fresh  # noqa: E402


def check(model_name: str, onnx_path: str | None) -> dict:
    res = {
        "model": model_name,
        "status": "error",
        "orig_nodes": None,
        "simp_nodes": None,
        "coverage_orig": None,
        "coverage_simp": None,
        "new_blocking_ops": None,
        "shape_risks_orig": None,
        "shape_risks_simp": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import onnx
        import tidl_backend as tidl

        from onnxsim import simplify

        if onnx_path:
            model = onnx.load(onnx_path)
        else:
            models = fresh("models", HERE)
            model = models.build(model_name)
        res["orig_nodes"] = len(model.graph.node)
        res["coverage_orig"] = tidl.coverage(model)
        res["shape_risks_orig"] = tidl.dynamic_shape_risks(model)

        try:
            simp, _check_ok = simplify(model)
        except Exception as exc:
            res["status"] = "simplify_error"
            res["error"] = f"{type(exc).__name__}: {exc}"
            return res
        res["simp_nodes"] = len(simp.graph.node)
        res["coverage_simp"] = tidl.coverage(simp)
        res["shape_risks_simp"] = tidl.dynamic_shape_risks(simp)

        new_blockers = sorted(tidl.new_blocking_op_types(model, simp))
        res["new_blocking_ops"] = new_blockers

        if new_blockers:
            res["status"] = "tidl_regression"
            res["error"] = (
                "simplification introduced op type(s) unlikely to be "
                f"TIDL-accelerator-schedulable: {new_blockers}"
            )
        else:
            res["status"] = "ok"
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
