#!/usr/bin/env python3
"""Check one model against the static TinyEngine coverage heuristic.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model
comes from ``models.py``; with an ``onnx_path`` the graph is loaded from
disk. The final stdout line is exactly ``__RESULT__<json>``.

Like ``scripts/edgeai/worker.py``, there is no real compiler to run the
graph through (see ``tinyengine_backend.py``), so there is nothing to crash
at the C++/native level -- this still runs each model in its own process to
match the sibling harnesses' shape and keep ``run_tinyengine_compat.py``
generic. For one model:

1. ``simplify`` it with onnxsim (``check_n=0``, the default).
2. Compute the static TinyEngine-blocker set for the original and the
   simplified graph (``tinyengine_ops.blocking_ops``), plus the
   static-shape risk (``tinyengine_backend.dynamic_shape_risks``).
3. If simplification introduced a blocking op type that wasn't already
   present -- ``tinyengine_regression`` (a failure): unlike TIDL's "part of
   the graph loses accelerator eligibility", this means "the whole compile
   would now fail outright" (TinyEngine has no partial-coverage fallback --
   see ``tinyengine_ops.py``'s docstring).
4. If onnxsim raised -- ``simplify_error``.

Status values:

* ``ok``                    - simplified cleanly, no new TinyEngine blocker.
* ``tinyengine_regression`` - simplification introduced a new blocking op type.
* ``simplify_error``        - onnxsim raised on this model.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# _local_import.fresh() is generic, vendor-agnostic tooling that lives under
# scripts/axera -- see scripts/edgeai/worker.py's docstring for why it's
# reused rather than duplicated here.
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
        "skip_op_risks_orig": None,
        "skip_op_risks_simp": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import onnx
        import tinyengine_backend as tinyengine

        from onnxsim import simplify

        if onnx_path:
            model = onnx.load(onnx_path)
        else:
            models = fresh("models", HERE)
            model = models.build(model_name)
        res["orig_nodes"] = len(model.graph.node)
        res["coverage_orig"] = tinyengine.coverage(model)
        res["shape_risks_orig"] = tinyengine.dynamic_shape_risks(model)
        res["skip_op_risks_orig"] = tinyengine.skip_op_risks(model)

        try:
            simp, _check_ok = simplify(model)
        except Exception as exc:
            res["status"] = "simplify_error"
            res["error"] = f"{type(exc).__name__}: {exc}"
            return res
        res["simp_nodes"] = len(simp.graph.node)
        res["coverage_simp"] = tinyengine.coverage(simp)
        res["shape_risks_simp"] = tinyengine.dynamic_shape_risks(simp)
        res["skip_op_risks_simp"] = tinyengine.skip_op_risks(simp)

        new_blockers = sorted(tinyengine.new_blocking_op_types(model, simp))
        res["new_blocking_ops"] = new_blockers

        if new_blockers:
            res["status"] = "tinyengine_regression"
            res["error"] = (
                "simplification introduced op type(s) with no TinyEngine "
                f"dispatch case, which fails the whole compile: {new_blockers}"
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
