#!/usr/bin/env python3
"""Check one model against the static Pulsar2 (Axera NPU) coverage heuristic.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model
comes from the shared ``common/synthetic_models.py`` suite; with an
``onnx_path`` the graph is loaded from disk. The final stdout line is exactly
``__RESULT__<json>``.

Unlike the QNN/OpenVINO/MIGraphX workers, there is no real compiler to run
the graph through (see ``pulsar2_backend.py``), so there is nothing to crash
at the C++ level -- this still runs each model in its own process to match
the sibling harnesses' shape and keep ``run_pulsar2_compat.py`` generic. For
one model:

0. If the model already has an Axera-compiled NPU subgraph node (i.e. it's
   a real `.axmodel`, not pre-compile ONNX) -- ``pulsar2_unsafe_for_simplify``
   *without calling* ``simplify()`` at all. Confirmed on a real AX650N with a
   real `.axmodel`: onnxsim currently corrupts these unconditionally (see
   ``pulsar2_ops.py``'s docstring for the full, exhausted list of parameter
   combinations that were tried and did not help) -- there is no safe way to
   run this step for such a model.
1. Otherwise, ``simplify`` it with onnxsim.
2. Compute the static Pulsar2-NPU-blocker set for the original and the
   simplified graph (``pulsar2_ops.blocking_ops``).
3. If simplification introduced a blocking op type that wasn't already
   present -- ``pulsar2_regression`` (a failure): simplification likely
   pushed part of the graph off Pulsar2's NPU path.
4. If simplification dropped NPU weight/command data a compiled subgraph
   node still references (shouldn't be reachable given step 0, but checked
   anyway as defense in depth) -- ``pulsar2_data_corrupted``.
5. If onnxsim's own correctness check failed -- ``simplify_check_failed``.

Status values:

* ``ok``                          - simplified cleanly, no new Pulsar2-NPU blocker.
* ``pulsar2_unsafe_for_simplify`` - input already has a compiled NPU subgraph; skipped.
* ``pulsar2_regression``          - simplification introduced a new blocking op type.
* ``pulsar2_data_corrupted``      - simplification dropped NPU weight/command data.
* ``simplify_check_failed``       - onnxsim's own numeric check reported a mismatch.
* ``simplify_error``              - onnxsim raised on this model.
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
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import onnx
        import pulsar2_backend as pulsar2

        from onnxsim import simplify

        if onnx_path:
            model = onnx.load(onnx_path)
        else:
            models = fresh("models", HERE)

            model = models.build(model_name)
        res["orig_nodes"] = len(model.graph.node)

        if pulsar2.unsafe_for_simplify(model):
            res["status"] = "pulsar2_unsafe_for_simplify"
            res["error"] = (
                "model already has a compiled Axera NPU subgraph node; "
                "onnxsim is confirmed to corrupt these (see pulsar2_ops.py) "
                "-- skipped rather than run"
            )
            return res

        try:
            simp, check_ok = simplify(model)
        except Exception as exc:
            res["status"] = "simplify_error"
            res["error"] = f"{type(exc).__name__}: {exc}"
            return res
        res["simp_nodes"] = len(simp.graph.node)

        res["coverage_orig"] = pulsar2.coverage(model)
        res["coverage_simp"] = pulsar2.coverage(simp)

        new_blockers = sorted(pulsar2.new_blocking_op_types(model, simp))
        res["new_blocking_ops"] = new_blockers

        corrupted = sorted(pulsar2.stripped_npu_data(simp))

        if corrupted:
            res["status"] = "pulsar2_data_corrupted"
            res["error"] = (
                "simplification dropped NPU weight/command data still "
                f"referenced by a compiled subgraph node: {corrupted}"
            )
        elif new_blockers:
            res["status"] = "pulsar2_regression"
            res["error"] = (
                "simplification introduced op type(s) unlikely to be "
                f"Pulsar2-NPU-schedulable: {new_blockers}"
            )
        elif not check_ok:
            res["status"] = "simplify_check_failed"
            res["error"] = "onnxsim's own correctness check reported a mismatch"
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
