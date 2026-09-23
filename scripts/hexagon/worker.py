#!/usr/bin/env python3
"""Check one model against tinygrad's Hexagon (DSP) / Metal devices, in an isolated subprocess.

Run as ``worker.py <model_name> [onnx_path] [--device DSP|METAL]``. With just
a name, the model comes from the shared ``common/synthetic_models.py`` suite;
with an ``onnx_path`` the graph is loaded from disk. Each model runs in its
own process (mirrors the Core ML / QNN harnesses) and the final stdout line
is exactly ``__RESULT__<json>``.

The check is framed as *original vs. simplified on the same tinygrad
device*, so a backend limitation (an op tinygrad's ``OnnxRunner`` does not
implement on that device, or a kernel the mock-DSP clang cannot build)
cancels out and only an onnxsim-introduced change can fail the run. For one
model:

1. ``simplify`` it with onnxsim.
2. Run the **original** graph on the device.
   * If that already fails to import/run, the backend simply does not
     support this graph -- status ``unsupported`` (reported, not a failure).
3. Run the **simplified** graph on the device.
   * If the original ran but the simplified does not, simplification broke
     device compatibility -- status ``hexagon_regression`` / 
     ``metal_regression`` (a failure).
4. Compare the two on-device outputs. Divergence beyond tolerance means
   simplification changed the on-device result -- a regression.
5. Also record the tinygrad-default-device reference diff, as information.

Status values (``<dev>`` is ``hexagon`` or ``metal``):

* ``ok``                  - original & simplified both ran and agreed.
* ``<dev>_regression``    - the simplified graph broke compat or changed results.
* ``simplify_error``      - onnxsim raised on this model.
* ``unsupported``         - the device backend could not handle even the original graph.
* ``skipped``             - the device is not available on this host.
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


def check(
    model_name: str, onnx_path: str | None, device: str | None = None
) -> dict:
    tag = (device or "DSP").lower()
    tag = "hexagon" if tag != "metal" else "metal"
    res = {
        "model": model_name,
        "device": device,
        "status": "error",
        "orig_nodes": None,
        "simp_nodes": None,
        "diff_vs_orig": None,
        "diff_vs_reference": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import tinygrad_hexagon_backend as backend
        import onnx

        ok, reason = backend.probe_device(device or backend.HEXAGON_DEVICE)
        if not ok:
            res["status"] = "skipped"
            res["error"] = reason
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

        feeds = backend.random_feeds(model, seed=0)

        # tinygrad default-device reference (informational end-to-end check).
        reference = backend.run_with_device(model, feeds, None)

        # Original graph on the device. If this fails, the backend can't
        # handle the graph at all -- not something onnxsim did.
        try:
            dev_orig = backend.run_with_device(model, feeds, device)
        except Exception as exc:
            res["status"] = "unsupported"
            res["error"] = f"original graph not supported on {device}: {exc}"
            return res

        # Simplified graph on the device. The original worked, so a failure
        # here is an onnxsim-introduced compatibility regression.
        try:
            dev_simp = backend.run_with_device(simp, feeds, device)
        except Exception as exc:
            res["status"] = f"{tag}_regression"
            res["error"] = (
                f"simplified graph broke {device} compile/run: {exc}"
            )
            return res

        # Simplification must not change the on-device result.
        agree, diff_backend = backend.compare(dev_orig, dev_simp)
        _, diff_ref = backend.compare(reference, dev_simp)
        res["diff_vs_orig"] = diff_backend
        res["diff_vs_reference"] = diff_ref
        if agree:
            res["status"] = "ok"
        else:
            res["status"] = f"{tag}_regression"
            res["error"] = (
                f"simplified {device} output diverged from original "
                f"(max_abs_diff={diff_backend:.3g})"
            )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model_name")
    ap.add_argument("onnx_path", nargs="?")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import tinygrad_hexagon_backend as _backend

    print(
        "__RESULT__"
        + json.dumps(check(args.model_name, args.onnx_path, args.device or _backend.HEXAGON_DEVICE))
    )
