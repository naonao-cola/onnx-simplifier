#!/usr/bin/env python3
"""Export RF-DETR model variants to ONNX and simplify them with onnxsim.

RF-DETR (https://github.com/roboflow/rf-detr) is Roboflow's real-time
DETR-style detector/segmenter. Its ``[onnx]`` extra already depends on
onnxsim (pinned ``<0.6.0``) and calls ``onnxsim.simplify(..., check_n=3)``
internally (see ``rfdetr/export/_onnx/exporter.py::onnx_simplify``).

This script is a standalone compatibility harness: it exports each RF-DETR
variant to ONNX via the public ``RFDETR*.export()`` API and then runs the
*current* onnxsim on the result, reproducing RF-DETR's own simplify call
(``check_n=3``, real ``input_data``, ``dynamic_input_shape=False``). It
reports node counts before/after and whether onnxsim's numerical
equivalence check passed.

Requirements (all from PyPI)::

    pip install rfdetr onnxruntime onnxsim

``onnxruntime`` is not optional here: onnxsim uses it to numerically compare
the original and simplified graphs (``check_n``). Without it, onnxsim falls
back to onnx's Python reference evaluator, whose ``GatherElements``
implementation (``numpy.choose``, capped at 64 choices) cannot execute
RF-DETR's decoder and the check crashes rather than simplifies. Installing
onnxruntime -- which RF-DETR's ``[onnx]`` extra already requires -- avoids
this.

Usage::

    python simplify_rfdetr.py                 # a small default set
    python simplify_rfdetr.py RFDETRNano RFDETRSegNano
    python simplify_rfdetr.py --all           # every non-"plus" variant
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import onnx
import onnxsim

# Variants exported by ``rfdetr`` (see ``rfdetr.__init__``). XLarge/2XLarge
# segmentation variants additionally require ``pip install rfdetr[plus]``.
DEFAULT_VARIANTS = ["RFDETRNano", "RFDETRSegNano", "RFDETRKeypointPreview"]
ALL_VARIANTS = [
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRBase",
    "RFDETRLarge",
    "RFDETRSegPreview",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRKeypointPreview",
]


def run_variant(variant: str, output_dir: str) -> dict:
    """Export one RF-DETR variant and simplify it with onnxsim."""
    import rfdetr

    result: dict = {
        "variant": variant,
        "onnxsim_version": onnxsim.__version__,
        "onnx_version": onnx.__version__,
    }
    try:
        model = getattr(rfdetr, variant)()

        t0 = time.time()
        path = str(model.export(output_dir=output_dir, verbose=False))
        result["export_s"] = round(time.time() - t0, 1)
        result["onnx_path"] = path

        graph = onnx.load(path)
        result["nodes_before"] = len(graph.graph.node)

        # Reconstruct the (static) input tensor baked into the exported graph
        # so the simplifier's equivalence check runs on a real forward pass.
        inp = graph.graph.input[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        result["input_shape"] = dims
        data = np.random.RandomState(0).rand(*dims).astype(np.float32)

        # This mirrors rfdetr/export/_onnx/exporter.py::onnx_simplify.
        t0 = time.time()
        model_opt, check_ok = onnxsim.simplify(
            path,
            check_n=3,
            input_data={inp.name: data},
            dynamic_input_shape=False,
        )
        result["simplify_s"] = round(time.time() - t0, 1)
        result["check_ok"] = bool(check_ok)
        result["nodes_after"] = len(model_opt.graph.node)
        onnx.save(model_opt, path.replace(".onnx", ".sim.onnx"))
        result["status"] = "OK" if check_ok else "SIMPLIFY_CHECK_FAILED"
    except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", help="RF-DETR class names to test")
    parser.add_argument("--all", action="store_true", help="test all non-plus variants")
    parser.add_argument("--output-dir", default="rfdetr_onnx", help="ONNX output directory")
    args = parser.parse_args()

    if args.all:
        variants = ALL_VARIANTS
    elif args.variants:
        variants = args.variants
    else:
        variants = DEFAULT_VARIANTS

    results = []
    for variant in variants:
        print(f"=== {variant} ===", flush=True)
        result = run_variant(variant, args.output_dir)
        print(json.dumps(result), flush=True)
        results.append(result)

    print("\n===== SUMMARY =====")
    ok = 0
    for r in results:
        status = r.get("status")
        ok += status == "OK"
        print(
            f"{r['variant']:24s} {status:22s} "
            f"nodes {r.get('nodes_before', '?')}->{r.get('nodes_after', '?')} "
            f"check_ok={r.get('check_ok', '?')} {r.get('error', '')}"
        )
    print(f"\n{ok}/{len(results)} variants simplified with onnxsim {onnxsim.__version__}")


if __name__ == "__main__":
    main()
