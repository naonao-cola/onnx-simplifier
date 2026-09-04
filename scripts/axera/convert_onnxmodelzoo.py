#!/usr/bin/env python3
"""Real end-to-end conversion: `onnxmodelzoo` model -> onnxsim -> `pulsar2
build` (Docker) -> optionally run on a real AXCL device. Optionally profile.

This is the manual `resnet18d_Opset18`/`googlenet-6` workflow from
`pulsar2_ops.py`'s docstring, turned into a reusable batch driver over
`pulsar2_docker.py`. Unlike `screen_onnxmodelzoo.py` (fast, static,
Docker/device-free -- run that first), this does a **real** compile per
model and needs a loaded Pulsar2 Docker image (see `pulsar2_docker.py`'s
docstring for how to get one). A connected AXCL device is optional; without
one, latency/on-device diff columns are left blank.

For each model:

1. Fetch the ONNX (`scripts/regression/model_zoo.py`).
2. Skip (status `skipped_not_single_image_input`) unless it has exactly one
   non-initializer input of rank 4 -- this driver only knows the
   single-image-classifier config shape (see `pulsar2_docker.py`'s
   `_image_classifier_config`); a model with a different input shape needs
   a hand-written config and `pulsar2_docker.build(config_path=...)` used
   directly instead.
2. `onnxsim.simplify()` it.
3. `pulsar2 build --target_hardware AX650` both the original and the
   simplified ONNX, with `--profile` passing through to
   `pulsar2_docker.build(profile=...)` -- writes a `chrome://tracing`
   `trace.json` per build when set (see README's "Real NPU profiling").
4. If both compiled and a device is available: run both on-device with the
   same random input and diff the raw output bytes -- bit-identical is the
   expected, confirmed-safe result (see `pulsar2_ops.py`'s docstring).

Usage:
    python convert_onnxmodelzoo.py --models resnet18d_Opset18 --profile
    python convert_onnxmodelzoo.py --models resnet18d_Opset18 googlenet-6 \\
        --work-dir /tmp/pulsar2_work --output pulsar2-convert.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_REGRESSION_DIR = os.path.join(os.path.dirname(HERE), "regression")
if _REGRESSION_DIR not in sys.path:
    sys.path.insert(0, _REGRESSION_DIR)

# Generic ImageNet-style normalization -- a reasonable default for
# compatibility testing (does it compile? does onnxsim change the result?),
# not tuned per-model for deployment accuracy. See pulsar2_docker.py.
_DEFAULT_MEAN = [123.675, 116.28, 103.53]
_DEFAULT_STD = [58.395, 57.12, 57.375]


def _single_image_input(model) -> "str | None":
    """Return the input tensor's name if `model` has exactly one
    non-initializer input of rank 4 (a plausible single-image classifier
    input), else None. Doesn't attempt to guess NCHW vs. NHWC -- the
    `_image_classifier_config` shape declares `src_layout: "NHWC"` for the
    *raw camera-like* input Pulsar2's preprocessing produces, which is
    independent of what layout the ONNX graph itself expects internally.
    """
    initializers = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    if len(inputs) != 1:
        return None
    dims = inputs[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        return None
    return inputs[0].name


def convert_one(
    name: str,
    work_dir: str,
    *,
    profile: bool,
    run_device: bool,
    target_hardware: str,
    image: str,
) -> dict:
    import numpy as np
    import onnx

    import model_zoo
    import pulsar2_docker as pd

    res = {
        "model": name,
        "status": "error",
        "orig_success": None,
        "orig_max_cycle": None,
        "orig_fused_subgraphs": None,
        "orig_trace": None,
        "simp_success": None,
        "simp_max_cycle": None,
        "simp_fused_subgraphs": None,
        "simp_trace": None,
        "device_bit_identical": None,
        "error": None,
    }
    try:
        path = model_zoo.fetch_model(name)
        model = onnx.load(path)

        tensor_name = _single_image_input(model)
        if tensor_name is None:
            res["status"] = "skipped_not_single_image_input"
            return res

        from onnxsim import simplify

        simp, _ = simplify(model)

        model_dir = os.path.join(work_dir, "model")
        dataset_dir = os.path.join(work_dir, "dataset")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        calib_tar = os.path.join(dataset_dir, "calib.tar")
        if not os.path.exists(calib_tar):
            pd.make_synthetic_calibration_tar(calib_tar)

        orig_rel = f"model/{name}_orig.onnx"
        simp_rel = f"model/{name}_simp.onnx"
        onnx.save(model, os.path.join(work_dir, orig_rel))
        onnx.save(simp, os.path.join(work_dir, simp_rel))

        build_kwargs = dict(
            tensor_name=tensor_name,
            mean=_DEFAULT_MEAN,
            std=_DEFAULT_STD,
            calibration_dataset_rel_path="dataset/calib.tar",
            target_hardware=target_hardware,
            image=image,
            profile=profile,
        )
        orig_result = pd.build(
            work_dir, orig_rel, f"output/{name}_orig", **build_kwargs
        )
        simp_result = pd.build(
            work_dir, simp_rel, f"output/{name}_simp", **build_kwargs
        )

        res["orig_success"] = orig_result.success
        res["orig_max_cycle"] = orig_result.max_cycle
        res["orig_fused_subgraphs"] = orig_result.fused_subgraphs
        res["orig_trace"] = orig_result.trace_path
        res["simp_success"] = simp_result.success
        res["simp_max_cycle"] = simp_result.max_cycle
        res["simp_fused_subgraphs"] = simp_result.fused_subgraphs
        res["simp_trace"] = simp_result.trace_path

        if not orig_result.success:
            res["status"] = "orig_build_failed"
            res["error"] = orig_result.error
            return res
        if not simp_result.success:
            res["status"] = "simp_build_failed"
            res["error"] = simp_result.error
            return res

        if run_device and pd.axcl_available():
            rng = np.random.RandomState(0)
            # Rough default: a single-image-classifier input is normally
            # small enough for a raw uint8 NHWC buffer at a common size; if
            # the model's real input resolution differs, this comparison is
            # skipped rather than guessed at.
            input_bytes = rng.randint(0, 256, 224 * 224 * 3, dtype=np.uint8).tobytes()
            orig_out = pd.run_on_device_with_input(
                orig_result.axmodel_path, tensor_name, input_bytes
            )
            simp_out = pd.run_on_device_with_input(
                simp_result.axmodel_path, tensor_name, input_bytes
            )
            if orig_out is not None and simp_out is not None:
                res["device_bit_identical"] = orig_out == simp_out

        res["status"] = "ok"
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True, help="model short names")
    ap.add_argument(
        "--work-dir", default=None, help="Docker mount dir (default: a temp dir)"
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="pass --compiler.npu_perf --debug.dump_frontend_graph to pulsar2 build",
    )
    ap.add_argument(
        "--no-device", action="store_true", help="skip the on-device bit-exact check"
    )
    ap.add_argument("--target-hardware", default="AX650")
    ap.add_argument("--image", default="pulsar2:6.0-lite")
    ap.add_argument("--output", default="pulsar2-convert.csv")
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="don't delete an auto-created temp work dir afterward",
    )
    args = ap.parse_args()

    import pulsar2_docker as pd

    if not pd.docker_image_available(args.image):
        print(f"error: Docker image not loaded: {args.image}", file=sys.stderr)
        print(
            "load one first: docker load -i ax_pulsar2_<version>_lite.tar.gz "
            "(see pulsar2_docker.py's docstring)",
            file=sys.stderr,
        )
        return 1

    work_dir = args.work_dir
    made_temp = False
    if work_dir is None:
        import tempfile

        work_dir = tempfile.mkdtemp(prefix="pulsar2_convert_")
        made_temp = True
    os.makedirs(work_dir, exist_ok=True)

    print(f"work dir: {work_dir}")
    if args.profile:
        print("profiling enabled: trace.json will be written per successful build")

    rows = []
    for i, name in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] {name} ...", flush=True)
        r = convert_one(
            name,
            work_dir,
            profile=args.profile,
            run_device=not args.no_device,
            target_hardware=args.target_hardware,
            image=args.image,
        )
        rows.append(r)
        print(f"  status={r['status']}", flush=True)
        if r["orig_trace"]:
            print(f"  orig trace: {r['orig_trace']}", flush=True)
        if r["simp_trace"]:
            print(f"  simp trace: {r['simp_trace']}", flush=True)
        if r["device_bit_identical"] is not None:
            print(f"  device bit-identical: {r['device_bit_identical']}", flush=True)
        if r["error"]:
            print(f"  error: {str(r['error'])[:300]}", flush=True)

    fields = list(rows[0].keys()) if rows else []
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)")

    if made_temp and not args.keep_work_dir:
        # Only output/ is root-owned (written by pulsar2 build inside
        # Docker -- see pulsar2_docker.force_rmtree()'s docstring); model/,
        # dataset/, config/ are host-owned (written directly by this
        # script) and come off cleanly with a plain rmtree afterward.
        output_dir = os.path.join(work_dir, "output")
        if os.path.exists(output_dir):
            pd.force_rmtree(output_dir, work_dir, args.image)
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"work dir kept at: {work_dir}")

    failures = [
        r for r in rows if r["status"] not in ("ok", "skipped_not_single_image_input")
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
