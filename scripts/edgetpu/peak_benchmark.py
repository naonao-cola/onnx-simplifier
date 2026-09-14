#!/usr/bin/env python3
"""Edge TPU peak-performance benchmark, off-device part.

For each model in ``models.py``: full-integer quantize (uint8 I/O,
``io_layout="nhwc"``) via onnxsim's public API, compile with
``edgetpu_compiler``, and print a roofline table (exact MACs, mapping,
on-chip/off-chip memory, predicted link- vs compute-bound times). Fully
mapped models are written out with a ``peak_manifest.json`` for
``benchmark_device.py`` to time on-device.

Needs TensorFlow (via onnxsim) and, for compilation, the ``edgetpu_compiler``
binary (``--skip-compile`` otherwise). No Edge TPU hardware is touched here.

Example::

    python peak_benchmark.py --out-dir peak_models
    python peak_benchmark.py --models pointwise-48 cliff-64x32 --samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_EDGETPU_DIR = os.path.dirname(os.path.abspath(__file__))
if _EDGETPU_DIR not in sys.path:
    sys.path.insert(0, _EDGETPU_DIR)

import models  # noqa: E402

# Link/compute assumptions for the roofline predictions (see README.md).
_USB2_BW = 40e6  # effective USB 2.0 bulk throughput, bytes/s
_USB3_BW = 350e6  # effective USB 3.0 bulk throughput, bytes/s
_PEAK_TOPS = 4e12  # Edge TPU INT8 spec peak, ops/s
_SUSTAINED_TOPS = 1.6e12  # literature best-CNN ceiling (~40% of peak), ops/s


def estimate(gmac: float, io_bytes: int) -> dict:
    """Roofline prediction for one model: compute vs link-bound times (ms) and
    the implied sustained TOPS on each USB generation. Pure arithmetic (2 ops
    per MAC), so unit-testable without any dependency."""
    ops = gmac * 2e9
    t_compute = ops / _SUSTAINED_TOPS * 1e3
    t_usb2 = io_bytes / _USB2_BW * 1e3
    t_usb3 = io_bytes / _USB3_BW * 1e3
    pred_usb2 = max(t_compute, t_usb2)
    pred_usb3 = max(t_compute, t_usb3)
    return {
        "t_compute_ms": t_compute,
        "t_usb2_ms": t_usb2,
        "t_usb3_ms": t_usb3,
        "tops_usb2": ops / (pred_usb2 / 1e3) / 1e12,
        "tops_usb3": ops / (pred_usb3 / 1e3) / 1e12,
        "usb3_regime": "compute" if t_compute > t_usb3 else "link",
    }


def _compiler_memories(log_text: str) -> tuple[str, str]:
    onchip = offchip = ""
    for line in log_text.splitlines():
        s = line.strip()
        if s.startswith("On-chip memory used"):
            onchip = s.split(":", 1)[1].strip()
        elif s.startswith("Off-chip memory used"):
            offchip = s.split(":", 1)[1].strip()
    return onchip, offchip


def run_suite(
    out_dir: str,
    *,
    names: list[str] | None = None,
    layout: str = "nhwc",
    samples: int = 5,
    io_dtype: str = "uint8",
    skip_compile: bool = False,
    compiler: str | None = None,
) -> tuple[list[dict], dict]:
    """Quantize + compile the suite; returns (table rows, manifest)."""
    import onnxsim

    rows: list[dict] = []
    manifest: dict = {}
    wanted = models.all_models()
    for name, bm in wanted.items():
        if names and name not in names:
            continue
        row: dict = {
            "model": name,
            "gmac": bm.gmac,
            "params_mb": bm.param_bytes / 1e6,
            "mapped": "",
            "subgraphs": "",
            "on_chip": "",
            "off_chip": "",
        }
        if not skip_compile:
            quantized = onnxsim.quantize_for_edgetpu(
                bm.model,
                num_calibration_samples=samples,
                inference_io_dtype=io_dtype,
                io_layout=layout,
            )
            out_path = os.path.join(out_dir, f"{name}_edgetpu.tflite")
            result = onnxsim.compile_for_edgetpu(
                quantized, output_path=out_path, compiler=compiler
            )
            row["mapped"] = (
                f"{result.mapped_ops}/{result.total_ops}" if result.success else "FAIL"
            )
            row["subgraphs"] = result.num_subgraphs
            row["on_chip"], row["off_chip"] = _compiler_memories(result.log_text)
            if result.fully_mapped:
                manifest[name] = {
                    "file": os.path.basename(out_path),
                    "gmac": bm.gmac,
                    "input_shape": bm.input_shape,
                    "io_layout": layout,
                }
        # Steady state: params stay cached on-chip, only I/O transfers.
        row.update(estimate(bm.gmac, bm.io_bytes))
        rows.append(row)
    return rows, manifest


def print_table(rows: list[dict]) -> None:
    print(
        f"{'model':14s} {'GMAC':>7s} {'params':>7s} {'map':>7s} "
        f"{'subg':>4s} {'on-chip':>10s} {'off-chip':>9s} | "
        f"{'USB2':>5s} {'USB3':>5s}  USB3 regime"
    )
    for r in rows:
        print(
            f"{r['model']:14s} {r['gmac']:7.3f} {r['params_mb']:6.2f}M "
            f"{r['mapped']:>7s} {str(r['subgraphs']):>4s} {r['on_chip']:>10s} "
            f"{r['off_chip']:>9s} | {r['tops_usb2']:5.2f} {r['tops_usb3']:5.2f}  "
            f"{r['usb3_regime']}-bound"
        )
    print(
        "\nTOPS columns assume the literature best-CNN ceiling (~40% of the "
        "4 TOPS peak) for compute and 40/350 MB/s effective USB bulk "
        "throughput; see README.md."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="peak_models")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--layout", default="nhwc", choices=["nchw", "nhwc"])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--io-dtype", default="uint8", choices=["uint8", "int8"])
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument(
        "--compiler",
        default=None,
        help="Explicit edgetpu_compiler path ($EDGETPU_COMPILER or PATH otherwise).",
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rows, manifest = run_suite(
        args.out_dir,
        names=args.models,
        layout=args.layout,
        samples=args.samples,
        io_dtype=args.io_dtype,
        skip_compile=args.skip_compile,
        compiler=args.compiler,
    )
    print_table(rows)
    if manifest:
        manifest_path = os.path.join(args.out_dir, "peak_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nmanifest: {len(manifest)} compiled models -> {manifest_path}")
    elif not args.skip_compile:
        print("\nno fully-mapped models; manifest not written")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
