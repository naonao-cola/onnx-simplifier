#!/usr/bin/env python3
"""Time compiled Edge TPU models on-device via LiteRT (needs hardware).

Reads the ``peak_manifest.json`` written by ``peak_benchmark.py`` (model file
+ exact GMACs), runs each model in a tight loop through LiteRT with the
``libedgetpu`` delegate, and reports median/min latency with the achieved
TOPS (2 ops per MAC) against the 4 TOPS INT8 spec peak.

Needs, once, on the machine with the device (NOT needed to build models)::

    sudo apt-get install libedgetpu1-std    # or libedgetpu1-max for max clocks
    pip install ai-edge-litert

plus USB access for the current user (without it ``load_delegate`` fails --
the usual cause; ``strace`` shows ``EACCES`` on ``/dev/bus/usb/...``)::

    echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a6e", MODE="0666", GROUP="plugdev"' \\
        | sudo tee /etc/udev/rules.d/99-edgetpu.rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
    sudo usermod -aG plugdev $USER   # then log out/in and replug the device

Example::

    python peak_benchmark.py --out-dir peak_models   # no hardware needed
    python benchmark_device.py --dir peak_models --runs 500
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np


def _load_litert():
    try:
        from ai_edge_litert import interpreter as litert
    except ImportError as exc:
        raise SystemExit("LiteRT is required here: pip install ai-edge-litert") from exc
    return litert


def time_model(
    litert, model_path: str, lib: str, warmup: int, runs: int
) -> tuple[float, float, float]:
    """Median/min/std invoke latency in ms for one compiled model."""
    delegate = litert.load_delegate(lib)
    interp = litert.Interpreter(
        model_path=model_path, experimental_delegates=[delegate]
    )
    interp.allocate_tensors()
    (detail,) = interp.get_input_details()
    feed = np.random.default_rng(0).integers(
        0, 256, size=list(detail["shape"]), dtype=np.uint8
    )
    interp.set_tensor(detail["index"], feed)
    for _ in range(warmup):
        interp.invoke()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), min(times), statistics.stdev(times)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="peak_models")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lib", default="libedgetpu.so.1")
    args = ap.parse_args(argv)
    litert = _load_litert()

    manifest_path = os.path.join(args.dir, "peak_manifest.json")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read manifest {manifest_path}: {exc}") from exc

    print(
        f"{'model':16s} {'GMAC':>7s} {'median_ms':>9s} {'min_ms':>8s} "
        f"{'TOPS_med':>8s} {'TOPS_min':>8s} {'%peak':>6s}"
    )
    failed = 0
    for tag, entry in manifest.items():
        path = os.path.join(args.dir, entry["file"])
        if not os.path.isfile(path):
            print(f"{tag:16s} MISSING {path}")
            failed += 1
            continue
        try:
            median_ms, min_ms, _ = time_model(
                litert, path, args.lib, args.warmup, args.runs
            )
        except Exception as exc:  # noqa: BLE001 -- report per model, keep going
            print(f"{tag:16s} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            failed += 1
            continue
        gops = entry["gmac"] * 2.0
        tops_med = gops / (median_ms / 1e3) / 1e3
        tops_min = gops / (min_ms / 1e3) / 1e3
        print(
            f"{tag:16s} {entry['gmac']:7.3f} {median_ms:9.3f} {min_ms:8.3f} "
            f"{tops_med:8.2f} {tops_min:8.2f} {100 * tops_min / 4.0:5.1f}%"
        )
    print(
        "peak reference: 4 TOPS INT8 (2 TOPS/W); literature best-CNN "
        "~40% of peak (~1.6 TOPS)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
