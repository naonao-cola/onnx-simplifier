"""Isolate calibration range from shape when sweeping a standalone ``Relu``'s
``teng2`` compute segment (see ``docs/axera-teng2-calibration-isolation.md``).

Every prior ``teng2`` sweep in this project (``dma_tile_sweep.py`` and its
descendants) seeded calibration with a *fixed* ``RandomState(0)`` per input,
not a fresh random seed per build -- but because the calibration array's
*shape* changes across a shape sweep, the same fixed-seed uniform draw yields
a different number of samples, and therefore a different observed min/max
(and so a different ``y_scale``/``y_zero``) at every shape. So shape and
calibration range were still confounded, just not for the reason "random
seed per build" suggests -- it was "sample count varies with shape, so the
extrema of a fixed uniform draw vary with shape" instead.

This module pins the calibration array's min and max to an *exact*, known
float32 value regardless of shape, by drawing the interior uniformly at
random (fixed seed, for determinism) and then overwriting element 0 and
element 1 with the exact bounds. That makes ``y_scale``/``y_zero`` a function
of the chosen bounds alone, not of shape, so a shape sweep at fixed bounds
and a bounds sweep at fixed shape each isolate one variable.

Usage::

    teng2_calibration_isolation.py build CASES.json WORK_ROOT
    teng2_calibration_isolation.py diff WORK_ROOT NAME_A NAME_B
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import onnx
from onnx import TensorProto, helper

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"
_NOISE = (301, 326)


def _pinned_array(shape: list[int], lo: float, hi: float, seed: int) -> np.ndarray:
    """A float32 array of ``shape`` whose exact min is ``lo`` and exact max is ``hi``."""
    rng = np.random.RandomState(seed)
    arr = rng.uniform(lo, hi, shape).astype(np.float32)
    flat = arr.reshape(-1)
    flat[0] = np.float32(lo)
    flat[1] = np.float32(hi)
    return arr


def build_one(
    root: str, name: str, shape: list[int], lo: float, hi: float, seed: int = 0
) -> int:
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, os.path.join(wd, "t.onnx"))
    n = int(np.prod(shape))
    samples = 2 if n > 2_000_000 else 4
    with tarfile.open(os.path.join(wd, "dataset", "x.tar"), "w") as tar:
        for k in range(samples):
            arr = _pinned_array(shape, lo, hi, seed * 100 + k)
            buf = io.BytesIO()
            np.save(buf, arr)
            info = tarfile.TarInfo(f"{k}.npy")
            info.size = len(buf.getvalue())
            buf.seek(0)
            tar.addfile(info, buf)
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": samples,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    json.dump(config, open(os.path.join(wd, "config", "step.json"), "w"))
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"teng2-cal-{name}-{os.getpid()}",
            "-v",
            f"{wd}:/data",
            _IMAGE,
            "pulsar2",
            "build",
            "--target_hardware",
            "AX650",
            "--input",
            "t.onnx",
            "--output_dir",
            "out",
            "--config",
            "config/step.json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


def _seg2(root: str, name: str) -> bytes:
    path = os.path.join(root, name, "out", "compiled.axmodel")
    model = onnx.load(path, load_external_data=False)
    blob = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )
    segs = mcode.segments(blob)[1]
    start, size = segs[2][0], segs[2][1]
    return blob[start : start + size]


def diff(root: str, name_a: str, name_b: str) -> None:
    a, b = _seg2(root, name_a), _seg2(root, name_b)
    print(f"{name_a}: {len(a)} bytes; {name_b}: {len(b)} bytes")
    if len(a) != len(b):
        print("lengths differ -- no positional diff")
        return
    diffs = [
        i for i in range(len(a)) if a[i] != b[i] and not (_NOISE[0] <= i < _NOISE[1])
    ]
    print(f"{len(diffs)} differing bytes (outside noise window): {diffs}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("build", "diff"):
        print(__doc__)
        return 2
    if argv[0] == "build":
        cases = json.load(open(argv[1]))
        root = argv[2]
        for name, shape, lo, hi, *rest in cases:
            seed = rest[0] if rest else 0
            rc = build_one(root, name, shape, lo, hi, seed)
            print(name, shape, lo, hi, "ok" if rc == 0 else f"rc={rc}", flush=True)
        return 0
    diff(argv[1], argv[2], argv[3])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
