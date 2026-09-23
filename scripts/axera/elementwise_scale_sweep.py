"""Build standalone AX650 elementwise models at a fixed shape with *controlled*
calibration ranges, for ``elementwise_scale_emit.py``'s validation.

Every earlier elementwise sweep in this project drew random calibration data,
so the realised min/max drifted with shape and seed. Here each calibration
sample is ``linspace(lo, hi)`` reshaped to the input shape, so MinMax
calibration sees exactly ``lo`` and ``hi`` and the resulting scale / zero point
are known from the case alone. Binary ops take a second range for ``z``.

This is a characterization/validation harness (it needs the
``pulsar2:7.0-lite`` Docker image); the emitter itself needs neither.

Usage::

    elementwise_scale_sweep.py build CASES.json WORK_ROOT
        # CASES: [[name, op, shape, [lo, hi]] or [name, op, shape, [lo, hi], [zlo, zhi]], ...]
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
from onnx import TensorProto, helper

_IMAGE = "pulsar2:7.0-lite"
_BINARY = ("Add", "Sub", "Mul", "Div")


def calibration_samples(shape, lo, hi, count=4):
    """``count`` identical samples spanning exactly ``[lo, hi]``."""
    n = int(np.prod(shape))
    sample = np.linspace(lo, hi, n, dtype=np.float64).astype(np.float32)
    return [sample.reshape(shape) for _ in range(count)]


def _write_tar(path, samples):
    with tarfile.open(path, "w") as tar:
        for k, arr in enumerate(samples):
            buf = io.BytesIO()
            np.save(buf, arr)
            info = tarfile.TarInfo(f"{k}.npy")
            info.size = len(buf.getvalue())
            buf.seek(0)
            tar.addfile(info, buf)


def build_one(root, name, op, shape, xr, zr=None):
    """Compile ``op`` at ``shape`` with calibration ranges ``xr`` (and ``zr``
    for a binary op) into ``<root>/<name>/out/compiled.axmodel``."""
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    inputs = ["x", "z"] if op in _BINARY else ["x"]
    graph = helper.make_graph(
        [helper.make_node(op, inputs, ["y"])],
        "g",
        [helper.make_tensor_value_info(i, TensorProto.FLOAT, shape) for i in inputs],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, os.path.join(wd, "t.onnx"))
    ranges = {"x": xr, "z": zr}
    cfg = []
    for i in inputs:
        lo, hi = ranges[i]
        _write_tar(
            os.path.join(wd, "dataset", f"{i}.tar"), calibration_samples(shape, lo, hi)
        )
        cfg.append(
            {
                "tensor_name": i,
                "calibration_dataset": f"./dataset/{i}.tar",
                "calibration_format": "Numpy",
                "calibration_size": 4,
            }
        )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": cfg,
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    with open(os.path.join(wd, "config", "step.json"), "w") as f:
        json.dump(config, f)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"ew-scale-{name}-{os.getpid()}",
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
    return result.returncode


def main(argv):
    if len(argv) != 3 or argv[0] != "build":
        print(__doc__)
        return 2
    cases = json.load(open(argv[1]))
    root = argv[2]

    def one(case):
        name, op, shape, xr = case[:4]
        zr = case[4] if len(case) > 4 else None
        return name, build_one(root, name, op, shape, xr, zr)

    with ThreadPoolExecutor(2) as pool:
        for name, rc in pool.map(one, cases):
            print(name, "ok" if rc == 0 else f"rc={rc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
