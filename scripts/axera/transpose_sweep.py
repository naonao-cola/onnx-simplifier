"""Shape sweep harness for standalone AX650 ``Transpose`` MCode characterization.

Builds one-node float32 ``Transpose`` models with Pulsar2 (Docker image
``pulsar2:7.0-lite``) and summarises each compiled ``.axmodel``: MCode length,
``npu_params`` length, and the per-segment sizes from the MCode's FlatBuffers
tail (``mcode.segments``). The findings are in
``docs/axera-transpose-mcode.md``. This is a characterization tool, not an
emitter: no shape-to-MCode predictor is validated yet.

Usage::

    transpose_sweep.py build CASES.json WORK_ROOT   # [[name, shape, perm], ...]
    transpose_sweep.py table WORK_ROOT              # one summary row per build
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
from onnx import TensorProto, helper

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"


def build_one(root: str, name: str, shape: list[int], perm: list[int]) -> int:
    """Compile ``Transpose(x, perm)`` to ``<root>/<name>/out/compiled.axmodel``."""
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    out_shape = [shape[p] for p in perm]
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["x"], ["y"], perm=perm)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, os.path.join(wd, "t.onnx"))
    rng = np.random.RandomState(0)
    samples = 2 if int(np.prod(shape)) > 2_000_000 else 4
    with tarfile.open(os.path.join(wd, "dataset", "x.tar"), "w") as tar:
        for i in range(samples):
            buf = io.BytesIO()
            np.save(buf, rng.uniform(-0.9, 0.9, shape).astype(np.float32))
            info = tarfile.TarInfo(f"{i}.npy")
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
    with open(os.path.join(wd, "config", "step.json"), "w") as f:
        json.dump(config, f)
    # a unique container name per build: the wrapper's name collides in threads
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"trsweep-{name}-{os.getpid()}",
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
    return proc.returncode


def summarize(axmodel_path: str) -> dict:
    """MCode length, ``npu_params`` length, and segment sizes of one build."""
    model = onnx.load(axmodel_path, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    blob = bytes(next(i for k, i in inits.items() if k.endswith("_neu")).raw_data)
    start, segments = mcode.segments(blob)[:2]
    return {
        "mcode": len(blob),
        "params": len(inits["npu_params"].raw_data),
        "seg_start": start,
        "seg_sizes": [seg[1] for seg in segments],
    }


def main(argv: list[str]) -> int:
    expected = {"build": 3, "table": 2}
    if not argv or expected.get(argv[0]) != len(argv):
        print(__doc__)
        return 2
    if argv[0] == "build":
        cases = json.load(open(argv[1]))
        root = argv[2]

        def job(item):
            index, (name, shape, perm) = item
            time.sleep(3 * (index % 3))  # stagger container starts
            return name, build_one(root, name, shape, perm)

        with ThreadPoolExecutor(3) as pool:
            for name, code in pool.map(job, enumerate(cases)):
                print(name, "ok" if code == 0 else f"failed ({code})", flush=True)
        return 0
    root = argv[1]
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "out", "compiled.axmodel")
        if os.path.exists(path):
            print(name, json.dumps(summarize(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
