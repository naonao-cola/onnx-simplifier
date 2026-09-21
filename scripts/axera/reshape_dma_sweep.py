"""Sweep harness for the non-fused AX650 ``Reshape`` DMA program.

Builds float32 ``Reshape -> Relu`` models (and the plain ``Relu`` at the output
shape, whose program is what a *fused* Reshape would equal) with Pulsar2 7.0-lite
in Docker, and summarises the compiled ``.axmodel`` files. Standalone ``Reshape``
does not compile, so ``Relu`` is the neighbour; see ``docs/axera-reshape.md``.
Findings are in ``docs/axera-reshape-dma.md``. Characterization tool, not an
emitter.

Usage::

    reshape_dma_sweep.py build CASES.json WORK_ROOT   # [[in_shape, out_shape], ...]
    reshape_dma_sweep.py table WORK_ROOT              # one summary row per build

Each case builds ``rs_<in>_to_<out>`` (``Reshape -> Relu``) and ``relu_<out>``.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"


def dims(shape) -> str:
    return "x".join(str(d) for d in shape)


def rs_name(shape_in, shape_out) -> str:
    return f"rs_{dims(shape_in)}_to_{dims(shape_out)}"


def relu_name(shape) -> str:
    return f"relu_{dims(shape)}"


def _model(shape_in, shape_out):
    """``Reshape(shape_in -> shape_out) -> Relu``, or plain ``Relu`` if no out."""
    if shape_out is None:
        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_in)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_in)],
        )
    else:
        target = onnx.numpy_helper.from_array(
            np.array(shape_out, dtype=np.int64), "target"
        )
        graph = helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "target"], ["t"]),
                helper.make_node("Relu", ["t"], ["y"]),
            ],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_in)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_out)],
            [target],
        )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def build_one(root: str, name: str, shape_in, shape_out) -> int:
    """Compile to ``<root>/<name>/out/compiled.axmodel``; skip if it exists."""
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    onnx.save(_model(shape_in, shape_out), os.path.join(wd, "t.onnx"))
    rng = np.random.RandomState(0)
    samples = 2 if int(np.prod(shape_in)) > 2_000_000 else 4
    with tarfile.open(os.path.join(wd, "dataset", "x.tar"), "w") as tar:
        for i in range(samples):
            buf = io.BytesIO()
            np.save(buf, rng.uniform(-0.9, 0.9, shape_in).astype(np.float32))
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
    out = os.path.join(wd, "out")
    if os.path.exists(out):
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{wd}:/data",
                _IMAGE,
                "rm",
                "-rf",
                "/data/out",
            ],
            check=True,
        )
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"rsdma-{name}-{os.getpid()}"[:120],
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
    tail = (proc.stdout + proc.stderr)[-200:].replace("\n", " ")
    print(name, "ok" if proc.returncode == 0 else f"FAIL {tail}", flush=True)
    return proc.returncode


def load(root: str, name: str):
    """Return ``(mcode_bytes, npu_params_bytes)`` of a finished build."""
    path = os.path.join(root, name, "out", "compiled.axmodel")
    model = onnx.load(path, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    blob = next(bytes(v.raw_data) for k, v in inits.items() if k.endswith("_neu"))
    return blob, bytes(inits["npu_params"].raw_data)


def build_cases(root: str, cases) -> None:
    jobs = {}
    for shape_in, shape_out in cases:
        jobs[rs_name(shape_in, shape_out)] = (shape_in, shape_out)
        jobs[relu_name(shape_out)] = (shape_out, None)
    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(build_one, root, n, *a) for n, a in jobs.items()]
        for f in futures:
            f.result()


def table(root: str) -> None:
    for name in sorted(os.listdir(root)):
        if not os.path.exists(os.path.join(root, name, "out", "compiled.axmodel")):
            continue
        blob, params = load(root, name)
        segs = mcode.segments(blob)[1]
        print(name, len(blob), len(params), [s[1] for s in segs])


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "build":
        cases = [(tuple(a), tuple(b)) for a, b in json.load(open(argv[1]))]
        build_cases(argv[2], cases)
        return 0
    if len(argv) == 2 and argv[0] == "table":
        table(argv[1])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
