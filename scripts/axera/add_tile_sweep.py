"""Shape/calibration sweep harness for the two-input elementwise DMA tile
table (see ``add_tile_predict.py``).

Builds one-node float32 ``Add(x, z)`` models with Pulsar2 (Docker image
``pulsar2:7.0-lite``) and checks ``npu_params`` against
``add_tile_predict.predict_params``'s prediction, pulling the calibration
scales out of Pulsar2's own ``quant_axmodel.json`` sidecar. This is a
characterization tool: it does not emit models.

Usage::

    add_tile_sweep.py build CASES.json WORK_ROOT
        # CASES.json: [[name, shape, {"x": [lo, hi], "z": [lo, hi]}], ...]
    add_tile_sweep.py check WORK_ROOT
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

from add_tile_predict import predict_params  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"


def build_one(root: str, name: str, shape: list[int], ranges: dict) -> int:
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    node = helper.make_node("Add", ["x", "z"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("z", TensorProto.FLOAT, shape),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, os.path.join(wd, "t.onnx"))
    n = int(np.prod(shape))
    samples = 2 if n > 2_000_000 else 4
    cfg_inputs = []
    seeds = {"x": 42, "z": 99}
    for name_ in ("x", "z"):
        lo, hi = ranges.get(name_, (-0.9, 0.9))
        rng = np.random.RandomState(seeds[name_])
        with tarfile.open(os.path.join(wd, "dataset", f"{name_}.tar"), "w") as tar:
            for k in range(samples):
                buf = io.BytesIO()
                np.save(buf, rng.uniform(lo, hi, shape).astype(np.float32))
                info = tarfile.TarInfo(f"{k}.npy")
                info.size = len(buf.getvalue())
                buf.seek(0)
                tar.addfile(info, buf)
        cfg_inputs.append(
            {
                "tensor_name": name_,
                "calibration_dataset": f"./dataset/{name_}.tar",
                "calibration_format": "Numpy",
                "calibration_size": samples,
            }
        )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": cfg_inputs,
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
            f"add-tile-{name}-{os.getpid()}",
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


def _npu_params(path: str) -> bytes:
    model = onnx.load(path, load_external_data=False)
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


def _scales(quant_json_path: str) -> dict:
    d = json.load(open(quant_json_path))
    tensor_configs = d["tensor_configs"]["y"]
    return {
        k: d["values"][str(tensor_configs[k]["hash"])]["scale"][0]
        for k in ("x", "z", "y")
    }


def check(root: str) -> int:
    mismatches = 0
    for name in sorted(os.listdir(root)):
        out_dir = os.path.join(root, name, "out")
        path = os.path.join(out_dir, "compiled.axmodel")
        if not os.path.exists(path):
            continue
        model = onnx.load(path, load_external_data=False)
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        real = _npu_params(path)
        scales = _scales(os.path.join(out_dir, "quant", "quant_axmodel.json"))
        try:
            pred = predict_params(shape, scales["x"], scales["z"], scales["y"])
        except ValueError as exc:
            print(f"{name:28s} {shape} out of scope: {exc}")
            continue
        ok = pred == real
        mismatches += not ok
        print(f"{name:28s} {shape} {'exact' if ok else 'MISMATCH'}")
    return mismatches


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("build", "check"):
        print(__doc__)
        return 2
    if argv[0] == "build":
        cases = json.load(open(argv[1]))
        root = argv[2]
        with ThreadPoolExecutor(3) as pool:
            for (name, shape, ranges), rc in zip(
                cases,
                pool.map(
                    lambda c: build_one(
                        root, c[0], c[1], {k: tuple(v) for k, v in c[2].items()}
                    ),
                    cases,
                ),
            ):
                print(name, shape, "ok" if rc == 0 else f"rc={rc}")
        return 0
    return 1 if check(argv[1]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
