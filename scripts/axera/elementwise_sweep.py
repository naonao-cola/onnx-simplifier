"""Pulsar2 sweep harness for standalone AX650 elementwise/activation ops.

Builds one-op float32 models (``relu``, ``sqrt``, ``add``, ``mul``, ``sub``,
``div``, ``gtcast`` = Greater(x, 0) -> Cast, ``rsum`` = ReduceSum over the last
axis) with the Docker image ``pulsar2:7.0-lite`` and MinMax Numpy calibration,
at up to three concurrent builds. Names starting with ``p`` pin every sample's
minimum and maximum to the calibration range, so scales are shape independent.
The findings are in ``docs/axera-elementwise-coverage.md``.

Usage::

    elementwise_sweep.py CASES.json
    # CASES.json: [[name, op, shape, seed, range_lo, range_hi], ...]  (last three optional)
"""

import io
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.environ.get("ELEMENTWISE_SWEEP_ROOT", "/tmp/t_elementwise")


def graph(op, shape):
    f = TensorProto.FLOAT

    def vi(n):
        return helper.make_tensor_value_info(n, f, shape)

    if op in ("relu", "sqrt"):
        nodes = [helper.make_node({"relu": "Relu", "sqrt": "Sqrt"}[op], ["x"], ["y"])]
        return nodes, [vi("x")], [vi("y")], [], ["x"]
    if op in ("add", "mul", "sub", "div"):
        nodes = [helper.make_node(op.capitalize(), ["x1", "x2"], ["y"])]
        return nodes, [vi("x1"), vi("x2")], [vi("y")], [], ["x1", "x2"]
    if op == "gtcast":
        z = numpy_helper.from_array(np.zeros([1], np.float32), "zero")
        nodes = [
            helper.make_node("Greater", ["x", "zero"], ["g"]),
            helper.make_node("Cast", ["g"], ["y"], to=f),
        ]
        return nodes, [vi("x")], [vi("y")], [z], ["x"]
    if op == "rsum":
        ax = numpy_helper.from_array(np.array([len(shape) - 1], np.int64), "axes")
        nodes = [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)]
        oshape = list(shape[:-1]) + [1]
        out = helper.make_tensor_value_info("y", f, oshape)
        return nodes, [vi("x")], [out], [ax], ["x"]
    raise ValueError(op)


def build_one(name, op, shape, seed=0, rng_lo=None, rng_hi=None):
    wd = os.path.join(ROOT, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(wd + "/dataset", exist_ok=True)
    os.makedirs(wd + "/config", exist_ok=True)
    nodes, ins, outs, inits, names = graph(op, shape)
    m = helper.make_model(
        helper.make_graph(nodes, "g", ins, outs, inits),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    m.ir_version = 8
    onnx.save(m, wd + "/t.onnx")
    rng = np.random.RandomState(seed)
    n = int(np.prod(shape))
    samples = 2 if n > 2_000_000 else 4
    cfg = []
    for nm in names:
        positive = op == "sqrt" or (op == "div" and nm == "x2")
        lo, hi = (0.1, 0.9) if positive else (-0.9, 0.9)
        if rng_lo is not None:
            lo, hi = rng_lo, rng_hi
        with tarfile.open(f"{wd}/dataset/{nm}.tar", "w") as t:
            for i in range(samples):
                b = io.BytesIO()
                arr = rng.uniform(lo, hi, shape).astype(np.float32)
                if name.startswith("p"):
                    # pin every sample's min/max so scales match across shapes
                    flat = arr.reshape(-1)
                    flat[0], flat[1] = lo, hi
                np.save(b, arr)
                b.seek(0)
                ti = tarfile.TarInfo(f"{i}.npy")
                ti.size = len(b.getvalue())
                t.addfile(ti, b)
        cfg.append(
            {
                "tensor_name": nm,
                "calibration_dataset": f"./dataset/{nm}.tar",
                "calibration_format": "Numpy",
                "calibration_size": samples,
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
    json.dump(config, open(wd + "/config/step.json", "w"))
    cmd = [
        "docker", "run", "--rm", "--name", f"ew-{name}-{os.getpid()}",
        "-v", f"{wd}:/data", "pulsar2:7.0-lite", "pulsar2", "build",
        "--target_hardware", "AX650", "--input", "t.onnx",
        "--output_dir", "out", "--config", "config/step.json",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        open(wd + "/build_error.txt", "w").write((r.stdout + r.stderr)[-2500:])
    print(name, "ok" if r.returncode == 0 else "FAIL", flush=True)
    return r.returncode


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda c: build_one(*c), cases))
