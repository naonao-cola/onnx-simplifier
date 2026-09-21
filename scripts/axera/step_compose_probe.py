#!/usr/bin/env python3
"""Composition probe: is a chain's MCode made of its parts' MCode?

Builds a ladder of small real-shaped graphs with Pulsar2 (Docker image
``pulsar2:7.0-lite``, ``--compiler.npu_perf`` so the per-engine task lists are
kept) and compares each chain with its standalone parts, engine segment by
engine segment (see ``step_attribution.SEGMENT_ENGINES``). Findings are in
``docs/axera-step-attribution.md``. Characterization only.

The ladder mirrors a ResNet18 layer4 convolution (3x3, 512 -> 512, 7x7,
batch 16) and the training step's classifier head (``[16,512] x [512,1000]`` with a
bias add):

    conv1, relu1, conv_relu, conv_relu_conv, mm, add1, mm_add

Usage::

    step_compose_probe.py build WORK_ROOT [NAME ...]   # 2 concurrent builds
    step_compose_probe.py compare WORK_ROOT
"""

from __future__ import annotations

import collections
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402
from step_attribution import SEGMENT_ENGINES  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"
_N, _C, _H = 16, 512, 7
_X = [_N, _C, _H, _H]
WINDOW = 12  # bytes; windows with fewer than 4 distinct values are ignored


def _vi(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _conv_weight(seed: int):
    rng = np.random.RandomState(seed)
    return numpy_helper.from_array(
        (rng.randn(_C, _C, 3, 3) * 0.05).astype(np.float32), f"w{seed}"
    )


def _conv(src: str, dst: str, seed: int):
    return helper.make_node(
        "Conv", [src, f"w{seed}"], [dst], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )


def _model(nodes, inputs, outputs, inits=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def cases() -> dict[str, tuple[onnx.ModelProto, dict[str, list[int]]]]:
    head_in = {"a": [16, 512], "b": [512, 1000]}
    return {
        "conv1": (
            _model(
                [_conv("x", "y", 1)], [_vi("x", _X)], [_vi("y", _X)], [_conv_weight(1)]
            ),
            {"x": _X},
        ),
        "relu1": (
            _model(
                [helper.make_node("Relu", ["x"], ["y"])], [_vi("x", _X)], [_vi("y", _X)]
            ),
            {"x": _X},
        ),
        "conv_relu": (
            _model(
                [_conv("x", "t", 1), helper.make_node("Relu", ["t"], ["y"])],
                [_vi("x", _X)],
                [_vi("y", _X)],
                [_conv_weight(1)],
            ),
            {"x": _X},
        ),
        "conv_relu_conv": (
            _model(
                [
                    _conv("x", "t", 1),
                    helper.make_node("Relu", ["t"], ["u"]),
                    _conv("u", "y", 2),
                ],
                [_vi("x", _X)],
                [_vi("y", _X)],
                [_conv_weight(1), _conv_weight(2)],
            ),
            {"x": _X},
        ),
        "mm": (
            _model(
                [helper.make_node("MatMul", ["a", "b"], ["y"])],
                [_vi("a", [16, 512]), _vi("b", [512, 1000])],
                [_vi("y", [16, 1000])],
            ),
            head_in,
        ),
        "add1": (
            _model(
                [helper.make_node("Add", ["y", "c"], ["z"])],
                [_vi("y", [16, 1000]), _vi("c", [1000])],
                [_vi("z", [16, 1000])],
            ),
            {"y": [16, 1000], "c": [1000]},
        ),
        "mm_add": (
            _model(
                [
                    helper.make_node("MatMul", ["a", "b"], ["t"]),
                    helper.make_node("Add", ["t", "c"], ["z"]),
                ],
                [_vi("a", [16, 512]), _vi("b", [512, 1000]), _vi("c", [1000])],
                [_vi("z", [16, 1000])],
            ),
            {**head_in, "c": [1000]},
        ),
    }


def build_one(root: str, name: str) -> int:
    model, inputs = cases()[name]
    wd = os.path.join(root, name)
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    onnx.save(model, os.path.join(wd, "t.onnx"))
    configs = []
    for tensor, shape in inputs.items():
        with tarfile.open(os.path.join(wd, "dataset", f"{tensor}.tar"), "w") as tar:
            for i in range(4):
                buf = io.BytesIO()
                data = np.random.RandomState(i).uniform(-0.9, 0.9, shape)
                np.save(buf, data.astype(np.float32))
                info = tarfile.TarInfo(f"{i}.npy")
                info.size = len(buf.getvalue())
                buf.seek(0)
                tar.addfile(info, buf)
        configs.append(
            {
                "tensor_name": tensor,
                "calibration_dataset": f"./dataset/{tensor}.tar",
                "calibration_format": "Numpy",
                "calibration_size": 4,
            }
        )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": configs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    with open(os.path.join(wd, "config", "step.json"), "w") as f:
        json.dump(config, f)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    cmd = [
        "docker", "run", "--rm", "--name", f"compose-probe-{name}-{os.getpid()}",
        "-v", f"{wd}:/data", _IMAGE, "pulsar2", "build", "--target_hardware", "AX650",
        "--input", "t.onnx", "--output_dir", "out", "--config", "config/step.json",
        "--compiler.npu_perf", "--debug.dump_frontend_graph",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        print(name, "FAILED", (proc.stdout + proc.stderr)[-300:], flush=True)
    return proc.returncode


def load_build(root: str, name: str) -> dict:
    base = os.path.join(root, name, "out")
    model = onnx.load(os.path.join(base, "compiled.axmodel"), load_external_data=False)
    blob = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )
    _, segs = mcode.segments(blob)
    trace = os.path.join(
        base, "compiler", "debug", "subgraph_npu_0", "b1", "trace.json"
    )
    with open(trace) as f:
        events = json.load(f)["traceEvents"]
    return {
        "len": len(blob),
        "segs": [blob[p : p + n] for p, n, _ in segs],
        "tasks": collections.Counter(e["tid"] for e in events),
    }


def windows(data: bytes) -> set[bytes]:
    out = set()
    for i in range(len(data) - WINDOW + 1):
        w = data[i : i + WINDOW]
        if len(set(w)) >= 4:
            out.add(w)
    return out


def contained(segment: bytes, references: list[bytes]) -> tuple[float, int]:
    """Fraction of ``segment``'s informative windows found in any reference."""
    ref: set[bytes] = set()
    for r in references:
        ref |= windows(r)
    seen = hit = 0
    for i in range(len(segment) - WINDOW + 1):
        w = segment[i : i + WINDOW]
        if len(set(w)) < 4:
            continue
        seen += 1
        hit += w in ref
    return (hit / seen if seen else float("nan")), seen


_TRACKS = {
    "conv": ("conv0", "conv1"),
    "teng2": ("teng2",),
    "cv3": ("cv3",),
    "sdma4": ("sdma4",),
}


def compare(root: str, chain: str, parts: list[str]) -> None:
    c = load_build(root, chain)
    ps = [load_build(root, p) for p in parts]
    print(f"\n== {chain} ({c['len']} B) vs parts {parts} ({[p['len'] for p in ps]} B)")
    print(
        f"  {'engine':10s} {'chain':>7s} {'sum parts':>10s} {'ratio':>6s}  chain content found in parts"
    )
    for i, engine in enumerate(SEGMENT_ENGINES):
        chain_b = len(c["segs"][i])
        sum_b = sum(len(p["segs"][i]) for p in ps)
        frac, seen = contained(c["segs"][i], [p["segs"][i] for p in ps])
        print(
            f"  {i}:{engine:8s} {chain_b:7d} {sum_b:10d} {chain_b / sum_b:6.2f}  {100 * frac:5.1f}% of {seen} windows"
        )
    for part, p in zip(parts, ps):
        found = [
            contained(p["segs"][i], [c["segs"][i]])[0]
            for i in range(len(SEGMENT_ENGINES))
        ]
        print(
            f"  standalone {part:8s} found in {chain}: "
            + ", ".join(
                f"{SEGMENT_ENGINES[i]} {100 * f:.0f}%" for i, f in enumerate(found)
            )
        )


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("build", "compare"):
        print(__doc__)
        return 2
    root = argv[1]
    if argv[0] == "build":
        names = argv[2:] or list(cases())
        with ThreadPoolExecutor(2) as pool:
            for name, rc in zip(names, pool.map(lambda n: build_one(root, n), names)):
                print(name, "ok" if rc == 0 else f"rc={rc}", flush=True)
        return 0
    compare(root, "conv_relu", ["conv1", "relu1"])
    compare(root, "conv_relu_conv", ["conv1", "conv1", "relu1"])
    compare(root, "mm_add", ["mm", "add1"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
