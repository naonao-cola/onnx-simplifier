"""Item 2 of `docs/axera-conv-gather-loose-ends.md`: does `docs/axera-gather-
aggregate-real.md` (PR #1761)'s "calibrate the Gather's input across its full
valid range" safety recipe hold for a real `Add`-sum aggregator, not just the
`MatMul` PR #1761 checked?

No Gather in the real training step (`/home/takecheeze/npu-scratch/t6-r18fold/
step.onnx`) feeds an `Add` or `ReduceSum` within three hops (checked directly
by walking the graph), so this builds a synthetic chain mirroring PR #1761's
real shape and methodology as closely as possible: `Gather(x[16,1,512,49],
idx[448], axis=3) -> Mul(mask) -> Reshape([16,1,512,4,112]) -> Slice*4 ->
Squeeze*4 -> Add -> Add -> Add`, a genuine 4-way tree-sum of gathered/masked
groups, with the same structured two-band calibration range PR #1761 used.

Result (see the doc's "Item 2" section for the full device numbers): the
failure mode reproduces (max err 2.19 retargeted-narrow vs. 0.037 correct)
and wide-range calibration on the Gather's input fixes it (0.039, matching
native device noise) -- confirming the recipe is a property of aggregation
generally, not specific to `MatMul`.

Usage::

    gather_aggregate_addsum_check.py build [NAME ...]   # writes ONNX + calib data
    gather_aggregate_addsum_check.py check              # static npu_params check
        (from already-built outputs under ROOT; no Docker/device)
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tarfile

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = "/home/takecheeze/npu-scratch/t_gather_aggregate_addsum"

X_SHAPE = [16, 1, 512, 49]
N_IDX = 448
GROUP = 112

IDX_REF = (np.arange(N_IDX) % 39).tolist()  # every index in the "small" [-0.3,0.3] band
IDX_ADV = (
    39 + np.arange(N_IDX) % 10
).tolist()  # every index in the "large" [-0.9,0.9] band


def make_model(idx: list[int]) -> onnx.ModelProto:
    idx_t = helper.make_tensor("idx", TensorProto.INT64, [N_IDX], idx)
    mask = (np.arange(N_IDX) % 7 != 0).astype(np.float32).reshape(1, 1, 1, N_IDX)
    mask_t = helper.make_tensor(
        "mask", TensorProto.FLOAT, list(mask.shape), mask.flatten().tolist()
    )
    shape_t = helper.make_tensor(
        "shape5", TensorProto.INT64, [5], [16, 1, 512, 4, GROUP]
    )
    nodes = [
        helper.make_node("Gather", ["x", "idx"], ["g"], axis=3),
        helper.make_node("Mul", ["g", "mask"], ["gm"]),
        helper.make_node("Reshape", ["gm", "shape5"], ["r"]),
        helper.make_node("Slice", ["r", "s0", "e0", "ax3"], ["t0"]),
        helper.make_node("Slice", ["r", "s1", "e1", "ax3"], ["t1"]),
        helper.make_node("Slice", ["r", "s2", "e2", "ax3"], ["t2"]),
        helper.make_node("Slice", ["r", "s3", "e3", "ax3"], ["t3"]),
        helper.make_node("Squeeze", ["t0", "ax3"], ["q0"]),
        helper.make_node("Squeeze", ["t1", "ax3"], ["q1"]),
        helper.make_node("Squeeze", ["t2", "ax3"], ["q2"]),
        helper.make_node("Squeeze", ["t3", "ax3"], ["q3"]),
        helper.make_node("Add", ["q0", "q1"], ["a01"]),
        helper.make_node("Add", ["q2", "q3"], ["a23"]),
        helper.make_node("Add", ["a01", "a23"], ["y"]),
    ]

    def c(name: str, vals: list[int]) -> onnx.TensorProto:
        return helper.make_tensor(name, TensorProto.INT64, [len(vals)], vals)

    inits = [
        idx_t,
        mask_t,
        shape_t,
        c("s0", [0]),
        c("e0", [1]),
        c("s1", [1]),
        c("e1", [2]),
        c("s2", [2]),
        c("e2", [3]),
        c("s3", [3]),
        c("e3", [4]),
        c("ax3", [3]),
    ]
    g = helper.make_graph(
        nodes,
        "gather_addsum",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, X_SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [16, 1, 512, GROUP])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def groundtruth(x: np.ndarray, idx: list[int]) -> np.ndarray:
    """The exact reference computation `y = sum of 4 masked/gathered groups`."""
    g = x[..., idx]
    mask = (np.arange(N_IDX) % 7 != 0).astype(np.float32).reshape(1, 1, 1, N_IDX)
    r = (g * mask).reshape(16, 1, 512, 4, GROUP)
    return r[..., 0, :] + r[..., 1, :] + r[..., 2, :] + r[..., 3, :]


def calib_tar(path: str, structured: bool) -> None:
    rng = np.random.RandomState(0)
    with tarfile.open(path, "w") as t:
        for i in range(4):
            a = np.zeros(X_SHAPE, dtype=np.float32)
            if structured:
                a[..., :39] = rng.uniform(-0.3, 0.3, a[..., :39].shape)
                a[..., 39:] = rng.uniform(-0.9, 0.9, a[..., 39:].shape)
            else:
                a[...] = rng.uniform(-0.9, 0.9, a.shape)
            buf = io.BytesIO()
            np.save(buf, a)
            buf.seek(0)
            ti = tarfile.TarInfo(f"{i}.npy")
            ti.size = len(buf.getvalue())
            t.addfile(ti, buf)


CASES = {
    "a_reference_narrow": (IDX_REF, True),
    "b_native_adversarial": (IDX_ADV, True),
    "c_wide_reference": (IDX_REF, False),
}


def build(name: str) -> str:
    idx, structured = CASES[name]
    wd = os.path.join(ROOT, name)
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    onnx.save(make_model(idx), os.path.join(wd, "t.onnx"))
    calib_tar(os.path.join(wd, "dataset", "x.tar"), structured)
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    json.dump(cfg, open(os.path.join(wd, "config", "step.json"), "w"))
    return wd


def npu_params(compiled_axmodel_path: str) -> bytes:
    model = onnx.load(compiled_axmodel_path, load_external_data=False)
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


def check_index_layout(out_dir_name: str = "out2") -> None:
    """Confirms npu_params' leading N words are exactly that build's own index
    array, for the a/b builds under ROOT (no Docker/device required beyond
    what already built them)."""
    a = npu_params(
        os.path.join(ROOT, "a_reference_narrow", out_dir_name, "compiled.axmodel")
    )
    b = npu_params(
        os.path.join(ROOT, "b_native_adversarial", out_dir_name, "compiled.axmodel")
    )
    wa = struct.unpack(f"<{N_IDX}I", a[: N_IDX * 4])
    wb = struct.unpack(f"<{N_IDX}I", b[: N_IDX * 4])
    assert list(wa) == IDX_REF, "a's leading words do not match its own indices"
    assert list(wb) == IDX_ADV, "b's leading words do not match its own indices"
    print("index layout confirmed for both builds")


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("build", "check"):
        print(__doc__)
        return 2
    if argv[0] == "build":
        for name in argv[1:] or list(CASES):
            print(name, "->", build(name))
        return 0
    check_index_layout()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
