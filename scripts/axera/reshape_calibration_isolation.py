"""Does holding calibration data fixed clean up the non-fused Reshape DMA
program sweep in ``docs/axera-reshape-dma.md``?

That sweep (``reshape_dma_sweep.py``) drew calibration samples with
``np.random.RandomState(0).uniform(-0.9, 0.9, shape_in)`` -- a fixed *seed*,
but ``shape_in`` varies across the sweep, so each build consumes a different
number of draws from the same RNG stream and the realised min/max (hence
``y_scale``/``y_zero``) differs slightly build to build. This module checks
whether that residual calibration variation was hiding cleaner shape-driven
structure, using an engineered calibration dataset whose baked-in min/max
(``-0.9``, ``0.9``) is identical for every shape in a sweep.

Findings are in ``docs/axera-reshape-calibration-isolation.md``.

Usage::

    reshape_calibration_isolation.py sensitivity WORK_ROOT
    reshape_calibration_isolation.py sweep WORK_ROOT [C ...]
    reshape_calibration_isolation.py matrix WORK_ROOT [C ...]
    reshape_calibration_isolation.py fields WORK_ROOT [C ...]
"""

from __future__ import annotations

import io
import json
import os
import struct
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


def _model(shape_in, shape_out):
    target = onnx.numpy_helper.from_array(np.array(shape_out, dtype=np.int64), "target")
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


def fixed_range_samples(shape, n=4, lo=-0.9, hi=0.9, seed=100):
    """``n`` samples of ``shape`` whose realised min/max is exactly ``(lo, hi)``
    regardless of ``shape``'s element count, unlike drawing uniform samples and
    hoping enough elements land near the bounds."""
    samples = []
    for i in range(n):
        arr = (
            np.random.RandomState(seed + i)
            .uniform(lo * 0.6, hi * 0.6, shape)
            .astype(np.float32)
        )
        flat = arr.reshape(-1)
        flat[0] = lo
        flat[1] = hi
        samples.append(arr)
    return samples


def build(root: str, name: str, shape_in, shape_out, calib_samples) -> int:
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    onnx.save(_model(shape_in, shape_out), os.path.join(wd, "t.onnx"))
    with tarfile.open(os.path.join(wd, "dataset", "x.tar"), "w") as tar:
        for i, arr in enumerate(calib_samples):
            buf = io.BytesIO()
            np.save(buf, arr.astype(np.float32))
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
                    "calibration_size": len(calib_samples),
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    json.dump(config, open(os.path.join(wd, "config", "step.json"), "w"))
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"rsc-{name}-{os.getpid()}"[:120],
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
    tail = (proc.stdout + proc.stderr)[-300:].replace("\n", " ")
    print(name, "ok" if proc.returncode == 0 else f"FAIL {tail}", flush=True)
    return proc.returncode


def _load_mcode(path: str) -> bytes:
    model = onnx.load(path, load_external_data=False)
    return bytes(
        next(i.raw_data for i in model.graph.initializer if i.name.endswith("_neu"))
    )


def weight_fold_name(c: int) -> str:
    return f"fixed_C{c}"


def build_weight_fold_sweep(root: str, cs) -> None:
    """``Reshape([C,C,3,3] -> [1,C,C,9]) -> Relu`` at every ``C`` in ``cs``, with
    byte-identical baked-in calibration range across every build."""
    for c in cs:
        shape_in = [c, c, 3, 3]
        shape_out = [1, c, c, 9]
        build(
            root,
            weight_fold_name(c),
            shape_in,
            shape_out,
            fixed_range_samples(shape_in),
        )


def seg2(root: str, c: int) -> bytes:
    mc = _load_mcode(os.path.join(root, weight_fold_name(c), "out", "compiled.axmodel"))
    segs = mcode.segments(mc)[1]
    start, length = segs[2][0], segs[2][1]
    return mc[start : start + length]


def matrix(root: str, cs) -> dict[tuple[int, int], int | None]:
    """Pairwise differing-byte counts of segment 2 across ``cs`` (``None`` if the
    segment lengths differ, i.e. the shapes use different total layouts)."""
    seg = {c: seg2(root, c) for c in cs}
    out = {}
    for a in cs:
        for b in cs:
            if len(seg[a]) != len(seg[b]):
                out[(a, b)] = None
            else:
                out[(a, b)] = sum(
                    1 for i in range(len(seg[a])) if seg[a][i] != seg[b][i]
                )
    return out


# Fields decoded for the C in {40, 56, 60, 64} sub-cluster (see the doc for the
# {44, 48, 52} counter-examples and the un-decoded remainder).
FIELD_OFFSETS = {
    "row_bytes_36c": 537,  # uint16 LE, value 36*C
    "c_minus_1_a": 550,  # uint8, value C-1
    "c_minus_1_b": 555,  # uint8, value C-1
    "half_c_sq_minus_1": 825,  # uint16 LE, value C*C//2 - 1
}


def decode_fields(root: str, c: int) -> dict[str, int]:
    s = seg2(root, c)
    return {
        "row_bytes_36c": struct.unpack_from("<H", s, FIELD_OFFSETS["row_bytes_36c"])[0],
        "c_minus_1_a": s[FIELD_OFFSETS["c_minus_1_a"]],
        "c_minus_1_b": s[FIELD_OFFSETS["c_minus_1_b"]],
        "half_c_sq_minus_1": struct.unpack_from(
            "<H", s, FIELD_OFFSETS["half_c_sq_minus_1"]
        )[0],
    }


def fields_match_formula(c: int, fields: dict[str, int]) -> bool:
    return (
        fields["row_bytes_36c"] == 36 * c
        and fields["c_minus_1_a"] == c - 1
        and fields["c_minus_1_b"] == c - 1
        and fields["half_c_sq_minus_1"] == c * c // 2 - 1
    )


def _cmd_sensitivity(root: str) -> None:
    shape_in = [32, 32, 3, 3]
    shape_out = [1, 32, 32, 9]
    narrow = [np.random.RandomState(i).uniform(-0.9, 0.9, shape_in) for i in range(4)]
    wide = [np.random.RandomState(i).uniform(-9.0, 9.0, shape_in) for i in range(4)]
    build(root, "sens_narrow", shape_in, shape_out, narrow)
    build(root, "sens_wide", shape_in, shape_out, wide)
    a = _load_mcode(os.path.join(root, "sens_narrow", "out", "compiled.axmodel"))
    b = _load_mcode(os.path.join(root, "sens_wide", "out", "compiled.axmodel"))
    if len(a) != len(b):
        print(f"different lengths: {len(a)} vs {len(b)}")
        return
    diffs = [i for i in range(len(a)) if a[i] != b[i] and not (301 <= i < 326)]
    print(
        f"length {len(a)}, {len(diffs)} bytes differ outside the noise window: {diffs}"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, root = argv[0], argv[1]
    cs = [int(c) for c in argv[2:]] or [40, 44, 48, 52, 56, 60, 64]
    if cmd == "sensitivity":
        _cmd_sensitivity(root)
    elif cmd == "sweep":
        build_weight_fold_sweep(root, cs)
    elif cmd == "matrix":
        m = matrix(root, cs)
        header = "     " + " ".join(f"{c:4d}" for c in cs)
        print(header)
        for a in cs:
            row = " ".join(
                "   x" if m[(a, b)] is None else f"{m[(a, b)]:4d}" for b in cs
            )
            print(f"{a:4d} {row}")
    elif cmd == "fields":
        for c in cs:
            f = decode_fields(root, c)
            print(c, f, "match" if fields_match_formula(c, f) else "NO MATCH")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
