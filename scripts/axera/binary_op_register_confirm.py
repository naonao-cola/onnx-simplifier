"""Reproduction harness for ``docs/axera-binary-op-register-semantics.md``:
build ``Add``/``Sub``/``Mul``/``Div(x[1,16,8,8], z[1,16,8,8])`` with engineered
calibration, then device-confirm register semantics by single-value patching.

Usage::

    binary_op_register_confirm.py build WORK_ROOT
        # 12 Pulsar2 7.0-lite builds: {add,sub,mul,div} x {c1,c2,c3}
    binary_op_register_confirm.py patch WORK_ROOT SPEC.json
        # SPEC: [[build, label, "mcode"|"params", offset, "f32"|"u16"|"u8set", factor], ...]

Calibration settings (input ranges ``x in [-a, a]``, ``z in [-b, b]``, with the
extremes planted so MinMax hits them exactly; ``Div`` uses ``z in [b/2, b]``):
``c1`` a=1, b=1; ``c2`` a=2, b=0.5; ``c3`` a=0.5, b=3.

``patch`` edits ONE value of a compiled model by a known factor (a float32 in
the mcode, a uint16 Q15 word in ``npu_params``, or sets one byte), runs it on
the AX8850 in ``axcl-vm`` under three stimuli (x only, z only, both), and fits
``patched = a * control + b`` per output lane (flat index ``mod 4``). The
shared device lock is taken per run; every patched run is followed by an
unpatched health run, and results are saved after each experiment so the run
is resumable. Never patch a scale register's lane-0 companion ``W`` slot --
``docs/axera-teng2-register-decode.md`` found that it can kill the card.
"""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
from onnx import TensorProto, helper

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SHAPE = (1, 16, 8, 8)
CONFIGS = {"c1": (1.0, 1.0), "c2": (2.0, 0.5), "c3": (0.5, 3.0)}
OPS = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}


def _calibration(op: str, a: float, b: float, n: int = 4, seed: int = 0):
    rng = np.random.RandomState(seed)
    xs, zs = [], []
    for _ in range(n):
        x = rng.uniform(-a, a, SHAPE).astype(np.float32)
        z = (
            rng.uniform(b / 2, b, SHAPE) if op == "div" else rng.uniform(-b, b, SHAPE)
        ).astype(np.float32)
        xf, zf = x.reshape(-1), z.reshape(-1)
        xf[0], xf[1] = a, -a
        zf[0], zf[1] = (b / 2, b / 2) if op == "div" else (b, -b)
        xs.append(x)
        zs.append(z)
    return xs, zs


def build_one(root: str, op: str, cfg: str) -> tuple[str, int]:
    name = f"{op}_{cfg}"
    wd = os.path.join(root, name)
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return name, 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node(OPS[op], ["x", "z"], ["y"])],
        "g",
        [
            helper.make_tensor_value_info(i, TensorProto.FLOAT, SHAPE)
            for i in ("x", "z")
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, os.path.join(wd, "t.onnx"))
    a, b = CONFIGS[cfg]
    inputs = []
    for tname, arrays in zip(("x", "z"), _calibration(op, a, b)):
        with tarfile.open(os.path.join(wd, "dataset", f"{tname}.tar"), "w") as tar:
            for k, arr in enumerate(arrays):
                buf = io.BytesIO()
                np.save(buf, arr)
                info = tarfile.TarInfo(f"{k}.npy")
                info.size = len(buf.getvalue())
                buf.seek(0)
                tar.addfile(info, buf)
        inputs.append(
            {
                "tensor_name": tname,
                "calibration_dataset": f"./dataset/{tname}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(arrays),
            }
        )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": inputs,
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
            f"binreg-{name}-{os.getpid()}",
            "-v",
            f"{wd}:/data",
            "pulsar2:7.0-lite",
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
    return name, result.returncode


def build(root: str) -> None:
    jobs = [(op, cfg) for op in OPS for cfg in CONFIGS]
    with ThreadPoolExecutor(2) as pool:
        for name, rc in pool.map(lambda j: build_one(root, *j), jobs):
            print(name, "ok" if rc == 0 else f"rc={rc}", flush=True)


def stimuli(op: str, cfg: str, seed: int = 5) -> dict:
    """x-only / z-only / both input pairs, inside the calibration range. For
    Mul and Div the "absent" input is held at a constant instead of zero."""
    a, b = CONFIGS[cfg]
    rng = np.random.RandomState(seed)
    x = rng.uniform(-0.9 * a, 0.9 * a, SHAPE).astype(np.float32)
    if op == "div":
        z = rng.uniform(0.6 * b, 0.95 * b, SHAPE).astype(np.float32)
        x0, z0 = (
            np.full(SHAPE, 0.5 * a, np.float32),
            np.full(SHAPE, 0.8 * b, np.float32),
        )
    else:
        z = rng.uniform(-0.9 * b, 0.9 * b, SHAPE).astype(np.float32)
        x0 = (
            np.zeros(SHAPE, np.float32)
            if op in ("add", "sub")
            else np.full(SHAPE, 0.5 * a, np.float32)
        )
        z0 = (
            np.full(SHAPE, 0.5 * b, np.float32)
            if op == "mul"
            else np.zeros(SHAPE, np.float32)
        )
    return {"x_only": (x, z0), "z_only": (x0, z), "both": (x, z)}


def lane_fit(control: np.ndarray, patched: np.ndarray) -> dict:
    """Per output lane (flat index mod 4): least-squares ``patched = a*control + b``."""
    out = {}
    for lane in range(4):
        c, p = control[lane::4], patched[lane::4]
        if np.ptp(c) > 0:
            a, b = np.polyfit(c, p, 1)
            resid = float(np.abs(p - (a * c + b)).max())
        else:
            a, b, resid = 0.0, float(np.mean(p - c)), float(np.ptp(p))
        out[lane] = {
            "a": round(float(a), 4),
            "b": round(float(b), 4),
            "resid": round(resid, 4),
        }
    return out


def _patched(buf: bytes, off: int, kind: str, factor: float) -> tuple[bytes, float]:
    b = bytearray(buf)
    if kind == "f32":
        old = struct.unpack_from("<f", b, off)[0]
        struct.pack_into("<f", b, off, old * factor)
    elif kind == "u16":
        old = struct.unpack_from("<H", b, off)[0]
        struct.pack_into("<H", b, off, min(65535, max(0, round(old * factor))))
    elif kind == "u8set":
        old = b[off]
        b[off] = int(factor)
    else:
        raise ValueError(kind)
    return bytes(b), old


def _with(
    model: onnx.ModelProto, mc: bytes | None = None, params: bytes | None = None
) -> bytes:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for init in m.graph.initializer:
        if mc is not None and init.name.endswith("_neu"):
            init.raw_data = mc
        if params is not None and init.name == "npu_params":
            init.raw_data = params
    return m.SerializeToString()


def _run(model_bytes: bytes, x: np.ndarray, z: np.ndarray):
    from teng2_fault_injection_add import _run_on_device_nolock, device_lock

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.axmodel")
        with open(path, "wb") as f:
            f.write(model_bytes)
        with device_lock():
            outs, log = _run_on_device_nolock(
                path, {"x": x.tobytes(), "z": z.tobytes()}
            )
    if outs is None:
        return None, log[-400:]
    return np.frombuffer(outs["y"], np.float32).copy(), None


def patch_experiments(root: str, spec_path: str) -> None:
    from binary_op_register_decode import load

    out_path = os.path.join(root, "confirm_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name, label, where, off, kind, factor in json.load(open(spec_path)):
        key = f"{name}:{label}"
        if key in results:
            continue
        path = os.path.join(root, name, "out", "compiled.axmodel")
        model = onnx.load(path, load_external_data=False)
        mc, params = load(path)
        if where == "mcode":
            new, old = _patched(mc, off, kind, factor)
            patched = _with(model, mc=new)
        else:
            new, old = _patched(params, off, kind, factor)
            patched = _with(model, params=new)
        base = _with(model)
        entry = {
            "where": where,
            "offset": off,
            "kind": kind,
            "factor": factor,
            "old": old,
            "stimuli": {},
        }
        op, cfg = name.split("_")
        for cond, (x, z) in stimuli(op, cfg).items():
            control, err = _run(base, x, z)
            if control is None:
                raise SystemExit(f"control run failed ({key}/{cond}): {err}")
            got, err = _run(patched, x, z)
            health, _ = _run(base, x, z)
            ok = health is not None and bool(np.array_equal(health, control))
            entry["stimuli"][cond] = (
                {"fault": err, "health_ok": ok}
                if got is None
                else {**lane_fit(control, got), "health_ok": ok}
            )
            if not ok:
                results[key] = entry
                json.dump(results, open(out_path, "w"), indent=1)
                raise SystemExit(
                    f"health run failed after {key}/{cond}: stop and check the card"
                )
            if got is None:
                break
        results[key] = entry
        json.dump(results, open(out_path, "w"), indent=1)
        print(key, json.dumps(entry["stimuli"]), flush=True)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "build":
        build(argv[1])
        return 0
    if len(argv) == 3 and argv[0] == "patch":
        patch_experiments(argv[1], argv[2])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
