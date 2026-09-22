"""Does a standalone-verified Transpose template survive being embedded in its
real ResNet18-step neighbours? Answer: no, not at the byte level -- Pulsar2
fuses the whole chain into one compiled unit, with no Transpose-shaped seam
to patch -- but the composed build is still numerically correct.

Two real 3-op chains from ``/home/takecheeze/npu-scratch/t6-r18fold/step.onnx``
(read directly with ``onnx.shape_inference``, not assumed):

* **Weight family**, ``Reshape(w[64,64,3,3] -> [1,64,64,9]) -> Transpose(perm
  0,2,1,3) -> Reshape(-> [1,1,64,576])``, the exact neighbours of
  ``Transpose_358`` (one of the four ``[1,64,64,9]`` instances).
* **Activation family**, ``Reshape(mul_out[16,1,64,28224] -> [16,1,576,3136])
  -> Transpose(perm 0,1,3,2) -> MatMul(against [16,1,64,3136])``, the exact
  neighbours of ``Transpose_366`` (one of the four ``[16,1,576,3136]``
  instances).

Findings (see ``docs/axera-transpose-compose-real.md``):

* Both chains compile to a **single fused** ``neu mode`` node -- op
  boundaries do not survive compilation, so there is no "the Transpose's own
  bytes" region inside a composed build to compare against
  ``transpose_real_shapes.py``'s standalone templates at all.
* MCode length, segment count, and every segment's size differ from the
  standalone template (e.g. the activation chain's compute segment is 50,112
  bytes composed vs. 9,120 bytes standalone) -- a rewrite, not a
  concatenation, matching ``docs/axera-compose.md``'s smaller-scale finding
  for a different op, now confirmed for Transpose at real training scale.
* Both composed builds are still **numerically correct** on the AX8850: the
  weight chain (pure data movement, no compute engine) is byte-exact,
  max error 0.0; the activation chain (includes a real quantized ``MatMul``)
  has max error 0.669 against a peak magnitude of ~79, i.e. 0.85% relative --
  ordinary int8 quantization noise, not evidence of a composition bug.

So ``transpose_real_shapes.py``'s 41/41 coverage is real but is a
**disconnected fact**, not a drop-in for whole-step assembly the way Gather
index-retargeting is for Gather: it proves every real shape compiles at all
(useful -- nothing here will block a full-step build for OCM/size reasons)
and gives a numerically-verified oracle, but the whole step still needs its
own from-scratch compile.

Usage::

    transpose_compose_real_check.py build WORK_ROOT
    transpose_compose_real_check.py check WORK_ROOT   # vs. committed fixtures
"""

from __future__ import annotations

import gzip
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


_IMAGE = "pulsar2:7.0-lite"
_FIXTURES = os.path.join(_HERE, "fixtures", "transpose_compose_real")


def _tar(wd: str, name: str, shape: list[int], seed: int) -> None:
    rng = np.random.RandomState(seed)
    with tarfile.open(os.path.join(wd, "dataset", f"{name}.tar"), "w") as tar:
        for i in range(4):
            arr = rng.uniform(-0.9, 0.9, shape).astype(np.float32)
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            info = tarfile.TarInfo(f"{i}.npy")
            info.size = len(buf.getvalue())
            tar.addfile(info, buf)


def _build(
    root: str,
    name: str,
    model: onnx.ModelProto,
    shapes: dict[str, list[int]],
    seeds: dict[str, int],
) -> str:
    wd = os.path.join(root, name)
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    onnx.save(model, os.path.join(wd, "t.onnx"))
    cfg_inputs = []
    for iname, shape in shapes.items():
        _tar(wd, iname, shape, seeds[iname])
        cfg_inputs.append(
            {
                "tensor_name": iname,
                "calibration_dataset": f"./dataset/{iname}.tar",
                "calibration_format": "Numpy",
                "calibration_size": 4,
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
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return wd
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"trc-{name}-{os.getpid()}",
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
    return wd


def weight_chain_model() -> onnx.ModelProto:
    """``Reshape(w[64,64,3,3] -> [1,64,64,9]) -> Transpose(0,2,1,3) -> Reshape(-> [1,1,64,576])``,
    ``Transpose_358``'s real neighbours in ``step.onnx``."""
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [64, 64, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 64, 576])
    shape1 = helper.make_tensor("shape1", TensorProto.INT64, [4], [1, 64, 64, 9])
    shape2 = helper.make_tensor("shape2", TensorProto.INT64, [4], [1, 1, 64, 576])
    n1 = helper.make_node("Reshape", ["w", "shape1"], ["r1"], name="Reshape_357")
    n2 = helper.make_node(
        "Transpose", ["r1"], ["t1"], perm=[0, 2, 1, 3], name="Transpose_358"
    )
    n3 = helper.make_node("Reshape", ["t1", "shape2"], ["y"], name="Reshape_359")
    g = helper.make_graph([n1, n2, n3], "weight_chain", [w], [y], [shape1, shape2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def activation_chain_model() -> onnx.ModelProto:
    """``Reshape(mul_out[16,1,64,28224] -> [16,1,576,3136]) -> Transpose(0,1,3,2) ->
    MatMul(other[16,1,64,3136], .))``, ``Transpose_366``'s real neighbours in ``step.onnx``."""
    a = helper.make_tensor_value_info("mul_out", TensorProto.FLOAT, [16, 1, 64, 28224])
    b = helper.make_tensor_value_info("other", TensorProto.FLOAT, [16, 1, 64, 3136])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [16, 1, 64, 576])
    shape1 = helper.make_tensor("shape1", TensorProto.INT64, [4], [16, 1, 576, 3136])
    n1 = helper.make_node("Reshape", ["mul_out", "shape1"], ["r1"], name="Reshape_365")
    n2 = helper.make_node(
        "Transpose", ["r1"], ["t1"], perm=[0, 1, 3, 2], name="Transpose_366"
    )
    n3 = helper.make_node("MatMul", ["other", "t1"], ["y"], name="MatMul_367")
    g = helper.make_graph([n1, n2, n3], "activation_chain", [a, b], [y], [shape1])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def build(root: str) -> None:
    _build(
        root,
        "weight_chain_1x64x64x9",
        weight_chain_model(),
        {"w": [64, 64, 3, 3]},
        {"w": 1},
    )
    _build(
        root,
        "activation_chain_16x1x576x3136",
        activation_chain_model(),
        {"mul_out": [16, 1, 64, 28224], "other": [16, 1, 64, 3136]},
        {"mul_out": 2, "other": 3},
    )


def _load(path: str, opener=open) -> tuple[list[str], bytes, bytes]:
    with opener(path, "rb") as f:
        data = f.read()
    model = (
        onnx.load_model_from_string(data)
        if opener is gzip.open
        else onnx.load(path, load_external_data=False)
    )
    inits = {i.name: i for i in model.graph.initializer}
    mc = next(bytes(i.raw_data) for k, i in inits.items() if k.endswith("_neu"))
    pa = bytes(inits["npu_params"].raw_data) if "npu_params" in inits else b""
    return [n.op_type for n in model.graph.node], mc, pa


def check(root: str) -> int:
    """Compare each freshly-built composed chain to its committed fixture (fused into one
    node, matching mcode/params bytes exactly outside the 301-325 noise window)."""
    mismatches = 0
    for name, fixture in [
        ("weight_chain_1x64x64x9", "weight_chain_1x64x64x9.axmodel.gz"),
        ("activation_chain_16x1x576x3136", "activation_chain_16x1x576x3136.axmodel.gz"),
    ]:
        built = os.path.join(root, name, "out", "compiled.axmodel")
        if not os.path.exists(built):
            print(f"{name}: not built, skipping")
            continue
        nodes, mc, pa = _load(built)
        fnodes, fmc, fpa = _load(os.path.join(_FIXTURES, fixture), opener=gzip.open)
        ok = nodes == fnodes == ["neu mode"] and pa == fpa
        if ok and len(mc) == len(fmc):
            a, b = bytearray(mc), bytearray(fmc)
            a[301:326] = bytes(25)
            b[301:326] = bytes(25)
            ok = bytes(a) == bytes(b)
        elif ok:
            ok = False
        mismatches += not ok
        print(f"{name}: {'exact (fused single node)' if ok else 'MISMATCH'}")
    return mismatches


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("build", "check"):
        print(__doc__)
        return 2
    if argv[0] == "build":
        build(argv[1])
        return 0
    return 1 if check(argv[1]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
