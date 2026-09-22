#!/usr/bin/env python3
"""Mask R-CNN split deployment: TVM int8 backbone on a Hexagon DSP + the rest in ONNX Runtime.

Run `prepare.py` first. The backbone (ResNet-50 + FPN + RPN head, static shapes) is imported with
TVM's Relay ONNX frontend, its QDQ pattern is converted to real int8 `qnn.conv2d` with
`FakeQuantizationToInteger`, and it is compiled for Hexagon and run on a connected phone. Its
outputs feed `rest.onnx` (proposal decoding, NMS, RoiAlign, box/mask heads) in ONNX Runtime, and
the final detections are compared with the full model in ONNX Runtime on the same images.

Notes for TVM 0.17 (the version this targets):
* The Relax ONNX frontend rejects 11 operator types in this model (NonZero, RoiAlign,
  NonMaxSuppression, TopK, ConvTranspose, QuantizeLinear, DequantizeLinear, ScatterElements,
  Floor, Not, And); Relay imports the whole graph but its VM compile of the dynamic remainder
  segfaults, so the remainder stays in ONNX Runtime.
* `Session.get_executor_from_factory` does not upload the factory's constants, so the weights are
  loaded explicitly with `load_params`; without it the DSP computes with uninitialised weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnx_compat  # noqa: F401  # onnx.mapping shim for TVM 0.17's ONNX frontend
import onnxruntime as ort
import tvm
import tvm.contrib.hexagon  # noqa: F401
from eval_common import canvas, compare
from tvm import relay
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker


def build_backbone(workdir: Path, height: int, width: int):
    import onnx

    model = onnx.load(str(workdir / "backbone.onnx"))
    mod, _ = relay.frontend.from_onnx(model, shape={"image": (3, height, width)}, freeze_params=True)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FakeQuantizationToInteger(
        hard_fail=False, optional_qnn_ops=["nn.max_pool2d", "nn.relu", "add"]
    )(mod)
    arch = tvm.target.hexagon("v73")
    with tvm.transform.PassContext(opt_level=3):
        return relay.build(mod, target=tvm.target.Target(arch, host=arch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("maskrcnn_work"))
    parser.add_argument("--model", type=Path, required=True, help="original MaskRCNN-12-qdq.onnx")
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--output", type=Path, default=Path("e2e_results.json"))
    args = parser.parse_args()

    names = [n for n in (args.workdir / "backbone_outputs.txt").read_text().split("\n") if n]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    full = ort.InferenceSession(str(args.model), options, providers=providers)
    rest = ort.InferenceSession(str(args.workdir / "rest.onnx"), options, providers=providers)
    inputs = [canvas(path, args.height, args.width) for path in args.images]
    references = [full.run(None, {"image": x}) for x in inputs]

    print("compiling the int8 backbone for Hexagon ...", flush=True)
    lib = build_backbone(args.workdir, args.height, args.width)
    tracker = Tracker(host="127.0.0.1", port=9197)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9197,
            "rpc_server_port": 7077,
            "workspace_base": "/data/local/tmp/tvm_maskrcnn_e2e",
            "adb_server_socket": None,
        },
    )
    rows = []
    try:
        launcher.start_server()
        with launcher.create_session() as session:
            executor = session.get_executor_from_factory(lib)
            executor.load_params(tvm.runtime.save_param_dict(lib.get_params()))
            for path, image, reference in zip(args.images, inputs, references):
                executor.set_input("image", image)
                start = time.time()
                executor.run()
                dsp_seconds = time.time() - start
                features = {n: executor.get_output(i).numpy() for i, n in enumerate(names)}
                start = time.time()
                detections = rest.run(None, features)
                row = {"image": path.name, **compare(reference, detections)}
                row["dsp_backbone_s"] = round(dsp_seconds, 2)
                row["ort_rest_s"] = round(time.time() - start, 3)
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        launcher.stop_server()
        tracker.terminate()
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
