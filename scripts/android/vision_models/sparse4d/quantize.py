"""int8 image backbone for the Sparse4D frame graphs, with onnxsim's whole-graph quantizer.

  python quantize.py --data <nuscenes-mini> --work <work> [--method minmax]

Quantizes only the ResNet-50 + FPN nodes (uint8 activations, int8 per-channel weights) of
frame_first.sim.onnx / frame_temp.sim.onnx; the decoder (anchor encoder, attention, DFA,
refinement) stays float (fp16 on the HTP). The FPN outputs get a QDQ where the float decoder
reads them. Calibration: the first 3 keyframes of 4 scenes other than the eval scene (the same set
bevformer_tiny calibrates on: scene-0061, -0553, -0757 and -1077, a night scene), run through
frame_first (the backbone is identical in both graphs); frame_temp reuses those ranges.
Writes frame_first.q8.onnx / frame_temp.q8.onnx.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from data import NuScenesMini

from onnxsim.calibration import calibrate
from onnxsim.full_qdq import quantize_full_qdq

CALIB_SCENES = ["scene-0061", "scene-0553", "scene-0757", "scene-1077"]


def backbone_nodes(model):
    """The image backbone + FPN: every node the GridSamples' value maps (input 0) depend on."""
    prod = {o: n for n in model.graph.node for o in n.output}
    todo = [n.input[0] for n in model.graph.node if n.op_type == "GridSample"]
    seen = set()
    while todo:
        n = prod.get(todo.pop())
        if n is None or n.name in seen:
            continue
        seen.add(n.name)
        todo += [x for x in n.input if x]
    return {s for s in seen if not s.startswith("/Cast")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    work = Path(a.work)
    ns = NuScenesMini(a.data)
    data = []
    for sc in CALIB_SCENES:
        for tok in ns.scene_samples(sc)[: a.frames]:
            f = ns.frame(tok)
            data.append({"rgb": f["rgb"], "proj": f["metas"]["projection_mat"].numpy()})
    first = onnx.load(str(work / "frame_first.sim.onnx"))
    inits = {i.name for i in first.graph.initializer}
    keep = set(backbone_nodes(first))
    acts = {x for n in first.graph.node if n.name in keep for x in list(n.input) + list(n.output) if x and x not in inits}
    print(f"{len(keep)} backbone nodes of {len(first.graph.node)} quantized; {len(data)} calibration frames")
    # calibrate once on frame_first; frame_temp has the same backbone tensor names
    ranges = calibrate(first, data, method=a.method, extra_tensor_names=sorted(acts))
    for name in ("frame_first", "frame_temp"):
        m = first if name == "frame_first" else onnx.load(str(work / f"{name}.sim.onnx"))
        keep_m = set(backbone_nodes(m))
        q = quantize_full_qdq(m, None, exclude_nodes=[n.name for n in m.graph.node if n.name not in keep_m],
                              ranges=ranges)
        onnx.save(q, str(work / f"{name}.q8.onnx"))
    print("wrote frame_first.q8.onnx, frame_temp.q8.onnx")


if __name__ == "__main__":
    main()
