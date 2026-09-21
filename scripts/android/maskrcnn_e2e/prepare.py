#!/usr/bin/env python3
"""Prepare the MaskRCNN-12-qdq model for a TVM Hexagon + ONNX Runtime split deployment.

1. Download `onnxmodelzoo/MaskRCNN-12-qdq` (Hugging Face).
2. Fix the image input to a static canvas and simplify with onnxsim, which constant-folds the
   shape arithmetic (Shape/Gather/Resize scales) that otherwise hides the static backbone.
3. Split the simplified graph automatically: the *static dense* subgraph (nodes that depend only
   on `image` and use dense operators: Conv, QuantizeLinear/DequantizeLinear, MaxPool, Add, Relu,
   Resize, Sigmoid, Reshape, Transpose, ...) becomes `backbone.onnx` (ResNet-50 + FPN + RPN head,
   76 of the model's 81 convolutions); everything else (proposal decoding, NMS, RoiAlign, box and
   mask heads, post-processing) becomes `rest.onnx`, which takes the backbone's outputs.

Both halves are checked to reproduce the full model bit-exactly in ONNX Runtime.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils

import onnxsim

DENSE = {
    "Conv", "Relu", "Add", "MaxPool", "QuantizeLinear", "DequantizeLinear", "Clip", "Sigmoid",
    "Mul", "Sub", "Unsqueeze", "Resize", "Transpose", "Reshape", "Flatten", "Concat", "Cast",
    "Constant", "Identity",
}  # fmt: skip


def split(model: onnx.ModelProto):
    """Return the tensor names crossing from the static dense subgraph into the rest."""
    graph = model.graph
    nodes = list(graph.node)
    constants = {i.name for i in graph.initializer} | {
        o for n in nodes if n.op_type == "Constant" for o in n.output
    }
    available = {"image"} | constants
    dense_nodes: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, node in enumerate(nodes):
            if index in dense_nodes or node.op_type not in DENSE:
                continue
            if all(name in available or name == "" for name in node.input):
                dense_nodes.add(index)
                available.update(node.output)
                changed = True
    consumers = collections.defaultdict(list)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(index)
    depends_on_image = {"image"}
    for node in nodes:
        if any(name in depends_on_image for name in node.input):
            depends_on_image.update(node.output)
    outputs = {o.name for o in graph.output}
    return [
        out
        for index in sorted(dense_nodes)
        for out in nodes[index].output
        if out in depends_on_image
        and (any(c not in dense_nodes for c in consumers[out]) or out in outputs)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("maskrcnn_work"))
    parser.add_argument("--model", type=Path, help="local MaskRCNN-12-qdq.onnx (default: download)")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1088)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    if model_path is None:
        from huggingface_hub import hf_hub_download

        model_path = Path(hf_hub_download("onnxmodelzoo/MaskRCNN-12-qdq", "MaskRCNN-12-qdq.onnx"))
    model = onnx.load(str(model_path))
    simplified, ok = onnxsim.simplify(
        model, overwrite_input_shapes={"image": [3, args.height, args.width]}
    )
    assert ok
    print(f"simplified: {len(model.graph.node)} -> {len(simplified.graph.node)} nodes")
    simplified_path = args.workdir / "maskrcnn_sim.onnx"
    onnx.save(simplified, str(simplified_path))

    boundary = split(simplified)
    (args.workdir / "backbone_outputs.txt").write_text("\n".join(boundary))
    outputs = [o.name for o in simplified.graph.output]
    utils.extract_model(str(simplified_path), str(args.workdir / "backbone.onnx"), ["image"], boundary)
    utils.extract_model(
        str(simplified_path), str(args.workdir / "rest.onnx"), boundary, outputs, check_model=False
    )
    backbone = onnx.load(str(args.workdir / "backbone.onnx"))
    rest = onnx.load(str(args.workdir / "rest.onnx"))
    print(
        f"backbone: {len(backbone.graph.node)} nodes, "
        f"{sum(n.op_type == 'Conv' for n in backbone.graph.node)} convs, {len(boundary)} outputs; "
        f"rest: {len(rest.graph.node)} nodes"
    )

    # Bit-exactness of the split in ONNX Runtime.
    rng = np.random.default_rng(0)
    image = rng.normal(0, 30, (3, args.height, args.width)).astype("float32")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    full = ort.InferenceSession(str(model_path), options, providers=providers).run(
        None, {"image": image}
    )
    features = ort.InferenceSession(
        str(args.workdir / "backbone.onnx"), options, providers=providers
    ).run(None, {"image": image})
    split_out = ort.InferenceSession(str(args.workdir / "rest.onnx"), options, providers=providers).run(
        None, dict(zip(boundary, features))
    )
    for a, b in zip(full, split_out):
        assert a.shape == b.shape and np.array_equal(a, b), "split does not reproduce the model"
    print("backbone + rest reproduce the full model exactly in ONNX Runtime")


if __name__ == "__main__":
    main()
