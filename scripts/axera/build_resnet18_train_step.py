#!/usr/bin/env python3
"""Build the AX8850 ResNet-18d resident training step.

The graph is the documented AXERA training shape: a 64x64 ``resnet18d``
forward model with the final residual block and classifier tail trainable.
The four live weights are the first block's downsample convolution, the two
convolutions in the second block, and the classifier weight.  The actual
update is kept in the graph by :func:`build_resident_step`, so the resulting
model can be used by ``resident_runner`` without copying weights each step.

Usage::

    build_resnet18_train_step.py OUT_DIR BATCH

The batch-1 export is retained in ``OUT_DIR`` and reused for subsequent
batch sizes.  The output step graph is
``OUT_DIR/resnet18_step_b<BATCH>.onnx``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resident_train_step as brts  # noqa: E402

TRAIN_PARAMS = [
    # resnet18d's downsample is AvgPool (index 0) followed by Conv (index 1).
    "layer4.0.downsample.1.weight",
    "layer4.1.conv1.weight",
    "layer4.1.conv2.weight",
    "fc.weight",
]


def export_forward(onnx_path: str) -> None:
    """Export a real, randomly initialized 64x64 ``resnet18d``.

    The dynamo exporter is intentional: the legacy exporter turns the
    ``Sequential`` block names into generic ``onnx::Conv_*`` names, making
    the trainable scope impossible to select reliably.
    """
    import timm
    import torch

    torch.manual_seed(0)
    model = timm.create_model("resnet18d", pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        (x,),
        onnx_path,
        opset_version=18,
        dynamo=True,
        input_names=["x"],
        output_names=["logits"],
    )


def _downgrade_reduce_axes_to_attr(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make the opset-18 dynamo export consumable by the opset-17 step graph."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializers = {t.name: t for t in out.graph.initializer}
    for node in out.graph.node:
        if node.op_type != "ReduceMean" or len(node.input) < 2:
            continue
        axes_name = node.input[1]
        axes_tensor = initializers.get(axes_name)
        if axes_tensor is None:
            continue
        axes = numpy_helper.to_array(axes_tensor).tolist()
        del node.input[1:]
        attrs = [
            attr
            for attr in node.attribute
            if attr.name not in {"axes", "noop_with_empty_axes"}
        ]
        del node.attribute[:]
        node.attribute.extend(attrs)
        node.attribute.append(onnx.helper.make_attribute("axes", axes))
    for opset in out.opset_import:
        if not opset.domain:
            opset.version = 17
    return out


def _fix_flatten_reshape(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace timm's batch-1 classifier flatten shape with ``[-1, C]``."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializers = {t.name: t for t in out.graph.initializer}
    fixed = 0
    for node in out.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape = initializers.get(node.input[1])
        if shape is None:
            continue
        dims = numpy_helper.to_array(shape).tolist()
        if len(dims) == 2 and dims[0] == 1:
            shape.CopyFrom(
                numpy_helper.from_array(
                    np.array([-1, dims[1]], dtype=np.int64), shape.name
                )
            )
            fixed += 1
    if fixed != 1:
        raise RuntimeError(f"expected one classifier flatten reshape, patched {fixed}")
    return out


def _prepare_forward(fwd_path: str, prepared_path: str) -> onnx.ModelProto:
    """Export-independent preprocessing shared by every requested batch.

    Constant folding and shape inference dominate repeated batch sweeps.  The
    resulting graph contains no batch-specific shape, so persist it once and
    only redo ``set_batch`` plus backward construction for later batches.
    """
    if os.path.exists(prepared_path):
        return onnx.load(prepared_path)

    fwd = onnx.load(fwd_path)
    fwd = _downgrade_reduce_axes_to_attr(fwd)
    fwd = _fix_flatten_reshape(fwd)
    onnx.checker.check_model(fwd)
    fwd = onnx.shape_inference.infer_shapes(fwd)
    fwd = brts._fold_constants(fwd)
    onnx.checker.check_model(fwd)
    onnx.save(fwd, prepared_path)
    return fwd


def build_step(out_dir: str, batch: int, metric_scale: float | None = None):
    """Build and save one batch-specific ResNet-18 training step."""
    if batch < 1:
        raise ValueError("batch must be positive")
    os.makedirs(out_dir, exist_ok=True)
    fwd_path = os.path.join(out_dir, "resnet18d_fwd.onnx")
    if not os.path.exists(fwd_path):
        export_forward(fwd_path)

    prepared_path = os.path.join(out_dir, "resnet18d_fwd_prepared.onnx")
    fwd = _prepare_forward(fwd_path, prepared_path)

    init_names = {t.name for t in fwd.graph.initializer}
    missing = [p for p in TRAIN_PARAMS if p not in init_names]
    if missing:
        raise RuntimeError(
            f"trainable params not found after export cleanup: {missing}"
        )

    fwd = brts.set_batch(fwd, batch)
    fwd = onnx.shape_inference.infer_shapes(fwd)
    with_loss = brts.add_mse_loss(fwd, "logits", num_classes=1000)
    step_model, state = brts.build_resident_step(
        with_loss, params=TRAIN_PARAMS, metric_scale=metric_scale
    )
    onnx.checker.check_model(step_model)

    step_path = os.path.join(out_dir, f"resnet18_step_b{batch}.onnx")
    onnx.save(step_model, step_path)
    return step_path, state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("batch", type=int)
    parser.add_argument(
        "--metric-scale",
        type=float,
        default=None,
        help="append output-only loss_scaled = loss * SCALE",
    )
    args = parser.parse_args(argv)
    step_path, state = build_step(args.out_dir, args.batch, args.metric_scale)
    model = onnx.load(step_path)
    print(f"batch={args.batch}: {len(model.graph.node)} nodes, wrote {step_path}")
    for param, output in state.items():
        print(f"  state: {param} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
