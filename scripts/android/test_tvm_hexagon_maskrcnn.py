#!/usr/bin/env python3
"""Run Mask R-CNN-shaped TVM kernels on a connected Hexagon Android DSP.

This is an opt-in hardware probe, not part of the normal test suite. It takes
the static ResNet/FPN tensor shapes and pooling attributes from a Mask R-CNN
ONNX model, generates representative convolution/pooling kernels with TVM's
Hexagon target, runs them through TVM RPC, and compares against TVM/LLVM CPU
results. Weights and activations are randomized: this checks kernel codegen
and DSP execution, not end-to-end model accuracy or graph delegation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import setuptools  # noqa: F401  # TVM's Hexagon helpers expect it at import time.
import tvm
from tvm import te, topi
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker


def _attrs(node):
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def _shape_map(model):
    result = {}
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        result[value.name] = tuple(
            dim.dim_value if dim.dim_value else None
            for dim in value.type.tensor_type.shape.dim
        )
    for value in model.graph.initializer:
        result[value.name] = tuple(value.dims)
    return result


def _find_conv(model, shape_map, data_shape, weight_shape):
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        actual_data = shape_map.get(node.input[0])
        batch_is_dynamic = (
            actual_data is not None
            and len(actual_data) == 4
            and actual_data[0] is None
            and actual_data[1:] == data_shape[1:]
        )
        if (actual_data == data_shape or batch_is_dynamic) and shape_map.get(node.input[1]) == weight_shape:
            return node
    raise RuntimeError(f"No Conv found with data={data_shape} and weight={weight_shape}")


def _find_pool(model, shape_map, data_shape):
    for node in model.graph.node:
        if node.op_type == "MaxPool" and shape_map.get(node.input[0]) == data_shape:
            return node
    raise RuntimeError(f"No MaxPool found with input={data_shape}")


def _model_workloads(model_path: Path, roi_batch: int):
    model = onnx.load(str(model_path))
    shape_overrides = {}
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 3 and dims[0].dim_value == 3:
            shape_overrides[value.name] = (3, 224, 224)
            break
    if not shape_overrides:
        raise RuntimeError("Expected a three-dimensional [3, height, width] image input")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    # Shape inference preserves symbolic image dimensions in this fixture;
    # set the sample size and infer once more to expose the backbone shapes.
    for value in model.graph.input:
        if value.name in shape_overrides:
            for dim, size in zip(value.type.tensor_type.shape.dim, shape_overrides[value.name]):
                dim.dim_value = size
                dim.ClearField("dim_param")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = _shape_map(model)

    conv_specs = [
        ("resnet_bottleneck_1x1", (1, 64, 56, 56), (64, 64, 1, 1)),
        ("resnet_bottleneck_3x3", (1, 64, 56, 56), (64, 64, 3, 3)),
        ("fpn_3x3", (1, 256, 14, 14), (256, 256, 3, 3)),
        ("roi_mask_head_3x3", (roi_batch, 256, 14, 14), (256, 256, 3, 3)),
    ]
    convs = []
    for name, data_shape, weight_shape in conv_specs:
        node = _find_conv(model, shapes, data_shape, weight_shape)
        attrs = _attrs(node)
        pads = attrs.get("pads", [0, 0, 0, 0])
        convs.append(
            (
                name,
                data_shape,
                weight_shape,
                tuple(attrs.get("strides", [1, 1])),
                (pads[0], pads[1]),
                (pads[2], pads[3]),
            )
        )

    pools = []
    for name, data_shape in [
        ("backbone_stem_maxpool", (1, 64, 112, 112)),
        ("fpn_maxpool", (1, 256, 7, 7)),
    ]:
        node = _find_pool(model, shapes, data_shape)
        attrs = _attrs(node)
        pools.append(
            (
                name,
                data_shape,
                tuple(attrs["kernel_shape"]),
                tuple(attrs.get("strides", [1, 1])),
                tuple(attrs.get("pads", [0, 0, 0, 0])),
            )
        )

    resizes = []
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        input_shape = shapes.get(node.input[0])
        if (
            input_shape is None
            or len(input_shape) != 4
            or input_shape[1] != 256
            or input_shape[2] not in (7, 14, 28)
            or input_shape[3] != input_shape[2]
        ):
            continue
        attrs = _attrs(node)
        if attrs.get("mode", b"nearest") != b"nearest":
            continue
        size = (input_shape[2] * 2, input_shape[3] * 2)
        resizes.append(
            (
                f"fpn_resize_{input_shape[2]}_to_{size[0]}",
                (1, 256, input_shape[2], input_shape[3]),
                size,
                attrs.get("coordinate_transformation_mode", b"half_pixel").decode(),
                attrs.get("nearest_mode", b"round_prefer_floor").decode(),
            )
        )
    if not resizes:
        raise RuntimeError("No nearest-neighbor FPN Resize operators found")
    return convs, pools, resizes


def _hexagon_target():
    target = tvm.target.hexagon("v73")
    return tvm.target.Target(target, host=target)


def _conv_module(name, data_shape, weight_shape, stride, pad_before, pad_after, target):
    x = te.placeholder(data_shape, name="x", dtype="float32")
    weight = te.placeholder(weight_shape, name="weight", dtype="float32")
    bias = te.placeholder((weight_shape[0],), name="bias", dtype="float32")
    padding = tuple(pad_before) + tuple(pad_after)
    conv = topi.nn.conv2d_nchw(x, weight, stride, padding, (1, 1), "float32")
    biased = te.compute(
        conv.shape,
        lambda n, c, h, w: conv[n, c, h, w] + bias[c],
        name="bias_add",
    )
    output = te.compute(
        conv.shape,
        lambda n, c, h, w: te.max(biased[n, c, h, w], 0.0),
        name="relu",
    )
    schedule = topi.hexagon.schedule_conv2d(output, layout="NCHW")
    # NCHW width is contiguous. The factor gives LLVM's Hexagon backend a
    # vectorization opportunity on the V73 target, with tail handling as needed.
    _, width_inner = schedule[conv].split(conv.op.axis[3], factor=32)
    schedule[conv].vectorize(width_inner)
    module = tvm.build(schedule, [x, weight, bias, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _pool_module(data_shape, kernel, stride, pads, target):
    x = te.placeholder(data_shape, name="x", dtype="float32")
    output = topi.nn.pool2d(
        x,
        kernel,
        stride,
        (1, 1),
        pads,
        "max",
        ceil_mode=False,
        layout="NCHW",
    )
    schedule = topi.hexagon.schedule_pool(output, layout="NCHW")
    module = tvm.build(schedule, [x, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _resize_module(data_shape, size, coordinate_mode, rounding_mode, target):
    x = te.placeholder(data_shape, name="x", dtype="float32")
    output = topi.image.resize2d(
        x,
        roi=(0.0, 0.0, 0.0, 0.0),
        size=size,
        layout="NCHW",
        method="nearest_neighbor",
        coordinate_transformation_mode=coordinate_mode,
        rounding_method=rounding_mode,
    )
    schedule = te.create_schedule(output.op)
    module = tvm.build(schedule, [x, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _cpu_conv(data, weight, bias, stride, pad_before, pad_after):
    data_shape, weight_shape = data.shape, weight.shape
    x = te.placeholder(data_shape, name="x", dtype="float32")
    w = te.placeholder(weight_shape, name="weight", dtype="float32")
    b = te.placeholder((weight_shape[0],), name="bias", dtype="float32")
    conv = topi.nn.conv2d_nchw(
        x, w, stride, tuple(pad_before) + tuple(pad_after), (1, 1), "float32"
    )
    biased = te.compute(
        conv.shape,
        lambda n, c, h, wi: conv[n, c, h, wi] + b[c],
        name="bias_add",
    )
    output = te.compute(
        conv.shape,
        lambda n, c, h, wi: te.max(biased[n, c, h, wi], 0.0),
        name="relu",
    )
    schedule = topi.hexagon.schedule_conv2d(output, layout="NCHW")
    _, width_inner = schedule[conv].split(conv.op.axis[3], factor=32)
    schedule[conv].vectorize(width_inner)
    module = tvm.build(schedule, [x, w, b, output], target="llvm", name="main")
    result = tvm.nd.empty(tuple(int(dim) for dim in output.shape))
    module["main"](tvm.nd.array(data), tvm.nd.array(weight), tvm.nd.array(bias), result)
    return result.numpy()


def run(args):
    convs, pools, resizes = _model_workloads(args.model, args.roi_batch)
    target = _hexagon_target()
    rng = np.random.default_rng(11)
    tracker = Tracker(host=args.rpc_host, port=args.tracker_port)
    launcher = HexagonLauncher(
        args.serial,
        rpc_info={
            "rpc_tracker_host": args.rpc_host,
            "rpc_tracker_port": args.tracker_port,
            "rpc_server_port": args.server_port,
            "workspace_base": args.device_workspace,
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        with launcher.create_session() as session:
            for name, data_shape, weight_shape, stride, pad_before, pad_after in convs:
                module, output_shape = _conv_module(
                    name, data_shape, weight_shape, stride, pad_before, pad_after, target
                )
                # The TVM RPC session accepts a module through its shared object
                # export; save keeps the Hexagon ELF standalone and avoids a host link.
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(0, 0.1, data_shape).astype("float32")
                weight = rng.normal(0, 0.05, weight_shape).astype("float32")
                bias = np.zeros((weight_shape[0],), dtype="float32")
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](
                    tvm.nd.array(data, session.device),
                    tvm.nd.array(weight, session.device),
                    tvm.nd.array(bias, session.device),
                    remote_output,
                )
                actual = remote_output.numpy()
                expected = _cpu_conv(data, weight, bias, stride, pad_before, pad_after)
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
                error = float(np.max(np.abs(actual - expected)))
                print(f"PASS {name}: output={output_shape} max_abs_err={error:.8g}", flush=True)

            for name, data_shape, kernel, stride, pads in pools:
                module, output_shape = _pool_module(data_shape, kernel, stride, pads, target)
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(size=data_shape).astype("float32")
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](tvm.nd.array(data, session.device), remote_output)
                actual = remote_output.numpy()

                x = te.placeholder(data_shape, name="x", dtype="float32")
                output = topi.nn.pool2d(
                    x, kernel, stride, (1, 1), pads, "max", ceil_mode=False, layout="NCHW"
                )
                schedule = topi.hexagon.schedule_pool(output, layout="NCHW")
                cpu_module = tvm.build(schedule, [x, output], target="llvm", name="main")
                expected = tvm.nd.empty(output_shape)
                cpu_module["main"](tvm.nd.array(data), expected)
                np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=0)
                print(f"PASS {name}: output={output_shape} exact", flush=True)

            for name, data_shape, size, coordinate_mode, rounding_mode in resizes:
                module, output_shape = _resize_module(
                    data_shape, size, coordinate_mode, rounding_mode, target
                )
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(size=data_shape).astype("float32")
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](tvm.nd.array(data, session.device), remote_output)
                actual = remote_output.numpy()

                x = te.placeholder(data_shape, name="x", dtype="float32")
                output = topi.image.resize2d(
                    x,
                    roi=(0.0, 0.0, 0.0, 0.0),
                    size=size,
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode=coordinate_mode,
                    rounding_method=rounding_mode,
                )
                cpu_module = tvm.build(
                    te.create_schedule(output.op), [x, output], target="llvm", name="main"
                )
                expected = tvm.nd.empty(output_shape)
                cpu_module["main"](tvm.nd.array(data), expected)
                np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=0)
                print(f"PASS {name}: output={output_shape} exact", flush=True)
    finally:
        launcher.stop_server()
        tracker.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Mask R-CNN ONNX model")
    parser.add_argument("--serial", required=True, help="adb device serial")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--tracker-port", type=int, default=9190)
    parser.add_argument("--server-port", type=int, default=7070)
    parser.add_argument("--device-workspace", default="/data/local/tmp/tvm_hexagon")
    parser.add_argument("--artifact-dir", type=Path, default=Path("/tmp/tvm-maskrcnn"))
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
