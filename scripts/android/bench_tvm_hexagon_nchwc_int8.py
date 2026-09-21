#!/usr/bin/env python3
"""Benchmark TVM's tensorized Hexagon vrmpy NCHWc int8 convolution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tvm
from tvm import te, topi
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import register_linker
from tvm.rpc.tracker import Tracker


def _configure_linker():
    toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
    if not toolchain:
        return
    clang_link = Path(toolchain) / "bin" / "hexagon-clang++"
    wrapper = Path("/tmp/tvm-hexagon-link-wrapper")
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        f"clang = {str(clang_link)!r}\n"
        "args = ['-Wl,--export-dynamic' if x == '-export-dynamic' else x for x in sys.argv[1:]]\n"
        "raise SystemExit(subprocess.call([clang, *args]))\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    register_linker(lambda: str(wrapper))


def _pack_input(x):
    n, channels, height, width = x.shape
    return x.reshape(n, channels // 32, 32, height, width).transpose(0, 1, 3, 4, 2).copy()


def _pack_weight(weight):
    out_channels, in_channels, kh, kw = weight.shape
    return (
        weight.reshape(out_channels // 32, 32, in_channels // 32, 8, 4, kh, kw)
        .transpose(0, 2, 5, 6, 3, 1, 4)
        .copy()
    )


def _unpack_output(output):
    n, out_chunks, height, width, block = output.shape
    return output.transpose(0, 1, 4, 2, 3).reshape(n, out_chunks * block, height, width)


def _reference(x, weight, kernel_size, padding):
    n, _, height, width = x.shape
    out_channels, in_channels, _, _ = weight.shape
    pad = kernel_size // 2 if padding else 0
    padded = np.pad(x.astype("int32"), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    result = np.zeros((n, out_channels, height, width), dtype="int32")
    w32 = weight.astype("int32")
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            result += np.einsum(
                "nchw,oc->nohw",
                padded[:, :, ky : ky + height, kx : kx + width],
                w32[:, :, ky, kx],
                optimize=True,
            )
    return result


def _build(input_shape, weight_shape, target):
    n, channels, height, width = input_shape
    out_channels, _, kh, kw = weight_shape
    x = te.placeholder((n, channels // 32, height, width, 32), "uint8", name="x_nchwc")
    weight = te.placeholder(
        (out_channels // 32, channels // 32, kh, kw, 8, 32, 4), "int8", name="weight_nchwc"
    )
    output = topi.hexagon.conv2d_NCHWc_int8(
        x,
        weight,
        (1, 1),
        (kh // 2, kw // 2),
        (1, 1),
        "NCHW32c",
        "NCHW32c",
        out_dtype="int32",
    )
    schedule = topi.hexagon.schedule_conv2d_NCHWc_int8([output])
    module = tvm.build(schedule, [x, weight, output], target=target, name="main")
    return module, tuple(int(dimension) for dimension in output.shape)


def _build_scalar_nchw(input_shape, weight_shape, target):
    n, channels, height, width = input_shape
    out_channels, _, kh, kw = weight_shape
    pad_h, pad_w = kh // 2, kw // 2
    x = te.placeholder(input_shape, "uint8", name="x_nchw")
    weight = te.placeholder(weight_shape, "int8", name="weight_oihw")
    padded = te.compute(
        (n, channels, height + 2 * pad_h, width + 2 * pad_w),
        lambda b, c, y, z: tvm.tir.if_then_else(
            tvm.tir.all(y >= pad_h, y < height + pad_h, z >= pad_w, z < width + pad_w),
            x[b, c, y - pad_h, z - pad_w],
            tvm.tir.const(0, "uint8"),
        ),
        name="pad",
    )
    rc = te.reduce_axis((0, channels), name="rc")
    ry = te.reduce_axis((0, kh), name="ry")
    rx = te.reduce_axis((0, kw), name="rx")
    output = te.compute(
        (n, out_channels, height, width),
        lambda b, oc, y, z: te.sum(
            padded[b, rc, y + ry, z + rx].astype("int32")
            * weight[oc, rc, ry, rx].astype("int32"),
            axis=[rc, ry, rx],
        ),
        name="output",
    )
    schedule = te.create_schedule(output.op)
    schedule[padded].compute_inline()
    batch, oc, oh, ow = schedule[output].op.axis
    ow_outer, ow_inner = schedule[output].split(ow, factor=8)
    outer = schedule[output].fuse(batch, oc, oh, ow_outer)
    schedule[output].reorder(outer, ow_inner, *schedule[output].op.reduce_axis)
    schedule[output].vectorize(ow_inner)
    schedule[output].parallel(outer)
    module = tvm.build(schedule, [x, weight, output], target=target, name="main")
    return module, tuple(int(dimension) for dimension in output.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--kernels", default="1,3", help="Comma-separated kernel sizes")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _configure_linker()

    target = tvm.target.hexagon("v73")
    target = tvm.target.Target(target, host=target)
    tracker = Tracker(host="127.0.0.1", port=9195)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9195,
            "rpc_server_port": 7075,
            "workspace_base": "/data/local/tmp/tvm_hexagon_nchwc_int8",
            "adb_server_socket": None,
        },
    )
    rng = np.random.default_rng(42)
    try:
        launcher.start_server()
        for kernel_size in map(int, args.kernels.split(",")):
            input_shape = (1, 64, 56, 56)
            weight_shape = (64, 64, kernel_size, kernel_size)
            x_np = rng.integers(0, 16, size=input_shape, dtype=np.uint8)
            weight_np = rng.integers(-8, 8, size=weight_shape, dtype=np.int8)
            expected = _reference(x_np, weight_np, kernel_size, padding=True)
            baseline, baseline_shape = _build_scalar_nchw(
                input_shape, weight_shape, target
            )
            assert baseline_shape == (
                input_shape[0], weight_shape[0], input_shape[2], input_shape[3]
            )
            baseline_path = Path("/tmp") / f"tvm_hexagon_scalar_nchw_int8_{kernel_size}x{kernel_size}.so"
            baseline.save(str(baseline_path))
            module, output_shape = _build(input_shape, weight_shape, target)
            path = Path("/tmp") / f"tvm_hexagon_nchwc_int8_{kernel_size}x{kernel_size}.so"
            module.save(str(path))
            with launcher.create_session() as session:
                dsp = session.device
                remote = session.load_module(session.upload(str(baseline_path), baseline_path.name))
                x = tvm.nd.array(x_np, dsp)
                weight = tvm.nd.array(weight_np, dsp)
                output = tvm.nd.empty(baseline_shape, "int32", dsp)
                remote["main"](x, weight, output)
                got = output.numpy()
                np.testing.assert_array_equal(got, expected)
                baseline_samples = remote.time_evaluator(
                    "main", dsp, number=1, repeat=args.repeat
                )(x, weight, output).results
                baseline_ms = float(np.median(baseline_samples) * 1e3)
            with launcher.create_session() as session:
                dsp = session.device
                remote = session.load_module(session.upload(str(path), path.name))
                x = tvm.nd.array(_pack_input(x_np), dsp)
                weight = tvm.nd.array(_pack_weight(weight_np), dsp)
                output = tvm.nd.empty(output_shape, "int32", dsp)
                remote["main"](x, weight, output)
                got = _unpack_output(output.numpy())
                np.testing.assert_array_equal(got, expected)
                samples = remote.time_evaluator(
                    "main", dsp, number=1, repeat=args.repeat
                )(x, weight, output).results
                elapsed_ms = float(np.median(samples) * 1e3)
                macs = input_shape[0] * weight_shape[0] * input_shape[2] * input_shape[3]
                macs *= input_shape[1] * kernel_size * kernel_size
                print(
                    f"uint8xint8_vrmpy_{kernel_size}x{kernel_size}: "
                    f"scalar_nchw={baseline_ms:.3f} ms, vrmpy_nchwc={elapsed_ms:.3f} ms, "
                    f"speedup={baseline_ms / elapsed_ms:.2f}x, MACs={macs:,}, "
                    f"{macs / elapsed_ms / 1e6:.2f} GMAC/s, exact={np.array_equal(got, expected)}",
                    flush=True,
                )
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
