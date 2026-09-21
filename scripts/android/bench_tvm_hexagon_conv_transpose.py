#!/usr/bin/env python3
"""Compare TVM's generic and parity-specialized stride-2 ConvTranspose on DSP."""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import setuptools  # noqa: F401  # TVM 0.17 imports distutils during module initialization.
import tvm
from test_tvm_hexagon_maskrcnn import (
    _conv_transpose_module,
    _hexagon_target,
    _model_workloads,
)
from tvm import te
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
        # Kernels only use C symbols; a dynamic libc++ dependency would pull in libc++abi,
        # which needs libc symbols (aligned_alloc, __cxa_thread_atexit_impl) the DSP lacks.
        "raise SystemExit(subprocess.call([clang, '-nostdlib++', *args]))\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    register_linker(lambda: str(wrapper))


def _direct_stride2_module(data_shape, weight_shape, bias_shape, target, width_tile):
    """Compute the four output parity planes without dilating and padding input."""
    n, in_channels, in_height, in_width = data_shape
    _, out_channels, kernel_h, kernel_w = weight_shape
    assert kernel_h == kernel_w == 2
    out_height, out_width = in_height * 2, in_width * 2
    data = te.placeholder(data_shape, name="data", dtype="float32")
    weight = te.placeholder(weight_shape, name="weight", dtype="float32")
    bias = te.placeholder(bias_shape, name="bias", dtype="float32")
    rc = te.reduce_axis((0, in_channels), name="rc")
    output = te.compute(
        (n, out_channels, out_height, out_width),
        lambda b, oc, y, x: te.sum(
            data[b, rc, y // 2, x // 2] * weight[rc, oc, y % 2, x % 2]
            + bias[oc] / in_channels,
            axis=rc,
        ),
        name="conv_transpose_direct",
    )
    schedule = te.create_schedule(output.op)
    batch, channel, height, width = schedule[output].op.axis
    width_outer, width_inner = schedule[output].split(width, factor=width_tile)
    outer = schedule[output].fuse(batch, channel, height, width_outer)
    schedule[output].reorder(outer, width_inner, *schedule[output].op.reduce_axis)
    schedule[output].vectorize(width_inner)
    schedule[output].parallel(outer)
    module = tvm.build(schedule, [data, weight, bias, output], target=target, name="main")
    return module, (n, out_channels, out_height, out_width)


def _channel_vectorized_module(
    data_shape, weight_shape, bias_shape, target, channel_tile, input_nhwc=False, pixel_block=0
):
    """Vectorize contiguous output channels and produce NHWC output.

    pixel_block > 0 register-blocks that many input pixels along width: each parity plane is a
    GEMM, so one weight-vector load is reused by pixel_block accumulators kept in registers.
    """
    n, in_channels, in_height, in_width = data_shape
    _, out_channels, kernel_h, kernel_w = weight_shape
    assert kernel_h == kernel_w == 2
    out_height, out_width = in_height * 2, in_width * 2
    data_layout = (
        (n, in_height, in_width, in_channels) if input_nhwc else data_shape
    )
    data = te.placeholder(data_layout, name="data", dtype="float32")
    # Host-side packing turns output-channel loads into contiguous vectors.
    weight = te.placeholder(
        (kernel_h, kernel_w, in_channels, out_channels), name="weight_packed", dtype="float32"
    )
    bias = te.placeholder(bias_shape, name="bias", dtype="float32")
    rc = te.reduce_axis((0, in_channels), name="rc")
    output = te.compute(
        (n, out_height, out_width, out_channels),
        lambda b, y, x, oc: te.sum(
            (data[b, y // 2, x // 2, rc] if input_nhwc else data[b, rc, y // 2, x // 2])
            * weight[y % 2, x % 2, rc, oc]
            + bias[oc] / in_channels,
            axis=rc,
        ),
        name="conv_transpose_nhwc",
    )
    schedule = te.create_schedule(output.op)
    if pixel_block:
        assert in_width % pixel_block == 0
        local = schedule.cache_write(output, "local")
        batch, height, width, channel = schedule[output].op.axis
        row, row_parity = schedule[output].split(height, factor=2)
        col, col_parity = schedule[output].split(width, factor=2)
        col_outer, col_inner = schedule[output].split(col, factor=pixel_block)
        channel_outer, channel_inner = schedule[output].split(channel, factor=channel_tile)
        schedule[output].reorder(
            batch, row, row_parity, col_parity, col_outer, channel_outer, col_inner, channel_inner
        )
        parallel_axis = schedule[output].fuse(batch, row)
        schedule[output].parallel(parallel_axis)
        schedule[local].compute_at(schedule[output], channel_outer)
        lb, ly, lx, lc = schedule[local].op.axis
        (lrc,) = schedule[local].op.reduce_axis
        schedule[local].reorder(lrc, lb, ly, lx, lc)
        lc_outer, lc_inner = schedule[local].split(lc, factor=channel_tile)
        schedule[local].reorder(lrc, lb, ly, lx, lc_outer, lc_inner)
        schedule[local].vectorize(lc_inner)
        schedule[local].unroll(lx)
        schedule[output].vectorize(channel_inner)
        schedule[output].unroll(col_inner)
    else:
        batch, height, width, channel = schedule[output].op.axis
        channel_outer, channel_inner = schedule[output].split(channel, factor=channel_tile)
        outer = schedule[output].fuse(batch, height, width, channel_outer)
        schedule[output].reorder(outer, channel_inner, *schedule[output].op.reduce_axis)
        schedule[output].vectorize(channel_inner)
        schedule[output].parallel(outer)
    module = tvm.build(schedule, [data, weight, bias, output], target=target, name="main")
    return module, (n, out_height, out_width, out_channels)


def _qf32_intrin(dtype, name, *args):
    return tvm.tir.call_llvm_intrin(
        dtype, f"llvm.hexagon.V6.{name}.128B", tvm.tir.const(len(args), "uint32"), *args
    )


def _qf32_module(
    data_shape, weight_shape, target, channel_vectors, pixel_block, unroll=4, parallel=True,
    weight_major=False,
):
    """NHWC ConvTranspose with hand-written HVX qf32 accumulation.

    LLVM converts qf32 <-> sf around every fmul/fadd (4 ops per MAC). Calling the HVX intrinsics
    directly chains vmpy.qf32.sf + vadd.qf32 (2 ops per MAC) and converts once at the end.
    Each iteration keeps pixel_block x channel_vectors accumulators in vector registers, so
    every weight-vector load is reused pixel_block times.
    """
    n, in_channels, in_height, in_width = data_shape
    _, out_channels, kernel_h, kernel_w = weight_shape
    assert kernel_h == kernel_w == 2
    lanes = 32
    assert in_width % pixel_block == 0 and out_channels % (lanes * channel_vectors) == 0
    assert in_channels % unroll == 0
    out_height, out_width = in_height * 2, in_width * 2
    data = te.placeholder((n, in_height, in_width, in_channels), name="data", dtype="float32")
    weight = te.placeholder((2, 2, in_channels, out_channels), name="weight_packed", dtype="float32")
    bias = te.placeholder((out_channels,), name="bias", dtype="float32")

    def body(ins, outs):
        ib = tvm.tir.ir_builder.create()
        # Flat aliases so vector Ramp indices address the whole (contiguous) buffer.
        def flat(buf):
            size = int(np.prod([int(d) for d in buf.shape]))
            return ib.buffer_ptr(tvm.tir.decl_buffer((size,), buf.dtype, data=buf.data))

        data_buf, weight_buf, bias_buf = (flat(b) for b in ins)
        out_buf = flat(outs[0])

        def vec(buf, index, dtype="float32"):
            return buf[tvm.tir.Ramp(index, 1, lanes)]

        kind = "parallel" if parallel else "serial"
        oc_span = lanes * channel_vectors
        oc_blocks = out_channels // oc_span
        col_blocks = in_width // pixel_block
        acc = ib.allocate(
            "int32", (pixel_block * channel_vectors * lanes,), name="acc", scope="local"
        )
        zero = tvm.tir.Broadcast(tvm.tir.const(0, "int32"), lanes)

        def slot(p, v):
            return tvm.tir.Ramp((p * channel_vectors + v) * lanes, 1, lanes)

        def emit_tile(batch, row, parity_y, parity_x, col_block, oc_block):
            oc0 = oc_block * oc_span
            pixel0 = col_block * pixel_block
            for p in range(pixel_block):
                for v in range(channel_vectors):
                    acc[slot(p, v)] = zero
            with ib.for_range(0, in_channels // unroll, name="rc_block") as rc_block:
                for u in range(unroll):
                    rc = rc_block * unroll + u
                    weights = []
                    for v in range(channel_vectors):
                        w_index = (
                            (parity_y * 2 + parity_x) * in_channels + rc
                        ) * out_channels + oc0 + v * lanes
                        weights.append(tvm.tir.reinterpret("int32x32", vec(weight_buf, w_index)))
                    for p in range(pixel_block):
                        d_index = (
                            (batch * in_height + row) * in_width + pixel0 + p
                        ) * in_channels + rc
                        splat = tvm.tir.reinterpret(
                            "int32x32", tvm.tir.Broadcast(data_buf[d_index], lanes)
                        )
                        for v in range(channel_vectors):
                            prod = _qf32_intrin("int32x32", "vmpy.qf32.sf", splat, weights[v])
                            acc[slot(p, v)] = _qf32_intrin(
                                "int32x32", "vadd.qf32", acc[slot(p, v)], prod
                            )
            for p in range(pixel_block):
                out_row = 2 * row + parity_y
                out_col = 2 * (pixel0 + p) + parity_x
                for v in range(channel_vectors):
                    value = tvm.tir.reinterpret(
                        "float32x32", _qf32_intrin("int32x32", "vconv.sf.qf32", acc[slot(p, v)])
                    ) + vec(bias_buf, oc0 + v * lanes)
                    o_index = (
                        (batch * out_height + out_row) * out_width + out_col
                    ) * out_channels + oc0 + v * lanes
                    out_buf[tvm.tir.Ramp(o_index, 1, lanes)] = value

        if weight_major:
            # One task per (parity, oc block): the task's weight slice stays cache-resident
            # while it sweeps every pixel, instead of every task re-streaming all weights.
            with ib.for_range(0, 4 * oc_blocks, name="task", kind=kind) as task:
                parity = task // oc_blocks
                oc_block = task % oc_blocks
                with ib.for_range(0, n * in_height, name="pixel_row") as pixel_row:
                    with ib.for_range(0, col_blocks, name="col_block") as col_block:
                        emit_tile(
                            pixel_row // in_height, pixel_row % in_height,
                            parity // 2, parity % 2, col_block, oc_block,
                        )
        else:
            with ib.for_range(0, n * in_height, name="task", kind=kind) as task:
                for parity_y in range(2):
                    for parity_x in range(2):
                        with ib.for_range(0, col_blocks, name="col_block") as col_block:
                            with ib.for_range(0, oc_blocks, name="oc_block") as oc_block:
                                emit_tile(
                                    task // in_height, task % in_height,
                                    parity_y, parity_x, col_block, oc_block,
                                )
        return ib.get()

    output = te.extern(
        (n, out_height, out_width, out_channels),
        [data, weight, bias],
        body,
        name="conv_transpose_qf32",
        dtype="float32",
    )
    schedule = te.create_schedule(output.op)
    # The Hexagon runtime allocates HVX-aligned buffers; tell LLVM so it emits aligned vmem
    # instead of vmem + valign pairs for every vector load.
    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, name=t.op.name, data_alignment=128, offset_factor=1
        )
        for t in (data, weight, bias, output)
    }
    module = tvm.build(
        schedule, [data, weight, bias, output], target=target, name="main", binds=binds
    )
    return module, (n, out_height, out_width, out_channels)


def _layout_copy_module(shape, target, to_nhwc, channel_tile=None):
    n, channels, height, width = shape
    input_shape = shape if to_nhwc else (n, height, width, channels)
    data = te.placeholder(input_shape, name="layout_input", dtype="float32")
    if to_nhwc:
        output = te.compute(
            (n, height, width, channels),
            lambda b, y, x, c: data[b, c, y, x],
            name="to_nhwc",
        )
        schedule = te.create_schedule(output.op)
        batch, y, x, channel = schedule[output].op.axis
        channel_outer, channel_inner = schedule[output].split(channel, factor=16)
        outer = schedule[output].fuse(batch, y, x, channel_outer)
        schedule[output].reorder(outer, channel_inner)
        schedule[output].vectorize(channel_inner)
    else:
        output = te.compute(
            shape,
            lambda b, c, y, x: data[b, y, x, c],
            name="to_nchw",
        )
        schedule = te.create_schedule(output.op)
        batch, channel, y, x = schedule[output].op.axis
        if channel_tile:
            channel_outer, channel_inner = schedule[output].split(channel, factor=channel_tile)
            schedule[output].reorder(batch, channel_outer, y, x, channel_inner)
            outer = schedule[output].fuse(batch, channel_outer, y, x)
            schedule[output].reorder(outer, channel_inner)
            schedule[output].vectorize(channel_inner)
        else:
            x_outer, x_inner = schedule[output].split(x, factor=16)
            outer = schedule[output].fuse(batch, channel, y, x_outer)
            schedule[output].reorder(outer, x_inner)
            schedule[output].vectorize(x_inner)
    return tvm.build(schedule, [data, output], target=target, name="main"), tuple(output.shape)


def _numpy_stride2_reference(data, weight, bias):
    batch, _, height, width = data.shape
    _, out_channels, kh, kw = weight.shape
    result = np.empty((batch, out_channels, height * 2, width * 2), dtype="float32")
    for y in range(kh):
        for x in range(kw):
            result[:, :, y::2, x::2] = np.einsum(
                "nchw,co->nohw", data, weight[:, :, y, x], optimize=True
            )
    result += bias[None, :, None, None]
    return result


def _run_kernel(session, module_path, inputs, output_shape, expected, repeat):
    remote = session.load_module(session.upload(str(module_path), module_path.name))
    device = session.device
    device_inputs = [tvm.nd.array(value, device) for value in inputs]
    output = tvm.nd.empty(output_shape, "float32", device)
    remote["main"](*device_inputs, output)
    result = output.numpy()
    max_error = float(np.max(np.abs(result - expected)))
    np.testing.assert_allclose(result, expected, rtol=2e-3, atol=2e-3)
    times = remote.time_evaluator("main", device, number=1, repeat=repeat)(
        *device_inputs, output
    ).results
    return float(np.median(times) * 1e3), max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--tiles", default="4,8,16")
    parser.add_argument("--pixel-blocks", default="0", help="comma list; 0 = unblocked schedule")
    parser.add_argument("--skip-copies", action="store_true")
    parser.add_argument("--qf32", action="store_true", help="hand-written HVX qf32 kernel")
    parser.add_argument("--weight-major", action="store_true", help="parallelize over weight slices (qf32)")
    parser.add_argument("--serial", action="store_true", help="disable parallel launch (qf32)")
    parser.add_argument("--unrolls", default="4", help="comma list of rc unroll factors (qf32)")
    parser.add_argument("--llvm-options", help="extra LLVM options, e.g. -unroll-count=1")
    parser.add_argument("--arch", help="Hexagon arch for codegen (default v73), e.g. v68/v69")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--channel-only", action="store_true")
    parser.add_argument("--channel-inputs", choices=("both", "nchw", "nhwc"), default="both")
    args = parser.parse_args()
    _configure_linker()

    if args.model:
        _, _, _, _, workload, _ = _model_workloads(args.model, args.roi_batch)
        _, data_shape, weight_shape, stride, pads, output_padding = workload
    else:
        data_shape = (args.roi_batch, 256, 14, 14)
        weight_shape = (256, 256, 2, 2)
        stride, pads, output_padding = (2, 2), (0, 0, 0, 0), (0, 0)
    if stride != (2, 2) or pads not in ((0, 0), (0, 0, 0, 0)) or output_padding != (0, 0):
        raise ValueError("This specialization currently requires stride 2, zero padding, and no output padding")
    if args.llvm_options or args.arch:
        base = str(tvm.target.hexagon(args.arch or "v73"))
        target = tvm.target.Target(
            base + (f" -llvm-options={args.llvm_options}" if args.llvm_options else "")
        )
        target = tvm.target.Target(target, host=target)
    else:
        target = _hexagon_target()
    rng = np.random.default_rng(59)
    data = rng.normal(0, 0.1, data_shape).astype("float32")
    weight = rng.normal(0, 0.05, weight_shape).astype("float32")
    bias = rng.normal(0, 0.01, (weight_shape[1],)).astype("float32")
    inputs = [data, weight, bias]
    expected = _numpy_stride2_reference(data, weight, bias)
    tracker = Tracker(host="127.0.0.1", port=9197)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9197,
            "rpc_server_port": 7077,
            "workspace_base": "/data/local/tmp/tvm_hexagon_deconv_tune",
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        if args.channel_only:
            baseline_ms = 5892.6  # Last phone-measured generic baseline (ms).
        else:
            baseline, out_shape = _conv_transpose_module(
                data_shape, weight_shape, stride, pads, output_padding, target
            )
            assert out_shape == expected.shape
            baseline_path = Path("/tmp/tvm_hexagon_conv_transpose_topi.so")
            baseline.save(str(baseline_path))
            with launcher.create_session() as session:
                baseline_ms, baseline_error = _run_kernel(
                    session, baseline_path, inputs, out_shape, expected, args.repeat
                )
            print(
                f"topi_transpose: median={baseline_ms:.3f} ms, "
                f"max_abs_err={baseline_error:.3g}",
                flush=True,
            )

        macs = data_shape[0] * weight_shape[1] * data_shape[2] * data_shape[3]
        macs *= data_shape[1] * weight_shape[2] * weight_shape[3]
        if not args.channel_only:
            for tile in map(int, args.tiles.split(",")):
                module, out_shape = _direct_stride2_module(
                    data_shape, weight_shape, (weight_shape[1],), target, tile
                )
                path = Path("/tmp") / f"tvm_hexagon_conv_transpose_direct_w{tile}.so"
                module.save(str(path))
                with launcher.create_session() as session:
                    elapsed_ms, max_error = _run_kernel(
                        session, path, inputs, out_shape, expected, args.repeat
                    )
                print(
                    f"direct_parity width_tile={tile}: median={elapsed_ms:.3f} ms, "
                    f"speedup={baseline_ms / elapsed_ms:.2f}x, "
                    f"MACs={macs:,}, max_abs_err={max_error:.3g}",
                    flush=True,
                )

        packed_weight = np.ascontiguousarray(weight.transpose(2, 3, 0, 1))
        expected_nhwc = np.ascontiguousarray(expected.transpose(0, 2, 3, 1))
        input_layouts = {
            "both": (False, True),
            "nchw": (False,),
            "nhwc": (True,),
        }[args.channel_inputs]
        for tile, pixel_block, unroll in itertools.product(
            map(int, args.tiles.split(",")),
            map(int, args.pixel_blocks.split(",")),
            map(int, args.unrolls.split(",")) if args.qf32 else (0,),
        ):
            for input_nhwc in input_layouts:
                if args.qf32:
                    if not input_nhwc or not pixel_block:
                        continue
                    module, out_shape = _qf32_module(
                        data_shape, weight_shape, target, tile // 32, pixel_block, unroll,
                        not args.serial, args.weight_major,
                    )
                else:
                    module, out_shape = _channel_vectorized_module(
                        data_shape, weight_shape, (weight_shape[1],), target, tile, input_nhwc,
                        pixel_block,
                    )
                input_data = (
                    np.ascontiguousarray(data.transpose(0, 2, 3, 1)) if input_nhwc else data
                )
                path = Path("/tmp") / f"tvm_hexagon_conv_transpose_nhwc_c{tile}_p{pixel_block}_u{unroll}_in{'nhwc' if input_nhwc else 'nchw'}.so"
                module.save(str(path))
                with launcher.create_session() as session:
                    elapsed_ms, max_error = _run_kernel(
                        session,
                        path,
                        [input_data, packed_weight, bias],
                        out_shape,
                        expected_nhwc,
                        args.repeat,
                    )
                print(
                    f"{'qf32' if args.qf32 else 'channel_vectorized_nhwc'} channel_tile={tile} pixel_block={pixel_block} unroll={unroll} input={'NHWC' if input_nhwc else 'NCHW'}: "
                    f"median={elapsed_ms:.3f} ms, speedup={baseline_ms / elapsed_ms:.2f}x, "
                    f"max_abs_err={max_error:.3g} (weight packing and layout conversion excluded)",
                    flush=True,
                )

        if args.skip_copies:
            return
        copy_specs = [
            ("input_nchw_to_nhwc", data_shape, data,
             np.ascontiguousarray(data.transpose(0, 2, 3, 1)), True),
            ("output_nhwc_to_nchw", expected.shape, expected_nhwc,
             np.ascontiguousarray(expected), False),
        ]
        for label, logical_shape, source, expected_copy, to_nhwc in copy_specs:
            variants = (None, 16, 32, 64, 128, 256) if label == "output_nhwc_to_nchw" else (None,)
            for channel_tile in variants:
                module, copy_shape = _layout_copy_module(
                    logical_shape, target, to_nhwc, channel_tile
                )
                suffix = f"_c{channel_tile}" if channel_tile else ""
                path = Path("/tmp") / f"tvm_hexagon_{label}{suffix}.so"
                module.save(str(path))
                with launcher.create_session() as session:
                    elapsed_ms, max_error = _run_kernel(
                        session,
                        path,
                        [np.ascontiguousarray(source)],
                        copy_shape,
                        expected_copy,
                        args.repeat,
                    )
                print(
                    f"layout_copy {label}{suffix}: median={elapsed_ms:.3f} ms, "
                    f"max_abs_err={max_error:.3g}",
                    flush=True,
                )
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
