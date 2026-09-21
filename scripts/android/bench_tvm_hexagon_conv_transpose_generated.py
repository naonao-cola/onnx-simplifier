#!/usr/bin/env python3
"""Mask-head ConvTranspose on Hexagon from *plain TE schedules* -- no hand-written intrinsics.

The kernels here are ordinary TE computes with a cache_write-style accumulator, split/reorder/
unroll/vectorize. `hexagon_qfloat.build` applies a TIR pass that turns the vectorized
multiply-accumulate chains into HVX qfloat intrinsic chains, so TVM generates what
`bench_tvm_hexagon_conv_transpose.py` / `..._fp16.py` write by hand.

Modes (`--no-pass` builds the same schedules with stock TVM for comparison):

* `f32`   - fp32 data/weights/output, fp32 accumulate (qf32 chain).
* `f16w`  - fp16 data/weights/output, fp32 accumulate via widening hf x hf -> qf32.
* `f16k`  - fp16 data/weights/output, qf16 partial sums of `--chunk` input channels widened
            into an fp32 total (the schedule expresses the two-level reduction).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import setuptools  # noqa: F401  # TVM 0.17 imports distutils during module initialization.
import tvm
from bench_tvm_hexagon_conv_transpose import (
    _configure_linker,
    _hexagon_target,
    _numpy_stride2_reference,
)
from hexagon_qfloat import build
from tvm import te
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker


def _placeholders(shape_info, dtype):
    n, ic, h, w, oc = shape_info
    data = te.placeholder((n, h, w, ic), name="data", dtype=dtype)
    weight = te.placeholder((2, 2, ic, oc), name="weight", dtype=dtype)
    bias = te.placeholder((oc,), name="bias", dtype=dtype)
    return data, weight, bias


def conv_transpose_module(mode, shape_info, target, vectors, pixel_block, unroll, chunk, use_pass):
    """Plain TE 2x2/stride-2 ConvTranspose (NHWC, packed weights) for `mode`."""
    n, ic, h, w, oc = shape_info
    fp16 = mode != "f32"
    lanes = 64 if fp16 else 32
    dtype = "float16" if fp16 else "float32"
    data, weight, bias = _placeholders(shape_info, dtype)
    acc_dtype = "float32" if mode in ("f32", "f16w", "f16k") else dtype

    shape = (n, h * 2, w * 2, oc)
    plane = (n, h, w, 2, 2, oc)  # [batch, row, col, kernel row, kernel col, out channel]

    def term(idx, r):
        b, y, x, py, px, c = idx
        if fp16 and mode == "f16w":  # widen the operands, not the product
            return data[b, y, x, r].astype(acc_dtype) * weight[py, px, r, c].astype(acc_dtype)
        return data[b, y, x, r] * weight[py, px, r, c]

    if mode == "f16k":
        # Two-level reduction: qf16 partial sums over `chunk` channels, fp32 total.
        outer = te.reduce_axis((0, ic // chunk), name="rco")
        inner = te.reduce_axis((0, chunk), name="rci")
        partial = te.compute(
            (*plane, ic // chunk),
            lambda *i: te.sum(term(i[:6], i[6] * chunk + inner), axis=inner),
            name="partial",
        )
        acc = te.compute(
            plane,
            lambda *i: te.sum(partial[(*i, outer)].astype("float32"), axis=outer),
            name="acc",
        )
    else:
        reduce = te.reduce_axis((0, ic), name="rc")
        acc = te.compute(plane, lambda *i: te.sum(term(i, reduce), axis=reduce), name="acc")
    # Interleave the four parity planes into the NHWC output (bias added here).
    out = te.compute(
        (n, h * 2, w * 2, oc),
        lambda b, y, x, c: (
            acc[b, y // 2, x // 2, y % 2, x % 2, c] + bias[c].astype(acc_dtype)
        ).astype(dtype),
        name="conv_transpose",
    )

    s = te.create_schedule(out.op)
    b_ax, y_ax, x_ax, c_ax = s[out].op.axis
    yo, yp = s[out].split(y_ax, factor=2)
    xo, xp = s[out].split(x_ax, factor=2)
    xoo, xoi = s[out].split(xo, factor=pixel_block)
    co, ci = s[out].split(c_ax, factor=lanes * vectors)
    # Both x parities sit inside the accumulator tile, so one splatted activation feeds both.
    s[out].reorder(b_ax, yo, yp, xoo, co, xoi, xp, ci)
    s[out].parallel(s[out].fuse(b_ax, yo))
    ci_o, ci_i = s[out].split(ci, factor=lanes)
    s[out].unroll(xoi)
    s[out].unroll(xp)
    s[out].unroll(ci_o)
    s[out].vectorize(ci_i)
    s[acc].compute_at(s[out], co)

    ab, ay, ax, apy, apx, ac = s[acc].op.axis
    aco, aci = s[acc].split(ac, factor=lanes)
    spatial = [ab, ay, ax, apy, apx, aco, aci]
    if mode == "f16k":
        (rco,) = s[acc].op.reduce_axis
        s[acc].reorder(rco, *spatial)
        s[partial].compute_at(s[acc], rco)
        p_axes = list(s[partial].op.axis)
        (rci,) = s[partial].op.reduce_axis
        pco, pci = s[partial].split(p_axes[5], factor=lanes)
        s[partial].reorder(p_axes[6], rci, *p_axes[:5], pco, pci)
        for axis in (rci, p_axes[2], p_axes[4], pco):
            s[partial].unroll(axis)
        s[partial].vectorize(pci)
    else:
        (rc,) = s[acc].op.reduce_axis
        rco, rci = s[acc].split(rc, factor=unroll)
        s[acc].reorder(rco, rci, *spatial)
        s[acc].unroll(rci)
    for axis in (ax, apx, aco):
        s[acc].unroll(axis)
    s[acc].vectorize(aci)

    module = build(s, [data, weight, bias, out], target, enable_pass=use_pass)
    return module, shape


def _reference(shape_info, rng, fp16):
    n, ic, size, _, oc = shape_info
    data = rng.normal(0, 0.1, (n, ic, size, size)).astype("float32")
    weight = rng.normal(0, 0.05, (ic, oc, 2, 2)).astype("float32")
    bias = rng.normal(0, 0.01, (oc,)).astype("float32")
    dtype = "float16" if fp16 else "float32"
    d, w_, b = (a.astype(dtype) for a in (data, weight, bias))
    expected = _numpy_stride2_reference(
        d.astype("float32"), w_.astype("float32"), b.astype("float32")
    ).transpose(0, 2, 3, 1)
    inputs = [
        np.ascontiguousarray(d.transpose(0, 2, 3, 1)),
        np.ascontiguousarray(w_.transpose(2, 3, 0, 1)),
        b,
    ]
    return inputs, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--modes", default="f32,f16w,f16k")
    parser.add_argument("--vectors", default="2", help="HVX vectors of output channels per tile")
    parser.add_argument("--pixel-blocks", default="2")
    parser.add_argument("--unrolls", default="4")
    parser.add_argument("--chunks", default="8", help="f16k: input channels per qf16 partial sum")
    parser.add_argument("--no-pass", action="store_true", help="stock TVM (no qfloat pass)")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _configure_linker()

    shape_info = (args.roi_batch, 256, 14, 14, 256)
    target = _hexagon_target()
    tracker = Tracker(host="127.0.0.1", port=9197)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9197,
            "rpc_server_port": 7077,
            "workspace_base": "/data/local/tmp/tvm_hexagon_deconv_generated",
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        for mode in args.modes.split(","):
            inputs, expected = _reference(shape_info, np.random.default_rng(59), mode != "f32")
            scale = float(np.max(np.abs(expected)))
            out_dtype = "float32" if mode == "f32" else "float16"
            chunks = map(int, args.chunks.split(",")) if mode == "f16k" else (0,)
            for vectors, pixel_block, unroll, chunk in itertools.product(
                map(int, args.vectors.split(",")),
                map(int, args.pixel_blocks.split(",")),
                map(int, args.unrolls.split(",")),
                chunks,
            ):
                label = (
                    f"{mode}{' (stock TVM)' if args.no_pass else ''} vec={vectors} "
                    f"pix={pixel_block} unroll={unroll}" + (f" chunk={chunk}" if chunk else "")
                )
                try:
                    module, shape = conv_transpose_module(
                        mode, shape_info, target, vectors, pixel_block, unroll, chunk, not args.no_pass
                    )
                    path = Path("/tmp") / f"tvm_hexagon_deconv_gen_{mode}_{vectors}_{pixel_block}_{unroll}_{chunk}.so"
                    module.save(str(path))
                    with launcher.create_session() as session:
                        remote = session.load_module(session.upload(str(path), path.name))
                        device = session.device
                        nd_in = [tvm.nd.array(a, device) for a in inputs]
                        output = tvm.nd.empty(shape, out_dtype, device)
                        remote["main"](*nd_in, output)
                        error = float(np.max(np.abs(output.numpy().astype("float32") - expected))) / scale
                        times = remote.time_evaluator("main", device, number=1, repeat=args.repeat)(
                            *nd_in, output
                        ).results
                    verdict = "" if error < 0.05 else "  ** WRONG RESULT **"
                    print(
                        f"{label}: median={np.median(times) * 1e3:.3f} ms, "
                        f"max_err/scale={error:.2e}{verdict}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - keep sweeping
                    print(f"{label}: FAILED ({type(exc).__name__}: {str(exc).splitlines()[-1][:160]})", flush=True)
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
