#!/usr/bin/env python3
"""Benchmark tinygrad's code generation through an onnxsim RPC server.

The server (started on the machine whose hardware you want to measure, with tinygrad installed)
runs each ONNX workload two ways -- with onnxruntime and with tinygrad on a chosen device and
codegen setting -- and this client compares them: correctness against onnxruntime, device-side
time per call, achieved GFLOP/s, and tinygrad's own kernel statistics (kernel count, kernel
time, GB moved).

    # on the target machine (tinygrad + onnxruntime installed; clang on PATH for the CPU device)
    python -m onnxsim.rpc server --host 127.0.0.1 --port 9191 --key bench
    # anywhere that can reach it
    python bench_codegen_rpc.py --port 9191 --key bench --device NV --beam 0,2

`--dsp-server HOST:PORT` benchmarks tinygrad's Hexagon renderer on a server started with
`MOCKDSP=1` (kernels run under qemu-hexagon-static): the "time" there is an instruction count,
reported as instructions per MAC instead of milliseconds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim.rpc as rpc


@dataclass
class Workload:
    name: str
    model: onnx.ModelProto
    inputs: Dict[str, np.ndarray]
    flops: float  # multiply-adds count as 2


def _build(body: str, initializers: Dict[str, np.ndarray]) -> onnx.ModelProto:
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["" : 13]> {body}')
    model.graph.initializer.extend(
        onnx.numpy_helper.from_array(v, k) for k, v in initializers.items()
    )
    return model


def conv(name, cin, cout, k, hw, stride=1, pad=None, batch=1) -> Workload:
    rng = np.random.default_rng(hash(name) % 2**32)
    pad = k // 2 if pad is None else pad
    out_hw = (hw + 2 * pad - k) // stride + 1
    body = (
        f"g (float[{batch},{cin},{hw},{hw}] x) => (float[{batch},{cout},{out_hw},{out_hw}] y) "
        f"{{ c = Conv<kernel_shape=[{k},{k}], strides=[{stride},{stride}], pads=[{pad},{pad},{pad},{pad}]>(x, w, b) y = Relu(c) }}"
    )
    weights = {
        "w": (rng.normal(size=(cout, cin, k, k)) * 0.05).astype("float32"),
        "b": (rng.normal(size=(cout,)) * 0.01).astype("float32"),
    }
    return Workload(
        name,
        _build(body, weights),
        {"x": rng.normal(size=(batch, cin, hw, hw)).astype("float32")},
        2.0 * batch * cout * cin * k * k * out_hw * out_hw,
    )


def matmul(name, n) -> Workload:
    rng = np.random.default_rng(n)
    body = f"g (float[{n},{n}] x) => (float[{n},{n}] y) {{ m = MatMul(x, w) a = Add(m, b) y = Relu(a) }}"
    weights = {
        "w": (rng.normal(size=(n, n)) * 0.05).astype("float32"),
        "b": rng.normal(size=(n,)).astype("float32"),
    }
    return Workload(
        name,
        _build(body, weights),
        {"x": rng.normal(size=(n, n)).astype("float32")},
        2.0 * n**3,
    )


def add_relu(name, c, hw) -> Workload:
    rng = np.random.default_rng(c)
    body = f"g (float[1,{c},{hw},{hw}] x, float[1,{c},{hw},{hw}] y) => (float[1,{c},{hw},{hw}] z) {{ s = Add(x, y) z = Relu(s) }}"
    inputs = {k: rng.normal(size=(1, c, hw, hw)).astype("float32") for k in ("x", "y")}
    return Workload(
        name, _build(body, {}), inputs, float(c * hw * hw)
    )  # 1 add per element


# ResNet-50 / Mask R-CNN backbone layer shapes.
GPU_WORKLOADS = [
    conv("stem_7x7_s2_3to64@224", 3, 64, 7, 224, stride=2),
    conv("conv3x3_64to64@56", 64, 64, 3, 56),
    conv("conv1x1_256to64@56", 256, 64, 1, 56),
    conv("conv3x3_256to256@14", 256, 256, 3, 14),
    matmul("matmul_1024", 1024),
    add_relu("add_relu_256@56", 256, 56),
]
DSP_WORKLOADS = [  # QEMU is slow: small shapes
    conv("conv1x1_64to64@28", 64, 64, 1, 28),
    conv("conv3x3_64to64@14", 64, 64, 3, 14),
    matmul("matmul_128", 128),
]


def bench(session, workload, runtime_spec, number, repeat):
    kwargs, label = runtime_spec
    model = session.load_model(workload.model, **kwargs)
    try:
        outputs = model.run(workload.inputs)
        timing = model.time_evaluator(workload.inputs, number=number, repeat=repeat)
    finally:
        model.close()
    return label, outputs, timing


def main() -> None:
    parser_ = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser_.add_argument("--host", default="127.0.0.1")
    parser_.add_argument("--port", type=int, default=9191)
    parser_.add_argument("--key", default="")
    parser_.add_argument(
        "--device",
        default=None,
        help="tinygrad device on the server (NV, CUDA, CPU, ...)",
    )
    parser_.add_argument(
        "--beam", default="0", help="comma list of BEAM widths; 0 = default heuristics"
    )
    parser_.add_argument("--number", type=int, default=5)
    parser_.add_argument("--repeat", type=int, default=3)
    parser_.add_argument(
        "--no-ort", action="store_true", help="skip the onnxruntime baseline"
    )
    parser_.add_argument("--dsp-server", default=None, metavar="HOST:PORT")
    parser_.add_argument("--dsp-key", default="")
    parser_.add_argument(
        "--only", default=None, help="substring filter for workload names"
    )
    parser_.add_argument("--json", default=None, help="write raw results here")
    args = parser_.parse_args()

    rows: List[dict] = []
    session = rpc.connect(args.host, args.port, key=args.key)
    print(
        f"server: {session.info['platform']} onnxruntime={session.info['onnxruntime']} tinygrad={session.info['tinygrad']}"
    )
    specs = []
    if not args.no_ort:
        specs.append(({"runtime": "onnxruntime"}, "onnxruntime"))
    for beam in (int(b) for b in args.beam.split(",")):
        options = {"BEAM": beam} if beam else {}
        specs.append(
            (
                {"runtime": "tinygrad", "device": args.device, "options": options},
                f"tinygrad {args.device or 'default'}"
                + (f" BEAM={beam}" if beam else ""),
            )
        )

    for workload in GPU_WORKLOADS:
        if args.only and args.only not in workload.name:
            continue
        reference = None
        for spec in specs:
            try:
                label, outputs, timing = bench(
                    session, workload, spec, args.number, args.repeat
                )
            except rpc.RPCError as error:
                print(f"{workload.name:24} {spec[1]:28} FAILED: {error}")
                continue
            out = next(iter(outputs.values()))
            if reference is None:
                reference = out
            err = float(np.abs(out - reference).max())
            gflops = workload.flops / timing.median / 1e9
            row = {
                "workload": workload.name,
                "runtime": label,
                "median_ms": timing.median * 1e3,
                "gflops": gflops,
                "max_abs_err_vs_first": err,
                "stats": timing.stats,
            }
            rows.append(row)
            stats = timing.stats or {}
            extra = (
                f"kernels={stats.get('kernels')} kernel_time={stats.get('kernel_time_s', 0) * 1e3:.3f}ms"
                if stats
                else ""
            )
            print(
                f"{workload.name:24} {label:28} {timing.median * 1e3:9.3f} ms {gflops:9.1f} GFLOP/s err={err:.1e} {extra}",
                flush=True,
            )
    session.close()

    if args.dsp_server:
        host, _, port = args.dsp_server.rpartition(":")
        dsp = rpc.connect(host, int(port), key=args.dsp_key)
        print(
            "\ntinygrad Hexagon renderer under qemu-hexagon-static (mock DSP; instruction counts):"
        )
        for workload in DSP_WORKLOADS:
            model = dsp.load_model(workload.model, runtime="tinygrad", device="DSP")
            outputs = model.run(workload.inputs)
            stats = model.time_evaluator(workload.inputs, number=1, repeat=1).stats
            model.close()
            instructions = (
                stats["kernel_time_s"] * 1e9
            )  # the mock reports inscount / 1e9
            macs = workload.flops / 2
            rows.append(
                {
                    "workload": workload.name,
                    "runtime": "tinygrad DSP (mock)",
                    "instructions": instructions,
                    "insn_per_mac": instructions / macs,
                    "stats": stats,
                }
            )
            print(
                f"{workload.name:24} kernels={stats['kernels']} insns={instructions / 1e6:9.1f}M insn/MAC={instructions / macs:6.2f}",
                flush=True,
            )
        dsp.close()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
