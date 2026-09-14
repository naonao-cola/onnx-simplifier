#!/usr/bin/env python3
"""Roofline benchmark for AMD XDNA2 NPUs (Ryzen AI 300 / AI Max 300).

Builds small, compute-bound INT8 toy models -- uniform Conv+Relu and
MatMul+Add towers with static shapes -- quantizes them, and reports both the
hardware-agnostic theory (``onnxsim.model_info``: FLOPs, traffic, arithmetic
intensity, ideal latency at peak TOPS) and, where the hardware exists,
measured latency and achieved TOPS.

The towers are deliberately NPU-shaped: one fused conv/matmul region, no
control flow, no post-processing, explicit Conv attributes (see
``onnxsim.legalize_for_vitisai``). They are what an NPU *should* be good at,
so the achieved-vs-peak gap measures the stack, not the model.

Subcommands::

    python bench/amd_npu_roofline.py theory [--quantizer onnxsim|ort]
    python bench/amd_npu_roofline.py prep [--quantizer onnxsim|ort] [--out DIR]
    python bench/amd_npu_roofline.py run <model> [--provider cpu|vitisai]
        [--runs N] [--cache-dir DIR] [--cache-key KEY]

``theory`` and ``prep`` need this repo's ``onnxsim`` (quantizer, ModelInfo,
VitisAI legalizer). ``run`` needs only ``onnx``/``onnxruntime``/``numpy`` --
run it inside the Ryzen AI venv for ``--provider vitisai`` (source the venv's
``bin/activate`` and ``/opt/xilinx/xrt/setup.sh`` first; the provider needs
both). Measured Strix Halo numbers live in
``bench/RESULTS_amd_npu_roofline.md``.

Environment: ``onnx``, ``onnxruntime``; ``onnxsim`` from this checkout for
anything but ``run``.
"""

import argparse
import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

# Peak INT8 throughput of the XDNA2 NPU (Strix Point / Strix Halo).
PEAK_TOPS = 50.0e12

# (tag, kind, params): conv = (channels, spatial, blocks), mm = (M, K, N, blocks).
SHAPES = [
    ("conv_base", "conv", (256, 56, 16)),
    ("conv_wide", "conv", (512, 28, 16)),
    ("conv_big", "conv", (512, 56, 12)),
    ("mm_2048", "mm", (2048, 2048, 2048, 8)),
    ("mm_1024", "mm", (1024, 1024, 1024, 32)),
    ("mm_4096", "mm", (4096, 4096, 4096, 1)),
    ("mm_skinny", "mm", (256, 4096, 4096, 8)),
]


def build_tower(tag, kind, params, seed=0):
    """A uniform tower model with random weights; shapes carry static dims
    and Conv nodes carry explicit attributes (what the NPU needs)."""
    rng = np.random.default_rng(seed)
    nodes, inits, prev = [], [], "x"
    if kind == "conv":
        c, hw, blocks = params
        for b in range(blocks):
            w = (rng.standard_normal((c, c, 3, 3)) * 0.05).astype(np.float32)
            inits += [
                onnx.numpy_helper.from_array(w, f"W{b}"),
                onnx.numpy_helper.from_array(np.zeros(c, dtype=np.float32), f"B{b}"),
            ]
            nodes.append(
                helper.make_node(
                    "Conv",
                    [prev, f"W{b}", f"B{b}"],
                    [f"c{b}"],
                    name=f"Conv{b}",
                    kernel_shape=[3, 3],
                    strides=[1, 1],
                    pads=[1, 1, 1, 1],
                    dilations=[1, 1],
                    group=1,
                )
            )
            nodes.append(
                helper.make_node("Relu", [f"c{b}"], [f"r{b}"], name=f"Relu{b}")
            )
            prev = f"r{b}"
        io_shape = [1, c, hw, hw]
    else:
        m, k, n, blocks = params
        for b in range(blocks):
            w = (rng.standard_normal((k, n)) * 0.01).astype(np.float32)
            bias = (rng.standard_normal(n) * 0.01).astype(np.float32)
            inits += [
                onnx.numpy_helper.from_array(w, f"W{b}"),
                onnx.numpy_helper.from_array(bias, f"B{b}"),
            ]
            nodes.append(
                helper.make_node("MatMul", [prev, f"W{b}"], [f"m{b}"], name=f"MM{b}")
            )
            nodes.append(
                helper.make_node("Add", [f"m{b}", f"B{b}"], [f"a{b}"], name=f"Add{b}")
            )
            prev = f"a{b}"
        io_shape = None
        in_shape, out_shape = [m, k], [m, n]
    if kind == "conv":
        in_shape = out_shape = io_shape
    model = helper.make_model(
        helper.make_graph(
            nodes,
            tag,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
            [helper.make_tensor_value_info(prev, TensorProto.FLOAT, out_shape)],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
    )
    # Intermediates need value_info or onnxsim's calibrator sees only the
    # graph input.
    return onnx.shape_inference.infer_shapes(model)


def _random_calib(input_name, shape, n=4, seed=7):
    rng = np.random.default_rng(seed)
    return [
        {input_name: rng.standard_normal(shape).astype(np.float32)} for _ in range(n)
    ]


def quantize(model, quantizer):
    """INT8 QDQ quantization + VitisAI legalization. ``onnxsim`` (default)
    is this repo's own quantizer; ``ort`` is ONNX Runtime's, kept as a
    reference -- on VitisAI EP 1.27 the two differ in tail handling (see the
    RESULTS file)."""
    import onnxsim

    input_name = model.graph.input[0].name
    dims = [
        d.dim_value if d.HasField("dim_value") else 1
        for d in model.graph.input[0].type.tensor_type.shape.dim
    ]
    if quantizer == "onnxsim":
        q = onnxsim.quantize_static(
            model, calibration_data=_random_calib(input_name, dims)
        )
    else:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        class _Reader(CalibrationDataReader):
            def __init__(self):
                self.batches = _random_calib(input_name, dims)
                self.i = 0

            def get_next(self):
                if self.i >= len(self.batches):
                    return None
                self.i += 1
                return self.batches[self.i - 1]

            def rewind(self):
                self.i = 0

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.onnx")
            dst = os.path.join(tmp, "dst.onnx")
            onnx.save(model, src)
            quantize_static(
                src,
                dst,
                _Reader(),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
            )
            q = onnx.load(dst)
    print("vitisai support:", onnxsim.check_vitisai_support(q), flush=True)
    return onnxsim.legalize_for_vitisai(q)


def theory_row(tag, kind, params, quantizer):
    from onnxsim.model_info import ModelInfo

    model = quantize(build_tower(tag, kind, params), quantizer)
    info = ModelInfo(model)
    flops = int(info.flops)
    mem = int(info.mem_access)
    return {
        "tag": f"{tag}[{quantizer}]",
        "gflop": flops / 1e9,
        "gbytes": mem / 1e9,
        "intensity": flops / mem,
        "ideal_ms": flops / PEAK_TOPS * 1e3,
    }


def cmd_theory(args):
    rows = [
        theory_row(tag, kind, params, args.quantizer) for tag, kind, params in SHAPES
    ]
    print(
        f"{'model':<22}{'GFLOP':>9}{'GB':>9}{'FLOP/B':>9}{'ideal@50T':>11}", flush=True
    )
    for r in rows:
        print(
            f"{r['tag']:<22}{r['gflop']:>9.1f}{r['gbytes']:>9.2f}"
            f"{r['intensity']:>9.1f}{r['ideal_ms']:>10.2f}ms",
            flush=True,
        )


def cmd_prep(args):
    os.makedirs(args.out, exist_ok=True)
    for tag, kind, params in SHAPES:
        model = quantize(build_tower(tag, kind, params), args.quantizer)
        path = os.path.join(args.out, f"{tag}_{args.quantizer}.onnx")
        onnx.save(model, path)
        print(f"wrote {path}", flush=True)


def _random_feed(sess):
    import onnxruntime as ort  # noqa: F401  (ensures the dependency reads clearly)

    feed = {}
    rng = np.random.default_rng(0)
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        dtype = {
            "tensor(float)": np.float32,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
            "tensor(int64)": np.int64,
        }.get(i.type, np.float32)
        feed[i.name] = (
            rng.standard_normal(shape).astype(dtype)
            if np.issubdtype(dtype, np.floating)
            else np.zeros(shape, dtype=dtype)
        )
    return feed


def cmd_run(args):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.time()
    if args.provider == "vitisai":
        sess = ort.InferenceSession(
            args.model,
            sess_options=opts,
            providers=[
                (
                    "VitisAIExecutionProvider",
                    {"cacheDir": args.cache_dir, "cacheKey": args.cache_key},
                ),
                "CPUExecutionProvider",
            ],
        )
    else:
        sess = ort.InferenceSession(
            args.model, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    print(
        f"session creation: {time.time() - t0:.1f}s providers: {sess.get_providers()}",
        flush=True,
    )
    feed = _random_feed(sess)
    for i in range(args.warmup):
        t0 = time.time()
        out = sess.run(None, feed)
        print(
            f"warmup {i}: {(time.time() - t0) * 1000:.0f} ms "
            f"out={[np.asarray(o).shape for o in out]}",
            flush=True,
        )
    ts = []
    for i in range(args.runs):
        t0 = time.time()
        sess.run(None, feed)
        dt = (time.time() - t0) * 1000
        ts.append(dt)
        print(f"run {i}: {dt:.1f} ms", flush=True)
    a = np.array(ts)
    try:
        flops = int(
            __import__("onnxsim.model_info", fromlist=["ModelInfo"])
            .ModelInfo(onnx.load(args.model, load_external_data=False))
            .flops
        )
        tops = flops / (a.mean() / 1e3) / 1e12
        tops_s = f" achieved={tops:.1f} TOPS ({tops / (PEAK_TOPS / 1e12) * 100:.0f}% of peak)"
    except Exception:
        tops_s = ""
    print(
        f"RESULT mean={a.mean():.1f}ms median={np.median(a):.1f}ms "
        f"p90={np.percentile(a, 90):.1f}ms fps={1000 / a.mean():.1f}{tops_s}",
        flush=True,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("theory", help="print the theoretical roofline table")
    t.add_argument("--quantizer", choices=["onnxsim", "ort"], default="onnxsim")
    pr = sub.add_parser("prep", help="write quantized tower models to OUT")
    pr.add_argument("--quantizer", choices=["onnxsim", "ort"], default="onnxsim")
    pr.add_argument("--out", default="/tmp/amd_npu_roofline")
    r = sub.add_parser("run", help="time one model on an execution provider")
    r.add_argument("model")
    r.add_argument("--provider", choices=["cpu", "vitisai"], default="cpu")
    r.add_argument("--runs", type=int, default=10)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--cache-dir", default="/tmp/vaip_cache_roofline")
    r.add_argument("--cache-key", default="roofline")
    args = p.parse_args(argv)
    if args.cmd == "theory":
        cmd_theory(args)
    elif args.cmd == "prep":
        cmd_prep(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
