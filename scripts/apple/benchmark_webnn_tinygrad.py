#!/usr/bin/env python3
"""Per-node and whole-model timing of rustnn's native WebNN (ONNX Runtime CPU
and Core ML) against tinygrad (CPU and Metal, with and without BEAM search)
on a Mac, through ``onnxsim.webnn_tinygrad_tuning``.

This is the entry point the ``rustnn-webnn`` job in
``.github/workflows/apple-integration.yml`` calls. It builds a few small
models, simplifies them with onnxsim, and prints a Markdown report. The job
appends that report to ``$GITHUB_STEP_SUMMARY``.

Device mapping in pywebnn 0.5.12:

- ``cpu`` is ONNX Runtime's CPU execution provider.
- ``npu`` is Core ML, with ``MLComputeUnits.cpuAndNeuralEngine`` and a
  fallback to ``.all``. Core ML decides per op where it actually runs, and a
  virtualized runner may not expose a Neural Engine at all. So an ``npu``
  row means "went through Core ML", not "ran on the ANE".
- ``gpu`` is *not* a GPU on macOS, because rustnn only registers ONNX
  Runtime's CPU EP. The GPU is covered by tinygrad's ``METAL`` device.

Each backend's outputs are checked against onnx's reference evaluator.
Results within ``--atol``/``--rtol`` (loose by default, since Core ML
computes in float16 on the ANE/GPU) can win.

Exit status is non-zero when a device named in ``--require`` produced no
valid timing for a model: it was unavailable, it errored, or it was out of
tolerance. Ops that WebNN can't lower are reported and don't fail the run.

Usage:
    benchmark_webnn_tinygrad.py
    benchmark_webnn_tinygrad.py --require npu,METAL --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import webnn_tinygrad_tuning as tuning


def _weight(rng, name, *shape):
    return numpy_helper.from_array(
        (rng.standard_normal(shape) * 0.1).astype(np.float32), name
    )


def _name_nodes(model):
    for i, node in enumerate(model.graph.node):
        node.name = node.name or f"{node.op_type}_{i}"
    return model


def build_models():
    rng = np.random.default_rng(0)
    cnn = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        cnn (float[1,3,128,128] x) => (float[1,10] y)
        {
          c1 = Conv<pads = [1, 1, 1, 1], strides = [2, 2]>(x, w1, b1)
          r1 = Relu(c1)
          c2 = Conv<pads = [1, 1, 1, 1]>(r1, w2, b2)
          r2 = Relu(c2)
          c3 = Conv<pads = [1, 1, 1, 1], group = 64>(r2, w3, b3)
          r3 = Relu(c3)
          p = GlobalAveragePool(r3)
          f = Flatten(p)
          y = Gemm<transB = 1>(f, w4, b4)
        }
        """
    )
    cnn.graph.initializer.extend(
        [
            _weight(rng, "w1", 32, 3, 3, 3),
            _weight(rng, "b1", 32),
            _weight(rng, "w2", 64, 32, 3, 3),
            _weight(rng, "b2", 64),
            _weight(rng, "w3", 64, 1, 3, 3),
            _weight(rng, "b3", 64),
            _weight(rng, "w4", 10, 64),
            _weight(rng, "b4", 10),
        ]
    )
    mlp = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        mlp (float[16,512] x) => (float[16,512] y)
        {
          h = MatMul(x, w1)
          hb = Add(h, b1)
          a = Relu(hb)
          o = MatMul(a, w2)
          ob = Add(o, b2)
          y = Softmax<axis = -1>(ob)
        }
        """
    )
    mlp.graph.initializer.extend(
        [
            _weight(rng, "w1", 512, 2048),
            _weight(rng, "b1", 2048),
            _weight(rng, "w2", 2048, 512),
            _weight(rng, "b2", 512),
        ]
    )
    return {"small_cnn": cnn, "mlp": mlp}


def _fmt(t):
    if not t.ok:
        return f"— ({t.error.splitlines()[0][:80]})"
    err = "" if t.max_abs_error is None else f" (err {t.max_abs_error:.1e})"
    return f"{t.median_ms:.3f} ms{err}"


def _table(results):
    configs = []
    for r in results:
        for t in r.timings:
            if t.config not in configs:
                configs.append(t.config)
    lines = [
        "| node | op | " + " | ".join(f"`{c}`" for c in configs) + " | winner |",
        "|---|---|" + "---|" * len(configs) + "---|",
    ]
    for r in results:
        by_config = {t.config: t for t in r.timings}
        cells = [_fmt(by_config[c]) if c in by_config else "" for c in configs]
        winner = f"`{r.winner.config}`" if r.winner else "none"
        lines.append(
            f"| {r.node_name or '(whole model)'} | {r.op_type} | "
            + " | ".join(cells)
            + f" | {winner} |"
        )
    return "\n".join(lines)


def _device_of(t):
    # "npu/coreml" -> "npu"; "METAL BEAM=2" -> "METAL"
    return t.config.split("/")[0] if t.backend == "webnn" else t.config.split(" ")[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--webnn-device-types", default="cpu,npu")
    ap.add_argument("--tinygrad-devices", default="CPU,METAL")
    ap.add_argument("--beams", default="0,2")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument(
        "--require",
        default="",
        help="comma-separated WebNN device types / tinygrad devices that must "
        "produce a valid timing for every model (e.g. npu,METAL)",
    )
    ap.add_argument("--output", help="write all results as JSON here")
    args = ap.parse_args(argv)

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    kwargs = dict(
        webnn_device_types=split(args.webnn_device_types),
        tinygrad_devices=split(args.tinygrad_devices),
        beams=[int(b) for b in split(args.beams)],
        runs=args.runs,
        atol=args.atol,
        rtol=args.rtol,
    )
    required = split(args.require)

    lines = ["## rustnn (WebNN) vs. tinygrad", ""]
    probe = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        probe (float[2,2] x) => (float[2,2] y) { y = Relu(x) }
        """
    )
    for device_type in kwargs["webnn_device_types"]:
        ok, reason = onnxsim.probe_rustnn(device_type)
        if ok:
            info = onnxsim.RustnnSession(probe, device_type=device_type).backend_info()
            lines.append(f"- WebNN `{device_type}`: backend_info={info}")
        else:
            lines.append(f"- WebNN `{device_type}`: unavailable ({reason})")
    lines.append("")

    failures, dump = [], {}
    for name, model in build_models().items():
        simplified, ok = onnxsim.simplify(model)
        assert ok, f"{name}: simplified model failed onnxsim's own check"
        _name_nodes(simplified)
        per_node = tuning.tune_model(simplified, **kwargs)
        whole = tuning.benchmark_model(simplified, **kwargs)
        results = per_node + [whole]
        unsupported = onnxsim.find_unsupported_webnn_ops(simplified)
        lines += [f"### `{name}`", ""]
        if unsupported:
            lines += [f"Not lowerable to WebNN: {unsupported}", ""]
        lines += [_table(results), ""]
        dump[name] = [json.loads(r.to_json()) for r in results]

        for device in required:
            valid = [
                t
                for r in results
                for t in r.timings
                if _device_of(t) == device and t.ok and t.within_tolerance is not False
            ]
            if not valid:
                errs = [
                    asdict(t)
                    for r in results
                    for t in r.timings
                    if _device_of(t) == device
                ]
                failures.append(
                    f"{name}: no valid timing on required {device!r}: {errs}"
                )

    report = "\n".join(lines)
    print(report)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dump, f, indent=2)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
