"""Prepare TensorRT benchmark variants of a classifier with onnxsim + NVIDIA ModelOpt.

Run under the onnxsim interpreter (Python >= 3.11) with ``nvidia-modelopt[onnx]``
installed. For each batch size it writes into OUT_DIR:

    b{B}.raw.onnx        batch fixed only (no onnxsim): the control
    b{B}.sim.onnx        onnxsim.simplify() of the above
    b{B}.raw.int8-{M}.onnx   ModelOpt INT8 (explicit Q/DQ) quantization of raw
    b{B}.sim.int8-{M}.onnx   ModelOpt INT8 quantization of the simplified model

for each calibration method M (``--methods``, ModelOpt: ``entropy`` (default) or ``max``).
Calibration uses ``--calib-npy`` (real preprocessed images, see ``imagenette_data.py``)
when given, otherwise random noise (fine for latency, meaningless for accuracy).

which ``bench_trtexec.py`` then builds/times with ``trtexec``. A stage that fails
is reported and skipped (that is itself a finding, e.g. ModelOpt rejecting an
unsimplified graph).

    python modelopt_pipeline.py resnet18.onnx OUT_DIR --batch 1 8 \
        --calib-npy data/calib_x.npy --methods entropy max
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
from onnx import version_converter

import onnxsim


def fix_batch(model, batch, opset):
    """Lift to ``opset``, pin the batch dim, drop weight-as-input declarations."""
    # Convert first: the C++ converter needs IR<4 models' initializers to still be
    # listed as graph inputs.
    model = version_converter.convert_version(model, opset)
    model.ir_version = 8
    inits = {t.name for t in model.graph.initializer}
    keep = [i for i in model.graph.input if i.name not in inits]
    del model.graph.input[:]
    model.graph.input.extend(keep)
    for io in list(model.graph.input) + list(model.graph.output):
        dims = io.type.tensor_type.shape.dim
        if dims:
            dims[0].dim_value = batch
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def quantize_int8(src, dst, calib, input_name, method):
    from modelopt.onnx.quantization import quantize

    quantize(
        onnx_path=str(src),
        quantize_mode="int8",
        calibration_data={input_name: calib},
        calibration_method=method,
        calibration_eps=["cpu"],
        output_path=str(dst),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--calib-samples", type=int, default=16, help="random-noise samples if no --calib-npy")
    ap.add_argument("--calib-npy", help="float32 NCHW calibration images (.npy)")
    ap.add_argument("--methods", nargs="+", default=["entropy"], choices=["entropy", "max"])
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = onnx.load(args.model)
    input_name = next(
        i.name for i in src.graph.input if i.name not in {t.name for t in src.graph.initializer}
    )
    rng = np.random.default_rng(0)
    for batch in args.batch:
        raw = fix_batch(src, batch, args.opset)
        onnx.save(raw, out / f"b{batch}.raw.onnx")
        t0 = time.perf_counter()
        sim, ok = onnxsim.simplify(raw)
        assert ok, "onnxsim correctness check failed"
        onnx.save(sim, out / f"b{batch}.sim.onnx")
        print(f"b{batch}: {len(raw.graph.node)} -> {len(sim.graph.node)} nodes "
              f"(onnxsim {time.perf_counter() - t0:.1f}s)")

        if args.calib_npy:
            calib = np.load(args.calib_npy).astype(np.float32)
        else:
            shape = [d.dim_value for d in raw.graph.input[0].type.tensor_type.shape.dim]
            shape[0] = args.calib_samples
            calib = rng.standard_normal(shape).astype(np.float32)
        for tag in ("raw", "sim"):
            for method in args.methods:
                name = f"b{batch}.{tag}.int8-{method}"
                try:
                    t0 = time.perf_counter()
                    quantize_int8(out / f"b{batch}.{tag}.onnx", out / f"{name}.onnx",
                                  calib, input_name, method)
                    n = sum(x.op_type == "QuantizeLinear"
                            for x in onnx.load(out / f"{name}.onnx").graph.node)
                    print(f"{name}: {n} QuantizeLinear ({time.perf_counter() - t0:.1f}s)")
                except Exception:
                    print(f"{name}: ModelOpt FAILED")
                    traceback.print_exc(limit=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
