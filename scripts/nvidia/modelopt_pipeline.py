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
import multiprocessing
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
from onnx import version_converter

import onnxsim


def fix_batch(model, batch, opset, chw=None):
    """Lift to ``opset``, pin the batch dim (and any dynamic C/H/W dims to ``chw``),
    drop weight-as-input declarations."""
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
            if chw and io in list(model.graph.input):
                for d, v in zip(list(dims)[1:], chw):
                    d.dim_value = v  # e.g. HF exports leave height/width symbolic
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def variant_options(model, variant):
    """Translate a ``+``-joined variant spec into ModelOpt ``quantize`` kwargs.

    hp32     keep everything that is not quantized in fp32 (ModelOpt's default is fp16)
    noattn   do not quantize the attention batched MatMuls (Q@K^T, attn@V: MatMul nodes
             with no constant input) -- only the weight Linear/Conv layers
    nohead   also skip the first Conv (patch/stem embedding) and the last Gemm/MatMul
    only-G   quantize only the weight Linears of one ViT-B group G in {qkv, proj, fc1, fc2},
             or of all four with G=linear (leaves residual Adds, attention, Conv, head alone)
             (matched by their [K, N] weight shape; a sensitivity probe -- everything
             else, incl. attention MatMuls, Conv and the head, stays unquantized)
    """
    kw, exclude = {}, []
    init = {t.name for t in model.graph.initializer}
    nodes = list(model.graph.node)
    for opt in variant.split("+"):
        if opt == "default":
            continue
        elif opt == "hp32":
            kw["high_precision_dtype"] = "fp32"
        elif opt == "noattn":
            exclude += [n.name for n in nodes
                        if n.op_type == "MatMul" and not any(i in init for i in n.input)]
        elif opt == "nohead":
            convs = [n for n in nodes if n.op_type == "Conv"]
            heads = [n for n in nodes if n.op_type in ("Gemm", "MatMul")
                     and any(i in init for i in n.input)]
            exclude += [n.name for n in convs[:1] + heads[-1:]]
        elif opt.startswith("only-"):
            want = opt[len("only-"):]
            wanted = {"qkv", "proj", "fc1", "fc2"} if want == "linear" else {want}
            dims = {t.name: list(t.dims) for t in model.graph.initializer}
            groups = {(768, 2304): "qkv", (768, 768): "proj", (768, 3072): "fc1", (3072, 768): "fc2"}
            for n in nodes:
                w = next((dims[i] for i in n.input if i in dims and len(dims[i]) == 2), None)
                if n.op_type not in ("Gemm", "MatMul") or w is None:
                    keep = False
                else:
                    trans_b = any(a.name == "transB" and a.i for a in n.attribute)
                    k, m_out = (w[1], w[0]) if trans_b else (w[0], w[1])
                    keep = groups.get((k, m_out)) in wanted
                if not keep:
                    exclude.append(n.name)
        else:
            raise ValueError(f"unknown variant option {opt!r}")
    if exclude:
        kw["nodes_to_exclude"] = [re.escape(n) for n in exclude]
    return kw


def name_nodes(model):
    """Give every unnamed node a unique name (simplified graphs often have none, and
    ModelOpt's node exclusion is name based)."""
    for i, n in enumerate(model.graph.node):
        if not n.name:
            n.name = f"{n.op_type}_{i}"
    return model


def quantize_int8(src, dst, calib, input_name, method, variant="default"):
    from modelopt.onnx.quantization import quantize

    kw = variant_options(onnx.load(str(src), load_external_data=False), variant)
    quantize(
        onnx_path=str(src),
        quantize_mode="int8",
        calibration_data={input_name: calib},
        calibration_method=method,
        calibration_eps=["cpu"],
        output_path=str(dst),
        **kw,
    )


def quantize_int8_isolated(src, dst, calib, input_name, method, variant="default"):
    """Run ``quantize_int8`` in a fresh process so ModelOpt's calibration memory (it keeps
    histograms of every intermediate tensor) is returned to the OS after each model --
    a ViT-B pipeline in one process gets OOM-killed on an 8 GB Jetson."""
    proc = multiprocessing.get_context("spawn").Process(
        target=quantize_int8, args=(src, dst, calib, input_name, method, variant))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"quantization process exited with {proc.exitcode}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--chw", type=int, nargs=3, help="pin dynamic C H W input dims (e.g. 3 224 224)")
    ap.add_argument("--calib-samples", type=int, default=16, help="random-noise samples if no --calib-npy")
    ap.add_argument("--calib-npy", help="float32 NCHW calibration images (.npy)")
    ap.add_argument("--calib-n", type=int, help="use only the first N --calib-npy samples (bounds memory)")
    ap.add_argument("--variants", nargs="+", default=["default"],
                    help="ModelOpt variants, '+'-joined from hp32/noattn/nohead (see variant_options)")
    ap.add_argument("--tags", nargs="+", default=["raw", "sim"], choices=["raw", "sim"])
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
        raw = fix_batch(src, batch, args.opset, args.chw)
        onnx.save(raw, out / f"b{batch}.raw.onnx")
        t0 = time.perf_counter()
        sim, ok = onnxsim.simplify(raw)
        assert ok, "onnxsim correctness check failed"
        name_nodes(sim)
        onnx.save(sim, out / f"b{batch}.sim.onnx")
        print(f"b{batch}: {len(raw.graph.node)} -> {len(sim.graph.node)} nodes "
              f"(onnxsim {time.perf_counter() - t0:.1f}s)")

        if args.calib_npy:
            calib = np.load(args.calib_npy).astype(np.float32)[: args.calib_n]
        else:
            shape = [d.dim_value for d in raw.graph.input[0].type.tensor_type.shape.dim]
            shape[0] = args.calib_samples
            calib = rng.standard_normal(shape).astype(np.float32)
        jobs = [(tag, variant, method) for tag in args.tags
                for variant in args.variants for method in args.methods]
        for tag, variant, method in jobs:
            name = f"b{batch}.{tag}.int8-{method}" + ("" if variant == "default" else f"-{variant}")
            if (out / f"{name}.onnx").exists():
                print(f"{name}: exists, skipped")
                continue
            try:
                t0 = time.perf_counter()
                quantize_int8_isolated(out / f"b{batch}.{tag}.onnx", out / f"{name}.onnx",
                                       calib, input_name, method, variant)
                n = sum(x.op_type == "QuantizeLinear"
                        for x in onnx.load(out / f"{name}.onnx").graph.node)
                print(f"{name}: {n} QuantizeLinear ({time.perf_counter() - t0:.1f}s)")
            except Exception:
                print(f"{name}: ModelOpt FAILED")
                traceback.print_exc(limit=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
