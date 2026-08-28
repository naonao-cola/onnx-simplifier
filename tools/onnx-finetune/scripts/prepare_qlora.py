#!/usr/bin/env python3
"""One-shot QLoRA prep: inject trainable LoRA adapters, then NF4-quantize
the (now frozen) base weights they sit on top of -- Dettmers et al. 2023's
actual recipe (freeze a 4-bit-quantized base model, train small full-
precision low-rank adapters on top of its on-the-fly dequantized weights),
rather than plain LoRA on an unquantized base.

Order matters, in both directions:

  - LoRA must be injected *before* NF4 quantization: onnxsim.nf4's own
    matcher looks for a plain MatMul/Gemm feeding straight off a 2-D
    initializer. Quantize first and that same node's weight input becomes
    a computed dequantize-subgraph output instead, and inject_lora.py's
    matcher (the same kind of plain-initializer check) would find nothing
    left to graft onto.

  - The injected lora_A/lora_B weights must then be excluded from
    quantization: onnxsim.nf4's matcher only looks at op_type and shape,
    and a LoRA branch is itself just a MatMul against a small 2-D
    initializer -- structurally indistinguishable from any other layer's
    weight. Left unexcluded, quantizing after injection would NF4-quantize
    the adapter too, defeating the point of keeping it full precision.
    This script always passes the injected pair names as skip_names, so
    that never happens.

Needs onnxsim importable (this repo's own package, i.e. built/installed --
see the top-level README) for the NF4 quantization step; plain
inject_lora.py needs only onnx + numpy.

The rest of the pipeline is unchanged: feed the output to
generate_artifacts.py --lora-params-file, train as usual, then
extract_lora_adapter.py / apply_lora_adapter.py work exactly as they do for
plain (non-quantized) LoRA -- they only ever touch lora_A/lora_B by name.
NF4 quantization is presently limited to 2-D MatMul/vanilla-Gemm weights
(see onnxsim/nf4.py); a 1x1-Conv LoRA target stays full precision here.
"""

import argparse
import json
import sys

import lora_surgery
import onnx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="input .onnx file")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="where to write the QLoRA-ready .onnx model",
    )
    p.add_argument("--rank", type=int, default=4, help="adapter rank (default 4)")
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA scaling numerator, delta is scaled by alpha/rank (default: equal to --rank, i.e. scale 1.0)",
    )
    p.add_argument(
        "--target-contains",
        action="append",
        default=[],
        help="only target MatMul/Gemm/1x1-Conv nodes whose weight initializer name contains this "
        "substring (repeatable). Omit to target every eligible node.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="NF4 quantization block size (bitsandbytes' own QLoRA default is 64)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--params-out", required=True, help="where to write the JSON adapter manifest"
    )
    args = p.parse_args()

    try:
        from onnxsim.nf4 import quantize_weight_only_nf4
    except ImportError as e:
        sys.exit(
            "error: NF4 quantization needs the onnxsim package importable (build/install this "
            f"repo's own wheel, see the top-level README) -- import failed: {e}"
        )

    alpha = args.alpha if args.alpha is not None else float(args.rank)

    model = onnx.load(args.model)
    added = lora_surgery.inject(
        model, args.target_contains, args.rank, alpha, seed=args.seed
    )
    if not added:
        sys.exit("error: no eligible MatMul/Gemm/1x1-Conv targets matched")

    lora_param_names = {n for pair in added for n in pair}
    model = quantize_weight_only_nf4(
        model, block_size=args.block_size, skip_names=lora_param_names
    )

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    with open(args.params_out, "w") as f:
        json.dump({"rank": args.rank, "alpha": alpha, "pairs": added}, f, indent=2)

    print(
        f"injected {len(added)} LoRA adapter(s) (rank={args.rank} alpha={alpha}) and "
        f"NF4-quantized the frozen base weights (block_size={args.block_size}) -> {args.output}"
    )
    print(f"wrote adapter manifest -> {args.params_out}")


if __name__ == "__main__":
    main()
