#!/usr/bin/env python3
"""Inject trainable LoRA (low-rank adaptation) branches into a bare .onnx
model, in place of full-parameter fine-tuning.

Every targeted MatMul/Gemm weight stays frozen; a small
`(alpha/rank) * (X @ lora_A @ lora_B)` branch is added alongside its output
(lora_A Kaiming-normal, lora_B zero, so the model is numerically unchanged
until trained). Feed the output to generate_artifacts.py --lora-params-file
to train only the injected adapter weights.
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
        help="where to write the LoRA-injected .onnx model",
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
        help="only target MatMul/Gemm nodes whose 2-D weight initializer name contains this "
        "substring (repeatable). Omit to target every eligible node.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--params-out",
        required=True,
        help="where to write the JSON adapter manifest (rank/alpha/param names), needed by "
        "generate_artifacts.py --lora-params-file and extract_lora_adapter.py",
    )
    args = p.parse_args()

    alpha = args.alpha if args.alpha is not None else float(args.rank)

    model = onnx.load(args.model)
    added = lora_surgery.inject(
        model, args.target_contains, args.rank, alpha, seed=args.seed
    )
    if not added:
        sys.exit(
            "error: no eligible MatMul/Gemm targets matched (2-D weight initializer, non-transposed input)"
        )

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    with open(args.params_out, "w") as f:
        json.dump({"rank": args.rank, "alpha": alpha, "pairs": added}, f, indent=2)

    print(
        f"injected {len(added)} LoRA adapter(s), rank={args.rank} alpha={alpha} -> {args.output}"
    )
    print(f"wrote adapter manifest -> {args.params_out}")


if __name__ == "__main__":
    main()
