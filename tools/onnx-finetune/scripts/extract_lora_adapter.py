#!/usr/bin/env python3
"""Pull the trained LoRA adapter weights out of a fine-tuned inference model.

onnx-finetune's --output-model is a full, standalone inference model (frozen
base weights plus the trained lora_A/lora_B branches injected by
inject_lora.py). This script copies out just the adapter initializers listed
in inject_lora.py's manifest into a small standalone .onnx file -- not a
runnable model, just a tensor container -- so the adapter can be shipped and
applied on its own instead of the whole fine-tuned model. See
apply_lora_adapter.py to graft it back onto a base model.
"""

import argparse
import json

import onnx
from onnx import helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("finetuned_model", help="output of onnx-finetune --output-model")
    p.add_argument(
        "--params-file", required=True, help="manifest written by inject_lora.py"
    )
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="where to write the standalone adapter .onnx",
    )
    args = p.parse_args()

    with open(args.params_file) as f:
        manifest = json.load(f)
    names = [n for pair in manifest["pairs"] for n in pair]

    model = onnx.load(args.finetuned_model)
    by_name = {i.name: i for i in model.graph.initializer}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(
            f"error: adapter params missing from {args.finetuned_model}: {missing}"
        )

    graph = helper.make_graph(
        nodes=[],
        name="lora_adapter",
        inputs=[],
        outputs=[],
        initializer=[by_name[n] for n in names],
    )
    adapter_model = helper.make_model(graph, opset_imports=model.opset_import)
    adapter_model.ir_version = model.ir_version
    onnx.save(adapter_model, args.output)
    print(
        f"wrote {len(names)} adapter tensor(s) (rank={manifest['rank']} alpha={manifest['alpha']}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
