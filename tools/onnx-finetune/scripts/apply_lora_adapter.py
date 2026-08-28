#!/usr/bin/env python3
"""Graft a previously trained LoRA adapter (extract_lora_adapter.py's output)
onto a fresh copy of its base model, without retraining -- the point of
keeping the adapter small and separate from the frozen base weights: one
adapter file can be re-applied to as many copies of the base model as
needed, or swapped for a different adapter trained the same way.
"""

import argparse
import json

import lora_surgery
import onnx
from onnx import numpy_helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "base_model", help="the original .onnx model inject_lora.py was run against"
    )
    p.add_argument("--adapter", required=True, help="output of extract_lora_adapter.py")
    p.add_argument(
        "--params-file", required=True, help="manifest written by inject_lora.py"
    )
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="where to write the ready-to-run inference model",
    )
    args = p.parse_args()

    with open(args.params_file) as f:
        manifest = json.load(f)

    adapter_model = onnx.load(args.adapter)
    values_by_name = {
        i.name: numpy_helper.to_array(i) for i in adapter_model.graph.initializer
    }

    lora_values = {}
    for a_name, b_name in manifest["pairs"]:
        weight_name = a_name.rsplit(".lora_A", 1)[0]
        lora_values[weight_name] = (values_by_name[a_name], values_by_name[b_name])

    model = onnx.load(args.base_model)
    lora_surgery.inject(
        model,
        None,
        manifest["rank"],
        manifest["alpha"],
        lora_values=lora_values,
        exact_names=set(lora_values.keys()),
    )

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"applied {len(lora_values)} adapter(s) from {args.adapter} -> {args.output}")


if __name__ == "__main__":
    main()
