#!/usr/bin/env python3
"""Export a trained LoRA adapter (extract_lora_adapter.py's output) to ONNX
Runtime's own native ``.onnx_adapter`` file format, loadable at inference
time via ``onnxruntime.LoraAdapter`` / ``RunOptions.add_active_adapter`` --
ONNX Runtime's built-in mechanism for swapping adapters in and out of a
running session without reloading or re-merging the base model.

No Olive needed: despite Olive's own ``convert-adapters`` CLI being the
usual way people reach this format, the actual serialization is just a
public onnxruntime API (``onnxruntime.AdapterFormat``, added in ORT 1.20)
that Olive's command is a thin wrapper around -- see
``ConvertAdaptersCommand.run`` in Olive's source. Calling it directly here
avoids an olive-ai/torch/peft dependency chain for what is, underneath,
one class's ``set_parameters`` + ``export_adapter``.

For this to be loadable, the base model must expose lora_A/lora_B as graph
*inputs* (not just baked initializers) under these exact names -- pass
--adapter-inputs to inject_lora.py or apply_lora_adapter.py when preparing
that model. Names only need to be internally consistent between the model
and the adapter file produced here; unlike Olive's own HuggingFace-oriented
naming convention, there is no fixed scheme to match since both sides come
from this same toolchain.
"""

import argparse
import json
import sys

import onnx
from onnx import numpy_helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("adapter", help="output of extract_lora_adapter.py")
    p.add_argument(
        "--params-file", required=True, help="manifest written by inject_lora.py"
    )
    p.add_argument(
        "-o", "--output", required=True, help="where to write the .onnx_adapter file"
    )
    args = p.parse_args()

    try:
        import onnxruntime as ort
        from packaging.version import Version
    except ImportError as e:
        sys.exit(
            f"error: this needs the onnxruntime package installed -- import failed: {e}"
        )
    if Version(ort.__version__) < Version("1.20"):
        sys.exit(
            f"error: onnxruntime.AdapterFormat needs onnxruntime >= 1.20, found {ort.__version__}"
        )

    with open(args.params_file) as f:
        manifest = json.load(f)
    names = [n for pair in manifest["pairs"] for n in pair]

    model = onnx.load(args.adapter)
    by_name = {i.name: i for i in model.graph.initializer}
    missing = [n for n in names if n not in by_name]
    if missing:
        sys.exit(f"error: adapter params missing from {args.adapter}: {missing}")

    weights = {n: numpy_helper.to_array(by_name[n]) for n in names}

    adapter_format = ort.AdapterFormat()
    adapter_format.set_parameters(
        {n: ort.OrtValue.ortvalue_from_numpy(v) for n, v in weights.items()}
    )
    adapter_format.export_adapter(args.output)
    print(
        f"wrote {len(weights)} adapter tensor(s) -> {args.output} (native ONNX Runtime .onnx_adapter)"
    )


if __name__ == "__main__":
    main()
