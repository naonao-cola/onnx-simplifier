#!/usr/bin/env python3
"""Export a Hugging Face causal LM to a real KV-cache Core ML decoder.

Bridges a "decoder-with-past" ONNX export (produced by `optimum-onnx`,
https://github.com/huggingface/optimum-onnx) to `onnxsim.export_coreml`. Unlike
a plain static-shape export, this keeps `sequence_length` and
`past_sequence_length` as genuinely dynamic axes all the way through to the
Core ML model (via `onnxsim.export_coreml`'s `dynamic_shapes` argument, see
`onnxsim/coreml_export.py`), so the resulting `.mlpackage` is one model that
supports both:

- **prefill**: one forward pass over the whole prompt (`sequence_length` =
  prompt length, `past_sequence_length` = 0), producing the initial cache.
- **decode**: one forward pass per new token (`sequence_length` = 1),
  consuming and extending the growing cache from the previous step.

This is a real, O(1)-per-decode-step KV cache -- each step only computes the
new token's attention over the accumulated `present_*` cache from every prior
step, instead of reprocessing the whole context window every time.
`run_llm_decode_benchmark.py` is the harness that actually runs this model
and carries the cache across steps.

Only `batch_size` is pinned to a concrete value (1); `sequence_length`,
`past_sequence_length`, and the composite `past_sequence_length +
sequence_length` axis ONNX exporters emit for the attention mask are left
dynamic, bounded by `--max-context-length` (the largest prompt + generated
tokens the exported model will accept).

`optimum`'s tracer cannot trace a `sequence_length` of exactly 1 for
Llama-family models without corrupting the attention mask (see
`onnx_export_from_model`'s "sequence length of 1" check) -- this only affects
which shape is used to *trace* the model, not what shapes the exported graph
accepts at runtime, so this script traces with `sequence_length=2` and lets
`onnxsim.simplify` + the dynamic Core ML export handle the true dynamic
range, `sequence_length=1` decode steps included.

Pipeline: `optimum.exporters.onnx` (decoder-with-past, dynamic axes) -> pin
`batch_size` to 1 -> `onnxsim.simplify` -> `onnxsim.export_coreml` with
`dynamic_shapes` for `sequence_length` / `past_sequence_length` / their sum.

Usage:
    python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \\
        --max-context-length 512 --output smollm2.mlpackage
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import onnx


def _fix_batch_size(model: onnx.ModelProto, batch_size: int) -> None:
    """Pin the `batch_size` dim_param to a concrete value everywhere it appears.

    `sequence_length`, `past_sequence_length`, and the composite
    `past_sequence_length + sequence_length` dim_params are deliberately left
    alone -- those are the axes this script's whole point is to keep dynamic.
    """
    for vi in list(model.graph.input) + list(model.graph.output):
        tt = vi.type.tensor_type
        for d in tt.shape.dim:
            if d.HasField("dim_param") and d.dim_param == "batch_size":
                d.Clear()
                d.dim_value = batch_size


def export_llm_to_coreml(
    model_id: str,
    output_path: str,
    *,
    max_context_length: int = 512,
    opset: int = 17,
    convert_to: str = "mlprogram",
) -> None:
    from optimum.exporters.onnx import main_export

    with tempfile.TemporaryDirectory(prefix="onnxsim_llm_export_") as tmpdir:
        print(
            f"Exporting {model_id!r} to ONNX (decoder-with-past, dynamic shapes, "
            f"opset {opset})...",
            flush=True,
        )
        main_export(
            model_id,
            output=tmpdir,
            task="text-generation-with-past",
            opset=opset,
            batch_size=1,
            # Only used to trace the model; the exported graph keeps
            # sequence_length/past_sequence_length dynamic regardless (see the
            # module docstring for why 2, not 1).
            sequence_length=2,
        )

        onnx_path = Path(tmpdir) / "model.onnx"
        model = onnx.load(str(onnx_path))

        print(
            "Pinning batch_size=1 (sequence_length/past_sequence_length stay dynamic)...",
            flush=True,
        )
        _fix_batch_size(model, batch_size=1)
        onnx.checker.check_model(model)

        print("Simplifying with onnxsim...", flush=True)
        import onnxsim

        simplified, ok = onnxsim.simplify(model)
        if not ok:
            raise RuntimeError("onnxsim.simplify() reported failure")
        print(
            f"  {len(model.graph.node)} -> {len(simplified.graph.node)} nodes",
            flush=True,
        )

        print(f"Converting to Core ML ({convert_to}), dynamic KV cache...", flush=True)
        onnxsim.export_coreml(
            simplified,
            output_path,
            convert_to=convert_to,
            dynamic_shapes={
                "sequence_length": (1, 1, max_context_length),
                "past_sequence_length": (0, 0, max_context_length - 1),
                "past_sequence_length + sequence_length": (1, 1, max_context_length),
            },
        )

        # Carry the tokenizer along so run_llm_decode_benchmark.py can load
        # everything it needs from `output_path`'s sibling files.
        out_dir = Path(output_path).parent
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            src = Path(tmpdir) / name
            if src.is_file():
                shutil.copy(src, out_dir / name)

    print(f"Wrote {output_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "model_id", help="Hugging Face model id or local path (a causal LM)"
    )
    ap.add_argument(
        "--output", default="model.mlpackage", help="Output .mlpackage path"
    )
    ap.add_argument(
        "--max-context-length",
        type=int,
        default=512,
        help="Upper bound on prompt + generated tokens the exported model will accept "
        "(default: 512). Unlike a fixed-context recompute model, this only bounds the "
        "KV cache's Core ML flexible-shape range -- actual per-step compute scales with "
        "how much of that range is in use, not the bound itself.",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--format",
        choices=["mlprogram", "neuralnetwork"],
        default="mlprogram",
        dest="convert_to",
    )
    args = ap.parse_args()

    export_llm_to_coreml(
        args.model_id,
        args.output,
        max_context_length=args.max_context_length,
        opset=args.opset,
        convert_to=args.convert_to,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
