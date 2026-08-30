#!/usr/bin/env python3
"""Export a Hugging Face causal LM to a static-shape Core ML decoder.

Bridges a "decoder-with-past" ONNX export (produced by `optimum-onnx`,
https://github.com/huggingface/optimum-onnx) to `onnxsim.export_coreml`, which
requires fully static input shapes (see `onnxsim/coreml_export.py`). A real
KV-cache decode loop needs the `past_sequence_length` axis to grow every step,
which is dynamic -- there is no way to bake that into one static-shape Core ML
model without extending the translator to support symbolic shapes.

This script sidesteps that with a different (and simpler) decode strategy: a
single static model with a fixed context window (`--max-length`), an *always
empty* KV cache (`past_sequence_length` pinned to 0), and the full token
sequence recomputed from scratch on every step (right-padded up to
`--max-length`, reading the logits at the last real position). Standard causal
masking already guarantees padding after the current position can't influence
earlier positions, so this is exactly as correct as real KV-cache decoding --
just O(context length) more compute per token instead of O(1), since nothing
from the previous step's forward pass is reused. `run_llm_decode_benchmark.py`
is the harness that actually runs this model and measures the resulting
decode throughput; the numbers it reports describe *this* recompute strategy,
not a production KV-cache deployment.

Pipeline: `optimum.exporters.onnx` (decoder-with-past, static batch/sequence
length) -> pin `past_sequence_length` to 0 across every input/output ->
`onnxsim.simplify` (folds the now-constant shape/mask arithmetic -- this is
what turns most of the graph's `Shape`/`Equal`/`Range`/`ConstantOfShape`
plumbing into plain constants before the Core ML translator ever sees it) ->
`onnxsim.export_coreml`.

Usage:
    python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \\
        --max-length 64 --output smollm2.mlpackage
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import onnx


def _fix_static_shapes(
    model: onnx.ModelProto, batch_size: int, sequence_length: int
) -> None:
    """Replace `optimum`'s symbolic dims with the concrete ones it actually traced.

    `optimum.exporters.onnx`'s `--no-dynamic-axes`/`--batch_size`/`--sequence_length`
    flags control the dummy inputs used to *trace* the model, not the dim_param
    names written into the exported graph's declared input/output shapes -- those
    stay symbolic regardless. The traced computation is already specialized to the
    concrete shapes given at export time, so it's safe (and necessary, for
    onnxsim's Core ML exporter) to overwrite the declared shapes to match.
    """
    dims = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "past_sequence_length": 0,
        "past_sequence_length + sequence_length": sequence_length,
    }
    for vi in list(model.graph.input) + list(model.graph.output):
        tt = vi.type.tensor_type
        for d in tt.shape.dim:
            if d.HasField("dim_param"):
                if d.dim_param not in dims:
                    raise ValueError(
                        f"Unrecognized dynamic dim {d.dim_param!r} on {vi.name!r}"
                    )
                value = dims[d.dim_param]
                d.Clear()
                d.dim_value = value


def export_llm_to_coreml(
    model_id: str,
    output_path: str,
    *,
    max_length: int = 64,
    opset: int = 17,
    convert_to: str = "mlprogram",
) -> None:
    from optimum.exporters.onnx import main_export

    with tempfile.TemporaryDirectory(prefix="onnxsim_llm_export_") as tmpdir:
        print(
            f"Exporting {model_id!r} to ONNX (decoder-with-past, opset {opset})...",
            flush=True,
        )
        main_export(
            model_id,
            output=tmpdir,
            task="text-generation-with-past",
            opset=opset,
            no_dynamic_axes=True,
            batch_size=1,
            sequence_length=max_length,
        )

        onnx_path = Path(tmpdir) / "model.onnx"
        model = onnx.load(str(onnx_path))

        print(
            f"Pinning shapes to batch=1, sequence_length={max_length}, past_sequence_length=0...",
            flush=True,
        )
        _fix_static_shapes(model, batch_size=1, sequence_length=max_length)
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

        print(f"Converting to Core ML ({convert_to})...", flush=True)
        onnxsim.export_coreml(simplified, output_path, convert_to=convert_to)

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
        "--max-length",
        type=int,
        default=64,
        help="Fixed context window: prompt + generated tokens must fit within this many "
        "tokens (default: 64). Every decode step reprocesses the whole window (see the "
        "module docstring), so this is also the per-step compute cost -- keep it small.",
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
        max_length=args.max_length,
        opset=args.opset,
        convert_to=args.convert_to,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
