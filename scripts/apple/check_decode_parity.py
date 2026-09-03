#!/usr/bin/env python3
"""Check that a Core ML decoder's greedy output agrees with its HF reference.

`run_llm_decode_benchmark.py` measures *speed*; this measures *correctness*
against something other than itself -- the original Hugging Face model,
running greedy `generate()` on CPU through `transformers` -- rather than only
comparing the exported model to itself run twice. That is the "device"
(here: real Core ML, exercised the same way the benchmark does -- prefill
once, then one single-token forward pass per step reusing the growing KV
cache) side of DeviceMark's device-vs-reference parity idea
(https://devicemark.github.io/methodology.html); the "HF" side is exactly
what this script loads for the reference. DeviceMark's own parity gate is
greedy **token-exact** (device text == Mac engine text == HF reference text,
verbatim) before a row's speed number is trusted at all -- this script is
deliberately looser than that, for a real reason:

This is deliberately **not** a bit-exact/token-exact check. Core ML's
`.mlpackage` runs in fp16 internally regardless of the ONNX graph's own
dtype (see `export_llm_to_coreml.py`'s module docstring), and `transformers`'
reference here runs in fp32 on CPU -- two different numeric paths through the
same weights. On a token whose top-2 logits are close, that's enough to flip
the greedy argmax, and one flipped token shifts every later token's context,
so occasional divergence after the first mismatch is expected and not itself
a bug. What this script actually watches for is *how much* the two disagree:
a low token-level agreement rate, or a divergence that happens almost
immediately, is the signal that something in the export/conversion pipeline
(not just fp16 rounding) is wrong.

Usage:
    python check_decode_parity.py HuggingFaceTB/SmolLM2-135M-Instruct \\
        smollm2.mlpackage --prompt "The capital of France is" \\
        --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def compare_token_sequences(reference_ids: list[int], candidate_ids: list[int]) -> dict:
    """Token-level agreement between two greedily-generated id sequences.

    Compares position-by-position over the overlapping length (generation
    lengths can differ, e.g. one side hits EOS earlier) and reports the
    fraction of positions that match plus the index of the first mismatch
    (`None` if the sequences agree everywhere compared). Kept separate from
    any model/tokenizer loading so it's plain, fast-testable logic.
    """
    n = min(len(reference_ids), len(candidate_ids))
    if n == 0:
        return {"compared": 0, "agreement_rate": 1.0, "first_divergence": None}

    first_divergence = None
    matches = 0
    for i in range(n):
        if reference_ids[i] == candidate_ids[i]:
            matches += 1
        elif first_divergence is None:
            first_divergence = i

    return {
        "compared": n,
        "agreement_rate": matches / n,
        "first_divergence": first_divergence,
    }


def _generate_reference_ids(
    model_id: str, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None
) -> list[int]:
    """Greedily generate `max_new_tokens` continuation ids with the original HF
    model on CPU -- the "ground truth" side of the comparison.
    """
    import torch
    from transformers import AutoModelForCausalLM

    print(f"Loading HF reference model {model_id!r} (CPU, fp32)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )
    return out[0, len(prompt_ids) :].tolist()


def _generate_coreml_ids(
    mlpackage: str,
    compute_units: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_llm_decode_benchmark import CoreMLDecoder

    print(f"Loading {mlpackage} (compute_units={compute_units})...", flush=True)
    decoder = CoreMLDecoder(mlpackage, compute_units=compute_units)
    return [
        token_id
        for token_id, _dt in decoder.generate(prompt_ids, max_new_tokens, eos_token_id)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "model_id", help="Hugging Face model id the .mlpackage was exported from"
    )
    ap.add_argument(
        "mlpackage", help="Path to the .mlpackage from export_llm_to_coreml.py"
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    ap.add_argument(
        "--min-agreement",
        type=float,
        default=0.8,
        help="Exit non-zero if the token-level agreement rate falls below this "
        "fraction (default: 0.8). fp16-vs-fp32 numeric drift makes bit-exact "
        "agreement the wrong bar -- see the module docstring.",
    )
    args = ap.parse_args()

    model_dir = Path(args.mlpackage).parent
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    prompt_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    print(f"Prompt: {args.prompt!r} ({len(prompt_ids)} tokens)", flush=True)

    reference_ids = _generate_reference_ids(
        args.model_id, prompt_ids, args.max_new_tokens, tokenizer.eos_token_id
    )
    candidate_ids = _generate_coreml_ids(
        args.mlpackage,
        args.compute_units,
        prompt_ids,
        args.max_new_tokens,
        tokenizer.eos_token_id,
    )

    print(
        f"HF reference : {tokenizer.decode(reference_ids, skip_special_tokens=True)!r}"
    )
    print(
        f"Core ML      : {tokenizer.decode(candidate_ids, skip_special_tokens=True)!r}"
    )

    result = compare_token_sequences(reference_ids, candidate_ids)
    print(
        f"\nagreement: {result['agreement_rate']:.1%} over {result['compared']} "
        f"compared tokens (reference: {len(reference_ids)}, Core ML: "
        f"{len(candidate_ids)})",
        flush=True,
    )
    if result["first_divergence"] is not None:
        print(
            f"first divergence at token index {result['first_divergence']}", flush=True
        )

    if result["agreement_rate"] < args.min_agreement:
        print(
            f"FAIL: agreement {result['agreement_rate']:.1%} is below "
            f"--min-agreement {args.min_agreement:.1%}",
            file=sys.stderr,
        )
        return 1

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
