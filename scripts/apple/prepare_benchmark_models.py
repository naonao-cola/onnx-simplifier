#!/usr/bin/env python3
"""Export a curated set of real instruct LMs to Core ML in one pass.

A batch wrapper around `export_llm_to_coreml.py`: runs its pipeline once per
model in `BENCHMARK_MODELS` (or a `--only` subset), each into its own
subdirectory, so the result is a ready-to-go directory of `.mlpackage`s --
`run_llm_decode_benchmark.py` can then be pointed at each one in turn to
produce the actual decode tok/s and peak-RSS numbers this whole `scripts/apple`
directory exists to measure (see that script's and `export_llm_to_coreml.py`'s
docstrings for what those numbers do and don't mean).

`BENCHMARK_MODELS` targets the same weight class DeviceMark's own on-device LLM
leaderboard tests (https://devicemark.github.io/, roughly 1-4B parameters),
spanning a few architecture families (Llama-style, Qwen2, Phi-3) rather than
just one, since `onnxsim/coreml_export.py`'s translator is a hand-written
ONNX-to-MIL mapping and different architectures exercise different op
combinations.

A few model families in that weight class require accepting a license on
Hugging Face and an authenticated `HF_TOKEN` to even download.
`meta-llama/Llama-3.2-*-Instruct` is in the default list below anyway: this
repo's CI has a read-only `HF_TOKEN` secret (see
`.github/workflows/coreml-integration.yml`'s `export-benchmark-models` job),
so it downloads there even though it can't in a token-less environment (set
`HF_TOKEN` yourself, or run `huggingface-cli login`, to download it locally).
`google/gemma-2-*-it` is left out of the default list even so -- Gemma 2's
architecture (alternating sliding-window/full attention, logit soft-capping)
hasn't been checked against this translator at all, unlike Llama-3.2's, which
only extends already-validated Llama-family op patterns to a gated checkpoint.
Pass `--only google/gemma-2-2b-it` to try it anyway; nothing about the
pipeline itself is family-specific, so it may well just work.

Not every model in the default list has actually been run through this
pipeline in every environment -- each entry's `notes` field says whether it
has been (and if not, why: usually that converting a multi-billion-parameter
model needs several GB of free RAM and disk, more than a constrained sandbox
has -- see `export_llm_to_coreml.py`'s module docstring for the memory
considerations that scale brings). Treat an unvalidated entry as "expected to
convert, based on the same op patterns validated models already exercise",
not as a guarantee.

Usage:
    # Export everything in BENCHMARK_MODELS into ./benchmark_models/<slug>/model.mlpackage
    python prepare_benchmark_models.py --output-dir benchmark_models

    # Just one model
    python prepare_benchmark_models.py --only Qwen/Qwen2.5-1.5B-Instruct

    # Then, per model, on macOS:
    python run_llm_decode_benchmark.py benchmark_models/Qwen_Qwen2_5_1_5B_Instruct/model.mlpackage \\
        --prompt "The capital of France is" --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from export_llm_to_coreml import export_llm_to_coreml


@dataclass(frozen=True)
class BenchmarkModel:
    model_id: str
    max_context_length: int
    dtype: str
    notes: str


BENCHMARK_MODELS: list[BenchmarkModel] = [
    BenchmarkModel(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        512,
        "fp32",
        "Smoke-test tier, not the weight class this suite is actually measuring -- "
        "validated end-to-end (export, convert, and real macOS predict via this repo's "
        "coreml-integration CI). Useful for a fast sanity check of the pipeline itself.",
    ),
    BenchmarkModel(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        512,
        "fp16",
        "Validated end-to-end (export + convert; on-device predict not yet run in CI).",
    ),
    BenchmarkModel(
        "Qwen/Qwen2.5-1.5B-Instruct",
        512,
        "fp16",
        "Validated end-to-end (export + convert; on-device predict not yet run in CI). "
        "First Qwen2-architecture model exercised -- confirms the translator isn't "
        "Llama-specific (QKV projection bias, different rope_theta, GQA).",
    ),
    BenchmarkModel(
        "Qwen/Qwen2.5-3B-Instruct",
        512,
        "fp16",
        "Attempted but OOM-killed (~13.9GB anon-rss) during the ONNX export/trace step "
        "itself, even with --dtype fp16, in a 15GB-RAM dev sandbox -- one tier up from "
        "Qwen2.5-1.5B-Instruct's ~1.5x smaller weights, which fit comfortably. Not a "
        "translator issue (never reached Core ML conversion); needs a machine with more "
        "headroom than that sandbox had. Same architecture as the validated 1.5B model, "
        "so expected to convert given enough memory.",
    ),
    BenchmarkModel(
        "microsoft/Phi-3.5-mini-instruct",
        512,
        "fp16",
        "Not yet exercised -- at 3.8B params, larger than Qwen2.5-3B-Instruct (which "
        "already OOM'd during export in a 15GB-RAM sandbox, see its entry above), so "
        "expected to hit the same memory ceiling there. Phi-3's fused qkv_proj/gate_up_proj "
        "projections (one big MatMul + Split, instead of separate q/k/v or gate/up "
        "projections) are a structurally different graph shape from Llama/Qwen2's separate "
        "projections, though built from ops (MatMul, Split) this translator already supports.",
    ),
    BenchmarkModel(
        "meta-llama/Llama-3.2-1B-Instruct",
        512,
        "fp16",
        "Gated -- needs HF_TOKEN (see the module docstring). Validated end-to-end "
        "(export + convert) via the export-benchmark-models CI job once access was "
        "granted; on-device predict not yet run. Confirms the gate only blocked the "
        "download -- the graph converts exactly like the already-validated Llama "
        "architecture (SmolLM2) once fetched.",
    ),
    BenchmarkModel(
        "meta-llama/Llama-3.2-3B-Instruct",
        512,
        "fp16",
        "Gated -- needs HF_TOKEN (see the module docstring). Not yet exercised -- at "
        "~3B params, expected to hit the same export-time memory ceiling "
        "Qwen2.5-3B-Instruct did in a 15GB-RAM sandbox (see its entry above); the "
        "gate itself is no longer a blocker (Llama-3.2-1B-Instruct is validated).",
    ),
]


def _slug(model_id: str) -> str:
    """A filesystem-safe directory name for `model_id`, e.g.
    'Qwen/Qwen2.5-1.5B-Instruct' -> 'Qwen_Qwen2_5_1_5B_Instruct'.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def prepare_benchmark_models(
    models: list[BenchmarkModel],
    output_dir: str,
    *,
    max_context_length: int | None = None,
    skip_existing: bool = True,
) -> list[tuple[str, str, Path, float | None]]:
    """Export each of `models` to `<output_dir>/<slug>/model.mlpackage`.

    Returns one `(model_id, status, mlpackage_path, seconds)` tuple per model
    (`status` is ``"ok"``, ``"skipped"``, or ``"FAILED"``) -- a failure is
    logged (full traceback) and skipped rather than aborting the rest of the
    batch, since these are independent, unrelated exports.
    """
    results: list[tuple[str, str, Path, float | None]] = []
    for m in models:
        out_path = Path(output_dir) / _slug(m.model_id) / "model.mlpackage"
        if skip_existing and out_path.exists():
            print(f"Skipping {m.model_id!r} (already exists at {out_path})", flush=True)
            results.append((m.model_id, "skipped", out_path, None))
            continue

        effective_max_context_length = max_context_length or m.max_context_length
        print(
            f"\n=== {m.model_id} (dtype={m.dtype}, "
            f"max_context_length={effective_max_context_length}) ===",
            flush=True,
        )
        t0 = time.time()
        try:
            export_llm_to_coreml(
                m.model_id,
                str(out_path),
                max_context_length=effective_max_context_length,
                dtype=m.dtype,
            )
        except Exception:
            traceback.print_exc()
            results.append((m.model_id, "FAILED", out_path, time.time() - t0))
            continue
        results.append((m.model_id, "ok", out_path, time.time() - t0))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--output-dir",
        default="benchmark_models",
        help="Directory to write each model's <slug>/model.mlpackage into "
        "(default: ./benchmark_models)",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to this model id (repeatable). Not limited to "
        "BENCHMARK_MODELS -- any Hugging Face causal LM id works, using "
        "--max-context-length and fp16 as defaults.",
    )
    ap.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Override every selected model's max context length (default: "
        "each model's own value in BENCHMARK_MODELS, or 512 for a model named "
        "via --only that isn't in that list).",
    )
    ap.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-export a model even if its .mlpackage already exists.",
    )
    args = ap.parse_args()

    by_id = {m.model_id: m for m in BENCHMARK_MODELS}
    if args.only:
        models = [
            by_id.get(model_id)
            or BenchmarkModel(model_id, 512, "fp16", "custom --only model")
            for model_id in args.only
        ]
    else:
        models = BENCHMARK_MODELS

    results = prepare_benchmark_models(
        models,
        args.output_dir,
        max_context_length=args.max_context_length,
        skip_existing=args.skip_existing,
    )

    print("\n=== Summary ===", flush=True)
    failed = False
    for model_id, status, out_path, seconds in results:
        timing = f"{seconds:.1f}s" if seconds is not None else "-"
        print(f"  [{status:7}] {model_id:45} {timing:>8}  {out_path}")
        failed = failed or status == "FAILED"

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
