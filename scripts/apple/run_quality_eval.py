#!/usr/bin/env python3
"""Run a quality-benchmark subset against a float (HF) or quantized (Core ML) model.

A thin CLI over `lm_eval.simple_evaluate` (from `lm-evaluation-harness`,
https://github.com/EleutherAI/lm-evaluation-harness) -- see
`bench/TODO_quality_retention_eval.md` for why that harness rather than a
custom scorer: it already implements IFEval, MMLU-Pro, and MATH-500's prompt
formatting, few-shot sampling, and (the part worth not reimplementing) each
benchmark's answer scoring. This script only adds two things on top: a
`--model coreml` backend (`lm_eval_coreml_adapter.py`, wrapping
`CoreMLDecoder`) so the same harness can score the *exported* model too, and
`--limit` defaulting to a small subset -- a full run of any of these
benchmarks is thousands of prompts, hours at this suite's current unbatched,
single-sequence-at-a-time Core ML decode design (see
`run_llm_decode_benchmark.py`'s module docstring). A `--limit`-restricted run
is explicitly *not* a benchmark-grade score (`lm_eval` itself warns about
this) -- treat its output as "did this get meaningfully worse", not as a
number comparable to a published IFEval/MMLU-Pro/MATH-500 leaderboard entry.

Run the float side (no macOS/Core ML needed) against the original Hugging
Face model:
    python run_quality_eval.py --model hf \\
        --model-args pretrained=HuggingFaceTB/SmolLM2-135M-Instruct \\
        --tasks ifeval --limit 20 --output float_ifeval.json

Run the quantized side (macOS only, real Core ML) against the exported model:
    python run_quality_eval.py --model coreml \\
        --model-args pretrained=smollm2.mlpackage \\
        --tasks ifeval --limit 20 --output coreml_ifeval.json

Then compare the two with `compute_retention.py`.
"""

from __future__ import annotations

import argparse
import json
import sys


def run_quality_eval(
    model: str,
    model_args: str,
    tasks: list[str],
    limit: int | None,
    num_fewshot: int | None,
    apply_chat_template: bool,
    max_gen_toks: int | None = None,
) -> dict:
    if model == "coreml":
        import lm_eval_coreml_adapter  # noqa: F401  registers the "coreml" model

    from lm_eval.evaluator import simple_evaluate

    results = simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=tasks,
        limit=limit,
        num_fewshot=num_fewshot,
        # Overrides every task's own generation_kwargs -- caps a benchmark's
        # potentially long default (MMLU-Pro's is 2048) uniformly, which
        # matters a lot more here than on a batched GPU eval: each token is
        # its own single-token forward pass on an unbatched decoder (HF on
        # CPU, or a real Core ML .mlpackage), so generation length dominates
        # wall-clock cost. See the module docstring.
        gen_kwargs={"max_gen_toks": max_gen_toks} if max_gen_toks else None,
        # The float side always runs on CPU (that's the point -- no macOS needed
        # for it, see the module docstring); left unset for "coreml", where
        # device isn't a concept CoreMLLM has (Core ML picks its own compute
        # unit -- see CoreMLDecoder's compute_units instead). Without this,
        # transformers' own device-map auto-detection can misbehave in a
        # CPU-only environment (observed: an AssertionError from
        # torch.cuda.current_device() during weight loading, even though
        # torch.cuda.is_available() is False) -- forcing "cpu" here sidesteps it.
        device="cpu" if model == "hf" else None,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=apply_chat_template,
        batch_size=1,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate() returned no results")

    return {
        "model": model,
        "model_args": model_args,
        "limit": limit,
        "num_fewshot": num_fewshot,
        "tasks": {
            task: {
                metric: value
                for metric, value in metrics.items()
                # lm_eval's own "metric_name,filter_name" convention for an actual
                # scored metric -- distinguishes it from bookkeeping fields on the
                # same dict ("alias", "sample_len", "name", none of which contain
                # a comma) and from each metric's paired stderr entry
                # ("metric_stderr,<filter_name>" -- the filter name varies per
                # task, e.g. MMLU-Pro's "custom-extract", not always "none", and
                # is itself sometimes the non-numeric string "N/A").
                if "," in metric and "_stderr," not in metric
            }
            for task, metrics in results["results"].items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model", required=True, choices=["hf", "coreml"], help="lm_eval model backend"
    )
    ap.add_argument(
        "--model-args",
        required=True,
        help="lm_eval model_args, e.g. 'pretrained=HuggingFaceTB/SmolLM2-135M-Instruct' "
        "(--model hf) or 'pretrained=model.mlpackage' (--model coreml)",
    )
    ap.add_argument(
        "--tasks",
        default="ifeval",
        help="Comma-separated lm_eval task names (default: ifeval). Other targets: "
        "hendrycks_math500 (MATH-500), mmlu_pro_<subject> (one MMLU-Pro subject; the "
        "full 'mmlu_pro' group is 14 subjects x 5-shot, likely too slow for a --limit "
        "run against an unbatched Core ML decoder)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of examples per task (default: 20). Not a benchmark-grade score "
        "at this size -- see the module docstring.",
    )
    ap.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Override each task's default few-shot count (default: use the task's own).",
    )
    ap.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Format prompts through the model's chat template (recommended for "
        "instruct models; lm_eval warns if this is left off for one).",
    )
    ap.add_argument(
        "--max-gen-toks",
        type=int,
        default=None,
        help="Cap every task's generation length (default: each task's own, e.g. "
        "MMLU-Pro's 2048) -- see the module docstring on why this matters more here "
        "than on a batched GPU eval.",
    )
    ap.add_argument(
        "--output", help="Write the JSON summary here (default: stdout only)"
    )
    args = ap.parse_args()

    result = run_quality_eval(
        args.model,
        args.model_args,
        args.tasks.split(","),
        args.limit,
        args.num_fewshot,
        args.apply_chat_template,
        args.max_gen_toks,
    )

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
