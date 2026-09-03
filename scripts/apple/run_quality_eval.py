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

Each scored metric in `--output`'s "tasks" field is `{"acc": <plain accuracy,
no-answer-within-budget counts as wrong>}`, plus (only when `--max-gen-toks`
was passed explicitly, so there's a known budget to check a response against)
`"total_n"`, `"completed_n"`, and `"acc_completed"` (accuracy over only the
samples that produced an answer before exhausting that budget) -- DeviceMark's
own retention definition, see `is_completed`'s docstring and
`bench/TODO_quality_retention_eval.md`'s "What DeviceMark measures".
`compute_retention.py` prefers `acc_completed` when present and defined,
falling back to `acc` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def is_completed(response_token_count: int, max_gen_toks: int | None) -> bool | None:
    """Whether a `generate_until` response finished before exhausting its
    generation budget -- DeviceMark's own definition of a benchmark item
    "producing an answer" within its token cap, see
    `bench/TODO_quality_retention_eval.md`'s "What DeviceMark measures".
    Validated empirically (2026-09-03) against real `mmlu_pro_biology`
    generations: every sample whose extracted answer was `lm_eval`'s
    `[invalid]` (couldn't find a parseable answer) had a response token count
    exactly equal to `max_gen_toks`, and every sample with a real extracted
    answer had a strictly lower count -- a response that hit the cap without
    naturally stopping (via `until` or EOS) is indistinguishable in length
    from one that happened to need every last token, but that's the same
    approximation DeviceMark's own cap-based definition makes.

    `None` when `max_gen_toks` isn't known -- no explicit `--max-gen-toks`
    was passed to this run, so there's no fixed budget to compare a
    response's length against (guessing a task's own YAML default isn't
    attempted here) -- callers should skip completed-only accounting
    entirely in that case rather than report a number quietly computed
    against the wrong cap.
    """
    if max_gen_toks is None:
        return None
    return response_token_count < max_gen_toks


def aggregate_completed_only(values: list) -> float | None:
    """Mean of per-sample metric values, restricted by the caller to
    "completed" samples (see `is_completed`) before this is called.

    Flattens any list-valued entries first: IFEval's `inst_level_*_acc`
    metrics are a list of per-instruction booleans *per sample* (a single
    response can satisfy some instructions and not others), not a scalar --
    `lm_eval`'s own aggregation pools every instruction across every sample
    before averaging, so this does the same, just over the completed subset,
    to keep a completed-only IFEval score on the same footing as its
    plain-`acc` counterpart. `None` on an empty list (no completed samples --
    matches `compute_retention.py`'s handling of an undefined ratio for the
    same reason: nothing to average).
    """
    flat = []
    for value in values:
        if isinstance(value, list):
            flat.extend(value)
        else:
            flat.append(value)
    if not flat:
        return None
    return sum(flat) / len(flat)


def _load_tokenizer(model: str, model_args: str):
    from transformers import AutoTokenizer

    pretrained = dict(kv.split("=", 1) for kv in model_args.split(","))["pretrained"]
    if model == "coreml":
        # Tokenizer files sit alongside the .mlpackage, exported by
        # export_llm_to_coreml.py -- same lookup lm_eval_coreml_adapter.py's
        # own CoreMLLM.__init__ uses.
        return AutoTokenizer.from_pretrained(str(Path(pretrained).parent))
    return AutoTokenizer.from_pretrained(pretrained)


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
        # Needed to recover each sample's raw response text below, for
        # completed-only (DeviceMark's acc_completed) accounting -- see
        # is_completed's docstring. Cheap: this is CPU tokenization of
        # already-generated text, not another forward pass.
        log_samples=True,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate() returned no results")

    tokenizer = _load_tokenizer(model, model_args) if max_gen_toks is not None else None

    tasks_out = {}
    for task, metrics in results["results"].items():
        samples = results["samples"].get(task, [])
        scored_metrics = {
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

        task_out = {}
        for metric_full, agg_value in scored_metrics.items():
            entry = {"acc": agg_value}
            if tokenizer is not None:
                metric_name = metric_full.split(",", 1)[0]
                completed_values = []
                completed_n = 0
                for sample in samples:
                    response = sample["resps"][0][0] if sample.get("resps") else ""
                    n_toks = len(
                        tokenizer(response, add_special_tokens=False)["input_ids"]
                    )
                    if is_completed(n_toks, max_gen_toks):
                        completed_n += 1
                        if metric_name in sample:
                            completed_values.append(sample[metric_name])
                entry["total_n"] = len(samples)
                entry["completed_n"] = completed_n
                entry["acc_completed"] = aggregate_completed_only(completed_values)
            task_out[metric_full] = entry
        tasks_out[task] = task_out

    return {
        "model": model,
        "model_args": model_args,
        "limit": limit,
        "num_fewshot": num_fewshot,
        "max_gen_toks": max_gen_toks,
        "tasks": tasks_out,
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
