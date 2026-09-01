# Open: quality + retention measurement for the Core ML LLM benchmark suite

**Status:** first slice implemented -- IFEval only, both directions (float + quantized
eval, retention computation), wired into a `workflow_dispatch`/schedule-only CI job.
`scripts/apple` now has `run_quality_eval.py` (an `lm-evaluation-harness` wrapper
supporting `--model hf` and `--model coreml`), `lm_eval_coreml_adapter.py` (the
`CoreMLDecoder`-wrapping `generate_until` adapter), and `compute_retention.py`
(quantized/float ratio, unit-tested in `tests/test_compute_retention.py`) -- see
`scripts/apple/README.md`'s "Quality and retention eval" section for usage. MMLU-Pro
and MATH-500 are validated as *working* through the same scripts (see "What's been
prototyped" below) but not yet in CI: a real compute-cost finding during prototyping
(below) means they need more thought before they're CI-feasible at all, not just a
config change. This doc's "Next steps" now reflects what's left, not a from-scratch
plan.

`scripts/apple` previously only measured decode speed (`run_llm_decode_benchmark.py`)
and prefill/decode correctness against the HF reference at the token level
(`check_decode_parity.py`, agreement rate + first divergence) -- it had no way to
answer "is the model still *good*" -- accuracy on a real benchmark -- or "how much
accuracy did quantization/fp16 conversion cost" -- the two other axes
[DeviceMark](https://devicemark.github.io/)'s leaderboard reports alongside speed and
memory.

## What DeviceMark measures (recalled, not re-fetched -- see the caveat below)

Quality: IFEval (instruction-following), MMLU-Pro (broad knowledge, harder
multiple-choice than plain MMLU), MATH-500 (grade-school-through-competition math).
Retention: `quantized_accuracy / float_accuracy` on the same benchmark(s) --
how much of the float model's capability survives quantization to the on-device
format. Both are computed per-model, alongside decode tok/s and memory, feeding one
leaderboard row per model.

**Caveat:** `devicemark.github.io` is blocked by this environment's outbound network
policy, so this section is from earlier context in this work, not a fresh read of
their methodology page. Re-check the actual page (benchmark set, subset sizes,
scoring exactly as they define it) before implementing, from an environment that can
reach it.

## Proposed scope for this repo

Full IFEval/MMLU-Pro/MATH-500 runs are thousands of prompts each -- at this suite's
current unbatched, single-sequence-at-a-time Core ML decode design (see
`run_llm_decode_benchmark.py`'s module docstring), that's hours per model on a
GitHub-hosted macOS runner, not viable for even a `workflow_dispatch`/schedule-only
job. Two knobs make it CI-feasible without abandoning the same benchmarks -- both now
implemented as `run_quality_eval.py` flags:

1. **Subset, not full set (`--limit`).** A fixed N-sample subset per benchmark
   (`quality-eval-macos`'s CI job uses 10). Reports a noisier but directionally useful
   score, not a leaderboard-grade number -- `lm_eval` itself warns about this at the
   CLI level, and `run_quality_eval.py`'s module docstring repeats the warning; never
   mistake it for a comparable DeviceMark score.
2. **Float side never needs macOS.** Retention's denominator (`float_accuracy`) only
   needs the original Hugging Face model running through `transformers` on CPU --
   exactly what `check_decode_parity.py`'s reference half already does, and what
   `run_quality_eval.py --model hf` does (forces `device=cpu` explicitly -- see that
   script for why). That run can happen anywhere (including a token-less dev sandbox,
   no Core ML/macOS needed), leaving only the numerator (`quantized_accuracy`, the
   Core ML `.mlpackage`, `--model coreml`) as the part that must run on a macOS runner.
   `quality-eval-macos` still runs both sides in the same job for simplicity (it needs
   the macOS runner for the Core ML half anyway), but nothing stops splitting the float
   side into a cheaper Linux job later if macOS runner minutes become the bottleneck.
3. **Generation length cap (`--max-gen-toks`), added after prototyping.** Not
   anticipated in the original version of this plan -- see "What's been prototyped"
   below for why it turned out to matter as much as `--limit`.

## Harness choice

[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
(EleutherAI) already implements IFEval, MMLU-Pro, and MATH-500 as tasks --
prompt templates, few-shot formatting, and (critically) the scoring/parsing logic
per benchmark (see "Per-benchmark scoring" below), which would otherwise be a
substantial reimplementation. It works against any model exposing its
`lm_eval.api.model.LM` interface (primarily `loglikelihood`/`generate_until`), so the
integration point is a small custom adapter class, not a fork of the harness itself:

- **Float side:** `lm_eval`'s existing `hf` model type already wraps
  `transformers.AutoModelForCausalLM` directly -- no adapter needed, just
  `run_quality_eval.py --model hf --model-args pretrained=<model_id>`.
- **Quantized side:** `scripts/apple/lm_eval_coreml_adapter.py`, wrapping
  `CoreMLDecoder` (from `run_llm_decode_benchmark.py`, already handles prefill +
  growing-KV-cache decode) to implement `generate_until`.
- **Resolved (was an open question in the original version of this plan):** does
  MMLU-Pro need `loglikelihood` (per-candidate teacher-forced logprobs), which
  `CoreMLDecoder`'s greedy `generate()` can't produce? No -- checked directly against
  the installed `lm-eval==0.4.12`: `mmlu_pro`'s task YAML (`_default_template_yaml`)
  declares `output_type: generate_until`, same as `ifeval` and `hendrycks_math500`
  (MATH-500). All three of this plan's target benchmarks are `generate_until` in this
  harness version, so `lm_eval_coreml_adapter.py` only implements that method;
  `loglikelihood`/`loglikelihood_rolling` raise if a future task actually needs them.
  (This is a property of the installed harness version's task definitions, not a law of
  nature -- worth a quick re-check of the relevant task YAML if `lm_eval` is ever
  upgraded across a version that might change it.)

Alternative considered: a from-scratch minimal harness (just load each benchmark's
public dataset, format prompts, generate, score). Rejected as the default plan --
scoring IFEval's programmatic instruction verifiers and MATH-500's answer-equivalence
correctly is fiddly and already solved in `lm-evaluation-harness`; reimplementing it
risks silently-wrong scores that look plausible. Worth falling back to only if the
harness's model-interface assumptions turn out to be a bad fit for
`CoreMLDecoder`'s unbatched, growing-KV-cache shape.

## Per-benchmark scoring considerations

- **IFEval:** programmatic verifiers (e.g. "response contains exactly 3 bullet
  points", "response is under N words") applied to the raw generated text -- no
  external judge model needed, deterministic once generation is done. Sensitive to
  generation length/truncation; `--max-new-tokens` needs to be generous enough that
  legitimate compliant responses aren't cut off and marked failed for the wrong
  reason.
- **MMLU-Pro:** multiple-choice with up to 10 options (vs. plain MMLU's 4) -- answer
  extraction from generated text (if using `generate_until`) needs to reliably find
  the model's chosen letter, which is more failure-prone with more options and with a
  small/lightly-instruction-tuned model (this suite's models are mostly 1-4B). Worth
  checking the harness's built-in MMLU-Pro answer-extraction regex against a few
  sample generations from a small model before trusting the aggregate score.
- **MATH-500:** free-form final-answer extraction + symbolic/numeric equivalence
  checking (e.g. `1/2` == `0.5` == `\frac{1}{2}`) -- `lm-evaluation-harness`'s task
  implementation (or the original MATH repo's `is_equiv`) handles this; a naive
  string-equality check would undercount correct answers in different but equivalent
  forms.

## What's been prototyped

All three target benchmarks were run end-to-end through `run_quality_eval.py --model
hf` against `HuggingFaceTB/SmolLM2-135M-Instruct` (CPU, this dev sandbox -- the
smallest, fastest-iterating already-validated model) to confirm the harness's task
definitions and this repo's dependency set actually work together, not just as a
design on paper:

- **IFEval:** works end-to-end; `quality-eval-macos`'s CI job runs this one.
- **`hendrycks_math500` (MATH-500):** works end-to-end, and was noticeably faster per
  example than MMLU-Pro at the same `--limit` (no explicit `max_gen_toks` in its task
  YAML, so it relies on the model naturally stopping rather than a large fixed budget).
- **`mmlu_pro_biology` (one MMLU-Pro subject, not the full 14-subject group):** works
  end-to-end, but is dramatically more expensive than the other two --
  **~2 minutes for a single example** at `--limit 1` (5-shot context, 2048-token
  `max_gen_toks`) against a 135M-parameter model on CPU. This is the finding that added
  `--max-gen-toks` to `run_quality_eval.py`: a benchmark's default generation budget
  directly sets wall-clock cost at this suite's unbatched-decode scale, and MMLU-Pro's
  default is an order of magnitude larger than the other two. Whether capping
  `--max-gen-toks` low enough to make MMLU-Pro CI-feasible also lets the model actually
  finish its "think step by step, then answer (X)" reasoning before being cut off (and
  so still score meaningfully) is untested -- see "Next steps".
- **Not yet run at all against the quantized (Core ML) side** -- everything above only
  exercised `--model hf`; `--model coreml` (`lm_eval_coreml_adapter.py`) has only been
  import/registration-checked (confirms `@register_model("coreml")` resolves and the
  class instantiates the right base class), not run against a real `.mlpackage` on real
  Core ML, since prototyping happened in a Linux dev sandbox with no macOS runtime.
  `quality-eval-macos`'s first real CI run is this adapter's first real-Core-ML test.
- A `run_quality_eval.py --model hf` call needs `device="cpu"` forced explicitly
  (`run_quality_eval.py` does this automatically for `--model hf`) -- without it,
  `transformers`' newer device-map auto-detection path
  (`caching_allocator_warmup` -> `torch.cuda.current_device()`) raised
  `AssertionError: Torch not compiled with CUDA enabled` in this CPU-only sandbox, even
  though `torch.cuda.is_available()` correctly reports `False` elsewhere. Not filed
  upstream; worked around locally since it's one explicit kwarg.

## Retention computation

```
retention = quantized_accuracy / float_accuracy   # per benchmark, per model
```

Both accuracies from the *same* subset (identical sampled prompts, identical scoring
code) so the ratio isolates the quantized-vs-float gap rather than sampling noise
between two different subsets. `float_accuracy` computed once per model (independent
of Core ML, reusable across dtype variants if this suite ever benchmarks int8/int4
Core ML variants of the same model). A ratio > 1.0 is possible on a small subset (both
sides have noise) and isn't a bug -- worth clamping/annotating rather than treating as
an error.

## Reporting

Following `run_llm_decode_benchmark.py`/`check_decode_parity.py`'s existing pattern:
print a human-readable summary and let the CI job `tee` it into
`$GITHUB_STEP_SUMMARY` (decode tok/s, memory, parity agreement, and now per-benchmark
quality + retention, all in one place per model). A machine-readable JSON artifact
alongside it (one object per model: `{model_id, benchmark, subset_n, quantized_acc,
float_acc, retention}`) would let a later step turn a run history into a trend
without re-parsing log text -- worth adding once there's more than one data point to
actually compare.

## Next steps, if picking this up

1. ~~Prototype the float-side eval first~~ -- done, see "What's been prototyped": all
   three target benchmarks confirmed working end-to-end via `run_quality_eval.py
   --model hf`.
2. ~~Resolve the `loglikelihood`-vs-`generate_until` question for MMLU-Pro~~ -- done,
   see "Harness choice": resolved to `generate_until` for all three benchmarks against
   the installed harness version, no teacher-forced-logprob path needed.
3. ~~Build the `CoreMLDecoder`-wrapping `lm_eval` adapter~~ -- done
   (`lm_eval_coreml_adapter.py`), but **not yet validated against real Core ML** --
   only import/registration-checked in a non-macOS sandbox so far. Confirming it
   end-to-end (does `quality-eval-macos` actually pass, does the Core ML side's IFEval
   score land in a sane range relative to the float side's on the same subset) is the
   most valuable next check once this doc's changes reach a macOS CI run.
4. ~~Wire a small-subset run into a new CI job~~ -- done (`quality-eval-macos`), but
   **IFEval only**. MMLU-Pro and MATH-500 remain future work:
   - MMLU-Pro's ~2-minutes-per-example cost (see "What's been prototyped") needs
     either a much lower `--max-gen-toks` (with a check that a still-useful score comes
     out -- an aggressively truncated "think step by step" response might just always
     score 0, which would be worse than not running it at all) or accepting a very
     small `--limit` (1-3 examples) purely as a smoke test rather than a real quality
     signal.
   - MATH-500 looked cheaper in the prototype and is probably the more promising second
     benchmark to add to `quality-eval-macos` -- worth trying before MMLU-Pro.
   - The full `mmlu_pro` group (14 subjects) was never attempted even structurally;
     only one subject (`mmlu_pro_biology`) has been run. Even after solving the
     per-example cost, decide whether CI should sample across all 14 subjects or just a
     couple of representative ones.
5. Re-read DeviceMark's actual methodology page (subset sizes if any, exact benchmark
   versions/splits, how they define retention) from an environment that isn't blocked
   from reaching `devicemark.github.io`, and reconcile any difference from this doc's
   recalled description -- still not done; everything implemented so far is sized by
   this repo's own CI constraints, not verified against DeviceMark's actual
   methodology.
6. Once more than one model has been run through `quality-eval-macos` (or a MATH-500/
   MMLU-Pro variant of it), consider the machine-readable JSON artifact idea in
   "Reporting" below -- not worth building for a single data point.
