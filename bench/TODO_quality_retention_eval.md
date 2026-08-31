# Open: quality + retention measurement for the Core ML LLM benchmark suite

**Status:** planning only -- nothing in this doc is implemented yet. `scripts/apple`
currently measures decode speed (`run_llm_decode_benchmark.py`) and prefill/decode
correctness against the HF reference at the token level (`check_decode_parity.py`,
agreement rate + first divergence). It has no way to answer "is the model still
*good*" -- accuracy on a real benchmark -- or "how much accuracy did quantization/
fp16 conversion cost" -- the two other axes
[DeviceMark](https://devicemark.github.io/)'s leaderboard reports alongside speed and
memory. This doc lays out an approach for both, sized to what this repo's CI can
actually run, without committing to an implementation yet.

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
job. Two knobs to make it CI-feasible without abandoning the same benchmarks:

1. **Subset, not full set.** A fixed random N-sample subset (e.g. 50-100 prompts) per
   benchmark, seeded for reproducibility. Reports a noisier but directionally useful
   score, not a leaderboard-grade number -- call this out explicitly in any output
   ("N=100 subset, not the full benchmark") so it's never mistaken for a comparable
   DeviceMark score.
2. **Float side never needs macOS.** Retention's denominator (`float_accuracy`) only
   needs the original Hugging Face model running through `transformers` on CPU --
   exactly what `check_decode_parity.py`'s reference half already does. That run can
   happen anywhere (including this dev sandbox, no Core ML/macOS needed), leaving only
   the numerator (`quantized_accuracy`, the Core ML `.mlpackage`) as the part that must
   run on a macOS runner. Splitting the two lets the float-side eval iterate fast in a
   normal Linux CI job or locally, independent of macOS runner minutes.

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
  `lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={model_id}", tasks=[...])`.
- **Quantized side:** a new adapter, something like `scripts/apple/lm_eval_coreml_adapter.py`,
  wrapping `CoreMLDecoder` (from `run_llm_decode_benchmark.py`, already handles
  prefill + growing-KV-cache decode) to implement `generate_until` (IFEval and
  MATH-500 are generation tasks: free-form / chain-of-thought responses scored by a
  verifier or answer-extraction, not by comparing log-likelihoods of fixed
  continuations). MMLU-Pro is multiple-choice and could use either `generate_until`
  (generate, then extract the letter) or `loglikelihood` (score each candidate
  continuation) depending on which the harness's stock MMLU-Pro task expects. Whether
  `loglikelihood` is reachable through `CoreMLDecoder`'s current single-token-at-a-time
  interface (it would need per-candidate teacher-forced logprobs, not just greedy
  argmax) is an open question the "Next steps" below need to resolve before committing
  to `lm-evaluation-harness` over a lighter custom scorer for MMLU-Pro specifically.

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

1. Re-read DeviceMark's actual methodology page (subset sizes if any, exact benchmark
   versions/splits, how they define retention) from an environment that isn't blocked
   from reaching `devicemark.github.io`, and reconcile any difference from this doc's
   recalled description before implementing.
2. Prototype the float-side eval first (no macOS needed): `lm_eval.simple_evaluate`
   against one already-validated model (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) on
   a small IFEval subset, to confirm the harness's task definitions and scoring run
   cleanly in this repo's dependency set before touching the Core ML side at all.
3. Resolve the `loglikelihood`-vs-`generate_until` question for MMLU-Pro against
   `CoreMLDecoder`'s interface (see "Harness choice" above) -- this decides whether
   `CoreMLDecoder` needs a new teacher-forced-logprob code path or whether
   `generate_until`-based answer extraction is good enough.
4. Build the `CoreMLDecoder`-wrapping `lm_eval` adapter and validate it end-to-end
   against one small model + one benchmark subset on a macOS runner, comparing its
   score to the float side's score on the same subset (sanity: they should be close
   for an unquantized-in-spirit fp16 Core ML model, not wildly different).
5. Wire a small-subset run into a new `workflow_dispatch`/schedule-only CI job (own
   job, not folded into `benchmark-decode-macos`, given the likely runtime), reporting
   per DeviceMark's three benchmarks plus retention, once 2-4 are validated.
