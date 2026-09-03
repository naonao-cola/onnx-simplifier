# Open: quality + retention measurement for the Core ML LLM benchmark suite

**Status:** first two slices implemented -- IFEval and MATH-500, both directions
(float + quantized eval, retention computation), wired into a
`workflow_dispatch`/schedule-only CI job. `scripts/apple` now has
`run_quality_eval.py` (an `lm-evaluation-harness` wrapper supporting `--model hf`
and `--model coreml`), `lm_eval_coreml_adapter.py` (the `CoreMLDecoder`-wrapping
`generate_until` adapter), and `compute_retention.py` (quantized/float ratio,
unit-tested in `tests/test_compute_retention.py`) -- see
`scripts/apple/README.md`'s "Quality and retention eval" section for usage.
MMLU-Pro is validated as *working* through the same scripts (see "What's been
prototyped" below) but not yet in CI: a real compute-cost finding during
prototyping (below) means it needs more thought before it's CI-feasible at all,
not just a config change. This doc's "Next steps" now reflects what's left, not a
from-scratch plan.

`scripts/apple` previously only measured decode speed (`run_llm_decode_benchmark.py`)
and prefill/decode correctness against the HF reference at the token level
(`check_decode_parity.py`, agreement rate + first divergence) -- it had no way to
answer "is the model still *good*" -- accuracy on a real benchmark -- or "how much
accuracy did quantization/fp16 conversion cost" -- the two other axes
[DeviceMark](https://devicemark.github.io/)'s leaderboard reports alongside speed and
memory.

## What DeviceMark measures (verified against methodology.html, 2026-09-03)

Confirmed by actually fetching `devicemark.github.io/methodology.html` (plain
`curl` reaches it fine over this environment's outbound proxy even though the
WebFetch tool's own network path returns `EGRESS_BLOCKED` for the domain --
worth remembering next time this looks unreachable). This section replaces the
earlier "recalled, not re-fetched" version; item 5 in "Next steps" below is
now done.

**Quality battery ("v0", `battery_version`):** IFEval, MMLU-Pro, MATH-500 --
same three benchmark *names* this doc already assumed, but not full runs of
any of them: DeviceMark samples a fixed **596-item battery** ("full596") --
**300 IFEval items** (of `google/IFEval`'s 541), **196 MMLU-Pro items**
(stratified over its 14 categories), **100 MATH-500 items** (of
`HuggingFaceH4/MATH-500`'s 500, stratified over 7 subjects) -- not the full
public benchmark sizes, and not the 5-shot MMLU-Pro this doc originally
assumed while prototyping: DeviceMark runs **0-shot**, chat template,
"thinking" OFF, **greedy, cap = 4096 output tokens**, and scores a
no-answer-within-budget response as **wrong** (kept in the denominator, so a
model can't gain by giving up). Scoring: official vendored google-research
IFEval checkers (mean of strict+loose, "mean-of-4"), `\boxed{letter}`
extraction for MMLU-Pro, `\boxed{}` + sympy symbolic equality for MATH-500.
GPQA-Diamond is deliberately excluded from this weight tier (floor effect at
<=1B, plus gated ToS).

**Retention:** `accuracy(int8) / accuracy(float baseline)`, per benchmark --
matches this doc's original formula, but on a subtler accuracy than plain
`acc`: DeviceMark reports **two** accuracies per benchmark --
`acc` (no-answer counts as wrong -- the "is this usable on-device" number) and
`acc_completed` (only items that produced an answer -- cap-independent).
**Retention is computed on `acc_completed`**, specifically because the
int8 and float sides hit the token cap at different points (different
weights, different no-answer rates), which would otherwise let the ratio be
dominated by cap-timing rather than actual quantization damage -- documented
on their side as "the retention confound," and it's why a retention number can
land above 100% at small n. `compute_retention.py`'s
`retention = quantized_score / float_score` on whatever `run_quality_eval.py`
already scored (no completed-only split -- see "Proposed scope" below) is the
same *shape* of ratio, on a coarser accuracy definition.

**The bigger gap: DeviceMark's on-device runtime is not Core ML.** Its
"shipped int8" column runs on the leaderboard author's own private on-device
LLM engine ("Core AI", `format=aimodel`/`runtime=coreai`, from
[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)) -- a
completely different inference stack from Apple's `CoreML.framework`/
coremltools that this repo's whole `scripts/apple` pipeline targets. (Two
rows on the current board use other native runtimes instead: Gemma 4 E2B
measures on Google's LiteRT-LM, and Apple's built-in Foundation Model measures
through its own `SystemLanguageModel` API -- neither is Core AI or Core ML
either.) So "the same two axes DeviceMark measures" (decode tok/s, memory) was
always true as a *description of what to measure*, but any of this repo's own
`iphone_tok_s`-shaped numbers are a different runtime's numbers on different
models, not a comparable point on DeviceMark's board -- see the model-roster
note below for how different.

**DeviceMark's actual model roster does not overlap with this repo's
`BENCHMARK_MODELS`.** As of the same fetch, the board's device-measured rows
are Qwen3.5-0.8B/2B/4B, LFM2.5-1.2B, Granite-4.0-H-1B, Youtu-LLM-2B (Tencent),
Nemotron-3-Nano-4B (NVIDIA, Mamba2 hybrid), Nanbeige4.1-3B, and Gemma 4 E2B
(5.4B raw / ~2B effective, QAT int4) -- plus Apple's built-in Foundation Model
and two cloud reference lines (Gemini Flash/Pro, not ranked). None of these
are SmolLM2, Qwen2.5, Llama-3.2, or Phi-3.5-mini -- this repo's own model list
targets the same rough *weight class* (DeviceMark's rows span roughly
0.8-5.4B raw params) and a spread of architecture families for translator
coverage, not literal reproduction of DeviceMark's specific rows. Decode speed
is measured on an **iPhone 17 Pro** (and an M4 Max Mac as a faster proxy for
rows still Mac-only), **warm-state** (engine loaded and specialized, cold
load/first-run excluded), at a **128-token prompt / 256-token decode**
protocol, two trials on a settled device -- this repo's own
`run_llm_decode_benchmark.py`/CI matrix uses a much shorter prompt (~5 tokens)
and `--max-new-tokens 20`, single trial, no disclosed thermal/settle
protocol -- a real, disclosed difference in rigor and scale, not a match to
DeviceMark's own numbers.

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
  `quality-eval-macos`'s CI job now runs this one too (re-confirmed working,
  ~7s/example on CPU at `--limit 3`, before wiring it in).
- **`mmlu_pro_biology` (one MMLU-Pro subject, not the full 14-subject group):** works
  end-to-end, but is dramatically more expensive than the other two --
  **~2 minutes for a single example** at `--limit 1` (5-shot context, 2048-token
  `max_gen_toks`) against a 135M-parameter model on CPU. This is the finding that added
  `--max-gen-toks` to `run_quality_eval.py`: a benchmark's default generation budget
  directly sets wall-clock cost at this suite's unbatched-decode scale, and MMLU-Pro's
  default is an order of magnitude larger than the other two.
  Follow-up (2026-09-03): capping `--max-gen-toks` **does** fix the cost --
  `--max-gen-toks 256` against `HuggingFaceTB/SmolLM2-135M-Instruct` (the exact model
  `quality-eval-macos` uses) brought `mmlu_pro_biology` down to **~14s/example**
  (`--limit 10` took ~2m38s wall-clock including model load, vs. the ~20min the 2048
  default would extrapolate to), an 8x+ speedup, comparable in order of magnitude to
  MATH-500's ~7s/example. But `--limit 10 --max-gen-toks 256` (and separately, `512`)
  both scored **`exact_match: 0.0` -- 0/10** for this model. Checked with
  `log_samples=True` whether that's the truncation artifact this doc originally
  worried about ("an aggressively truncated response might just always score 0"): it
  is not -- every sampled generation finished well under either cap (150-1355 chars,
  nowhere near 256 or 512 tokens) with a complete "The answer is (X)" pattern (or, for
  3/10 at the 256 cap, a repetition loop that never produced one, correctly scored
  `[invalid]`/wrong by the harness's own extractor); the model just doesn't answer
  10-way MCQ biology correctly at 135M params, matched by 512-token responses showing
  the same coherent-but-wrong reasoning. So the fix works, but exposes a *different*
  blocker for wiring this into CI -- see "Next steps".
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
alongside it -- `compute_retention.py --output`'s `"records"` list, one object per
(task, metric): `{model_id, benchmark, metric, subset_n, float_acc, quantized_acc,
retention}` (`build_records()`, unit-tested in `test_compute_retention.py`) --
now exists and is uploaded from `quality-eval-macos` as the `quality-retention-results`
artifact, one JSON per benchmark (IFEval, MATH-500). Lets a later step turn a run
history into a trend without re-parsing log text; nothing reads these back yet (no
trend-plotting script exists) -- that's the next piece if this is picked up further.

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
4. ~~Wire a small-subset run into a new CI job~~ -- done (`quality-eval-macos`).
   ~~MATH-500 stays future work~~ -- done: `hendrycks_math500` (same
   `HuggingFaceH4/MATH-500` source DeviceMark's own battery draws from) is now a
   second benchmark in `quality-eval-macos`, `--limit 10 --max-gen-toks 256`, its own
   float/coreml/retention step trio mirroring IFEval's. Re-confirmed cheap
   (~7s/example on CPU at `--limit 3` in this environment, matching the earlier
   prototype) before wiring it in; `sympy` (its answer-equivalence checker's direct
   import, not one of `lm-eval`'s own declared dependencies) added explicitly to the
   job's pip install line rather than relying on it arriving transitively via
   `optimum-onnx`'s torch dependency. MMLU-Pro remains future work, now for a
   different reason than originally thought:
   - ~~Its ~2-minutes-per-example cost needs a much lower `--max-gen-toks`~~ -- tried
     (2026-09-03): `--max-gen-toks 256` against `HuggingFaceTB/SmolLM2-135M-Instruct`
     (the exact model `quality-eval-macos` runs) cuts `mmlu_pro_biology` to
     ~14s/example (`--limit 10` in ~2m38s wall-clock incl. model load), a similar
     order of magnitude to MATH-500 -- solves the cost problem, and verified (via
     `log_samples=True` on the raw generations) that this isn't just cutting
     responses off mid-answer: every sample finished naturally well under the cap.
   - **New blocker found instead:** at that same `--limit 10`, this model's *float*
     score on `mmlu_pro_biology` is **0/10** (also 0/10 at `--max-gen-toks 512`,
     ruling out the cap as the cause -- the raw generations are complete, coherent,
     and simply wrong, or end in a repetition loop with no answer). A float score of
     exactly 0 makes `compute_retention.py`'s `quantized / float` ratio undefined by
     design (reported as `None`/"N/A (float score was 0)", not `inf`) -- so wiring
     MMLU-Pro into `quality-eval-macos` as currently configured (same tiny model, same
     `--limit`) would spend CI time every run for a retention number that's *always*
     "N/A", not a useful signal. Options, untried: (a) a noticeably more capable model
     in the CI matrix for this one benchmark (135M params may just be under the floor
     for 10-way MCQ MMLU-Pro -- harder than plain MMLU by design), (b) a much larger
     `--limit` to reduce the chance of an all-zero small sample (works against the
     CI-budget rationale for `--limit` in the first place, and untested whether it
     actually gets off zero for this model), or (c) picking a different, possibly
     easier MMLU-Pro subject than biology (untested -- MMLU-Pro's difficulty is
     fairly uniform by design, so this is a low-confidence option).
   - The full `mmlu_pro` group (14 subjects) was never attempted even structurally;
     only one subject (`mmlu_pro_biology`) has been run. Moot until the zero-float-score
     blocker above is resolved -- no point deciding CI subject coverage for a benchmark
     that can't yet produce a non-`None` retention number.
5. ~~Re-read DeviceMark's actual methodology page~~ -- done, see "What DeviceMark
   measures" above (subset sizes, 0-shot not 5-shot, cap=4096, `acc_completed`-based
   retention, and the bigger finding that DeviceMark's own on-device runtime is a
   private "Core AI" engine, not Core ML, so this repo's numbers were never going to be
   directly comparable to a DeviceMark board row regardless of subset size). Nothing in
   this repo's own implementation needed to change as a result -- the CI-feasibility
   constraints (`--limit`, `--max-gen-toks`) that already exist are still the right call
   for a GitHub-hosted macOS runner's budget; this was a documentation-accuracy gap, not
   a scope gap. One thing worth doing if this is picked up again: switch
   `compute_retention.py`'s inputs to a completed-only accuracy (excluding no-answer
   items from the denominator) to match how DeviceMark defines retention specifically,
   rather than the plain `acc` `run_quality_eval.py` currently reports -- not done here
   since it touches `run_quality_eval.py`'s output schema, not just docs.
6. ~~Once more than one model has been run through `quality-eval-macos`... consider
   the machine-readable JSON artifact idea~~ -- done: MATH-500 landing alongside
   IFEval (see item 4) gave a second data point, so `compute_retention.py --output`
   now writes a flat `"records"` list (`build_records()`) and `quality-eval-macos`
   uploads both benchmarks' JSON as the `quality-retention-results` artifact -- see
   "Reporting" above. Nothing consumes these across runs yet (no trend-plotting
   script) -- that would be the next piece once there's an actual run history worth
   trending.
