# onnx-deploy (design sketch)

**Status: design sketch / skeleton, not a shipped tool.** It is not wired into
the top-level build, has not been compiled or run against a real model in
this environment (no ONNX Runtime dev package was available here), and is
not covered by CI. Treat the header/CLI below as a concrete starting point
for the missing piece described here, not a finished library.

## The question this answers

*Is there a C++ library in this repo that glues together the multiple `.onnx`
files `optimum-onnx` exports for one model, so they can be deployed as a
single autoregressive-generation pipeline?*

No. Today that gluing only exists in Python, via `optimum.onnxruntime`'s
`ORTModelForSeq2SeqLM` / `ORTModelForCausalLM` (see the "Transformers export"
section of the top-level `README.md` and `tests/test_optimum_export_deploy.py`,
which drives exactly that Python class end-to-end against onnxsim's output).
onnxsim's own C++ core (`onnxsim/`) is a graph simplifier -- it takes one
`onnx::ModelProto` in and produces a simplified one out. Its only use of the
ONNX Runtime C++ API is `onnxsim/dlpack_bridge.h`'s single-model,
single-call `ModelExecutor::Run()`, used internally for constant folding
during simplification (see `docs/dlpack-executor.md`). It has no notion of
multiple persistent sessions, a KV-cache loop, or tensors flowing from one
model into another -- there is nothing in this repo to build a C++-only
deployment on top of except that low-level executor seam.

This directory sketches what a real one would look like, as an independent
tool (own `CMakeLists.txt`, own binary) in the same spirit as
`tools/onnx-finetune/`: it consumes plain ONNX files, has no dependency on
onnxsim's own build, and is not part of the Python wheel (see the repo's
`CLAUDE.md` note that the wheel never builds or links ONNX Runtime).

## What optimum-onnx actually exports

`optimum.exporters.onnx.main_export(..., task="text2text-generation-with-past")`
(or `main_export(..., no_post_process=True)`, which is what this repo's own
test uses -- see below) writes a *directory*, not one file:

```
encoder_model.onnx             # seq2seq only (T5, BART, Whisper, ...)
decoder_model.onnx             # first decode step -- no KV-cache inputs
decoder_with_past_model.onnx   # every step after -- KV-cache in AND out
config.json, generation_config.json, tokenizer files, ...
```

Decoder-only causal LMs (GPT-2, Llama, ...) export the same
`decoder_model.onnx` / `decoder_with_past_model.onnx` pair with no
`encoder_model.onnx`.

By default, recent `optimum-onnx` merges the two decoder files into one
`decoder_model_merged.onnx` with a top-level `If` node switching branches on
a boolean `use_cache_branch` input. **This sketch deliberately targets the
plain three-file split instead** (`no_post_process=True`), for the same
reason `tests/test_optimum_export_deploy.py` does: as of this writing,
onnxsim simplifying the merged file's `If` branches produces a model that
fails at runtime with an ONNX Runtime broadcast error in cross-attention --
see that test's docstring. The split shape is also simply easier to drive
from C++: two ordinary sessions and an `if (step == 0)` in the host loop,
instead of one session where the same branch selection has to happen
*inside* the graph via an extra input tensor. Supporting the merged shape
later is a matter of adding one more input (`use_cache_branch`, a length-1
bool tensor) to `RunDecoderStep` and pointing both "sessions" at the same
`Ort::Session` -- noted as a follow-up below.

## Design

### The actual "glue": renaming `present.*` outputs into `past_key_values.*` inputs

Every decode step after the first, ONNX Runtime hands back cache tensors
named `present.{i}.key` / `present.{i}.value` (plus, for seq2seq models,
`present.{i}.encoder.key` / `.value` for cross-attention, alongside
`present.{i}.decoder.key` / `.value` for self-attention). The *next* call
needs those same tensors fed back in as `past_key_values.{i}.key` /
`.value` (`.decoder.`/`.encoder.` likewise). That rename is 100% positional
by naming convention -- no shapes, dtypes, or model architecture need to be
known to do it -- so the pipeline never hardcodes layer counts or head
dims; it just maps every output whose name starts with `present.` to the
input name with `past_key_values.` substituted in, for however many such
outputs the loaded graph actually has (`HarvestPresentIntoCache` in the
header below).

Because `Ort::Value` is move-only and owns (or borrows) its own buffer, this
rename is a pointer/ownership move, not a tensor copy: the previous
`Run()` call's output buffer becomes the next call's input directly, with no
round trip through host arrays.

### Two sessions, one host-side loop

```
                 ┌───────────────────┐
input_ids ─────▶ │ encoder_model.onnx│──▶ encoder_hidden_states  (seq2seq only,
                 └───────────────────┘                            run once)
                            │
                            ▼
      ┌─────────────────────────────────────────────┐
      │ step 0: decoder_model.onnx                   │
      │   in : decoder_start_token, enc_hidden_states│
      │   out: logits, present.*                     │──┐
      └─────────────────────────────────────────────┘   │ present.* renamed to
                            │                             │ past_key_values.* for
                            ▼                             │ every step after
      ┌─────────────────────────────────────────────┐   │
      │ step i>0: decoder_with_past_model.onnx       │◀──┘
      │   in : last_token, enc_hidden_states,        │
      │        past_key_values.* (from cache)         │
      │   out: logits, present.* ──────────────────────┐
      └─────────────────────────────────────────────┘   │ loop back in
                            │  argmax(logits) -> token   │
                            ▼                             │
                     append to output ◀───────────────────┘
                     stop at eos_token_id / max_new_tokens
```

A decoder-only export runs the same loop with no encoder step and no
`encoder_hidden_states`/`.encoder.` cache entries -- `is_seq2seq()` on the
pipeline just reflects whether `encoder_model.onnx` was found in the
directory, and `RunDecoderStep` skips the encoder-only inputs when absent.

### What is deliberately out of scope here

- **Tokenization.** The pipeline's `Generate()` takes and returns `int64_t`
  token ids, not strings. Wiring up a real tokenizer (`tokenizer.json`) is a
  separate, orthogonal concern -- see e.g.
  [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp) (a C++
  wrapper over Hugging Face's Rust `tokenizers` crate) as the natural thing
  to link in front of this. Keeping it out lets this stay a small,
  dependency-free header.
- **Sampling strategies.** `ArgmaxLastToken` is greedy-only. Top-k/top-p/
  temperature sampling is a small, independent change to that one function.
- **`config.json` parsing.** Generation parameters (`eos_token_id`,
  `decoder_start_token_id`, `max_new_tokens`) are passed in explicitly via
  `GenerationConfig` rather than read from `generation_config.json`, so this
  header has no JSON dependency at all. A CLI wrapper that does want to read
  them from the export directory can add a small JSON library (e.g.
  vendored `nlohmann/json`, single header) without touching the pipeline
  itself -- everything the pipeline needs to know about tensor shapes and
  layer count it already gets for free from the loaded `Ort::Session`s'
  own input/output metadata, not from `config.json`.
- **Batch size > 1** and **beam search**. The sketch's KV-cache map is keyed
  purely by tensor name, so batching would mainly be a matter of building
  wider input tensors and growing the attention mask per batch row; beam
  search needs an actual reorder-cache-by-beam-index step this sketch
  doesn't have.
- **The merged `decoder_model_merged.onnx` shape** (see above).

## Files

- `include/onnx_deploy/kv_cache_pipeline.h` -- the pipeline itself:
  `KvCachePipeline` loads whichever of `encoder_model.onnx` /
  `decoder_model.onnx` / `decoder_with_past_model.onnx` exist in a directory,
  and exposes `Generate(input_ids, config) -> vector<int64_t>`.
- `src/main.cpp` -- a minimal CLI: `onnx-deploy <export_dir> <id1,id2,...>
  [--max-new-tokens N] [--eos-token-id N] [--decoder-start-token-id N]`,
  prints the generated token ids. No tokenizer -- pipe in ids you got from
  `AutoTokenizer` separately, the same "simplest possible format" choice
  `tools/onnx-finetune` makes for its raw float32 training data.
- `CMakeLists.txt` -- builds against a plain ONNX Runtime C++ package (the
  official prebuilt release tarball, or a `pip install onnxruntime`'s
  bundled headers/lib both work here -- unlike `onnx-finetune`, this does
  *not* need a training-enabled from-source build, since generation only
  needs the ordinary inference C++ API).

## Building (once ONNX Runtime is available)

```sh
# Point at an extracted onnxruntime-linux-x64-<ver>.tgz release, or any
# prefix containing include/onnxruntime_cxx_api.h and lib/libonnxruntime.so
cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.19.2
cmake --build build

python3 -c "
import optimum.exporters.onnx as e
e.main_export('hf-internal-testing/tiny-random-t5', output='t5_export',
               task='text2text-generation-with-past', no_post_process=True)
"
./build/onnx-deploy t5_export 0,100 --max-new-tokens 8 --eos-token-id 1
```
