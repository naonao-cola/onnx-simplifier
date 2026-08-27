# onnx-deploy

A standalone C++ tool (own `CMakeLists.txt`, no dependency on onnxsim's own
build -- see the repo's `CLAUDE.md`) that glues together the multiple `.onnx`
files `optimum-onnx` exports for one model into a single autoregressive
generation pipeline, entirely in C++/Python -- no `optimum`, no `torch`, no
Python required at all for the C++ side.

**Status: built and CI-verified** (`.github/workflows/onnx-deploy.yml`) for
the native CLI, the C ABI, and the Python extension -- see "Verifying the
flow" below for exactly what that CI proves and how to reproduce it
locally. The WASM target described near the end is a design writeup only,
not yet built.

## The question this answers

*Is there a C++ library in this repo that glues together the multiple `.onnx`
files `optimum-onnx` exports for one model, so they can be deployed as a
single autoregressive-generation pipeline -- and can it be built as a
dynamic library / WASM module where the actual ONNX Runtime binary is
swapped at runtime instead of baked in at build time?*

Not before this directory. That gluing previously only existed in Python,
via `optimum.onnxruntime`'s `ORTModelForSeq2SeqLM` / `ORTModelForCausalLM`
(see the "Transformers export" section of the top-level `README.md` and
`tests/test_optimum_export_deploy.py`, which drives exactly that Python
class end-to-end against onnxsim's output). onnxsim's own C++ core
(`onnxsim/`) is a graph simplifier -- it takes one `onnx::ModelProto` in and
produces a simplified one out. Its only use of the ONNX Runtime C++ API is
`onnxsim/dlpack_bridge.h`'s single-model, single-call `ModelExecutor::Run()`,
used internally for constant folding during simplification (see
`docs/dlpack-executor.md`) -- it has no notion of multiple persistent
sessions, a KV-cache loop, or a runtime-swappable ONNX Runtime binary.

## What optimum-onnx actually exports

`optimum.exporters.onnx.main_export(..., task="text2text-generation-with-past",
no_post_process=True)` writes a *directory*, not one file:

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
a boolean `use_cache_branch` input. **This tool deliberately targets the
plain three-file split instead** (`no_post_process=True`), for the same
reason `tests/test_optimum_export_deploy.py` does: as of this writing,
onnxsim simplifying the merged file's `If` branches produces a model that
fails at runtime with an ONNX Runtime broadcast error in cross-attention --
see that test's docstring. Supporting the merged shape later is a matter of
adding one more input (`use_cache_branch`, a length-1 bool tensor) to
`KvCachePipeline::RunDecoderStep` and pointing both "sessions" at the same
`Ort::Session` -- not done here.

## Design

### Layer 1: the KV-cache glue (`include/onnx_deploy/kv_cache_pipeline.h`)

Every decode step after the first, ONNX Runtime hands back cache tensors
named `present.{i}.key` / `present.{i}.value` (plus, for seq2seq models,
`present.{i}.encoder.key` / `.value` for cross-attention alongside
`present.{i}.decoder.key` / `.value` for self-attention). The *next* call
needs those same tensors fed back in as `past_key_values.{i}.key` /
`.value`. `KvCachePipeline::HarvestPresentIntoCache` does that rename by
string substitution alone -- no shapes, dtypes, or layer/head counts are
ever hardcoded, so the same code drives any architecture that follows the
naming convention. Because `Ort::Value` is move-only and owns (or borrows)
its own buffer, the rename is a pointer/ownership move, not a tensor copy.

A subtlety this actually needs to get right (and the toy model below is
specifically built to catch a regression in): some architectures' cache
entries (T5-style cross-attention) are computed once at step 0 and never
re-output by `decoder_with_past_model.onnx` afterward. `RunDecoderStep`
therefore always *borrows* a view of a cache entry for a `Run()` call
(`detail::BorrowView`) rather than moving it out -- an entry that isn't
refreshed by that call's outputs stays owned in the cache, valid for the
next step too.

Two sessions, one host-side loop: encoder once (seq2seq only), then
`decoder_model.onnx` for step 0, then `decoder_with_past_model.onnx` in a
loop with the cache fed back each time, greedy-argmax over `logits` each
step, until `eos_token_id` or `max_new_tokens`.

### Layer 2: the swappable-libort C ABI (`onnx_deploy_c_api.h` / `.cpp`)

`kv_cache_pipeline.h` builds against ONNX Runtime's C++ API with
`ORT_API_MANUAL_INIT` defined, which means the ORT function table
(`Ort::Global<void>::api_`) is **not** resolved at static-init time by a
linked `OrtGetApiBase()` call -- nothing in this header, or in
`onnx_deploy_c_api.cpp`, references any symbol from libonnxruntime at link
time at all. `onnx_deploy_load_ort(libort_path)` resolves the real thing at
*runtime*: `dlopen`/`LoadLibrary` the given `libonnxruntime.so`/`.dylib`/
`.dll`, `dlsym`/`GetProcAddress` its `OrtGetApiBase` export, call
`GetApi(ORT_API_VERSION)`, and hand the resulting `OrtApi*` to
`Ort::InitApi()`. This is ONNX Runtime's own documented mechanism for
"custom operator libraries that are not linked to onnxruntime" (see the
comment above `Ort::InitApi(const OrtApi*)` in `onnxruntime_cxx_api.h`),
applied to the whole pipeline instead of just a custom op.

The payoff: `libonnx_deploy_c.so` builds and links with **zero** dependency
on any specific ONNX Runtime binary -- `ldd` on it shows only libstdc++/
libgcc_s/libc/libm, confirmed in CI (see below). The same compiled artifact
can be pointed at a CPU build, a GPU/EP-specific build, or a newer/older
version, by passing a different path to `onnx_deploy_load_ort` -- no
recompile. (ORT's C API is itself only forward-compatible in the
"newer .so serves an older API version request" direction, not the other
way around -- see "Verifying the flow" below.)

The ABI's shape mirrors onnxsim's own C ABI conventions
(`onnxsim/capi/onnxsim_c_api.h`): every fallible call returns an
`OnnxDeployStatus`, takes a nullable `char** out_error` for a freshly
`malloc`'d message on failure, and no C++ exception ever crosses the
`extern "C"` boundary.

### Layer 3: consumers

- **`src/main.cpp`** (`onnx-deploy` CLI) -- a thin consumer of the C ABI
  above, included specifically so building/running it is also an
  executable smoke test of the ABI itself, not just of the C++ core it
  wraps: `onnx-deploy --libort PATH <export_dir> <id1,id2,...>
  [--max-new-tokens N] [--eos-token-id N] [--decoder-start-token-id N]`. No
  tokenizer -- pipe in ids you got from `AutoTokenizer` separately, the same
  "simplest possible format" choice `tools/onnx-finetune` makes for its raw
  float32 training data.
- **`python/onnx_deploy_py.cc`** -- a compiled [nanobind](https://github.com/wjakob/nanobind)
  extension over the same C ABI (not over `kv_cache_pipeline.h` directly, so
  it has no ONNX Runtime header dependency of its own), following the same
  nanobind convention as onnxsim's own `onnxsim/cpp2py_export.cc`. See "Why
  a Python extension at all" below.

## Why a Python extension at all, when `optimum.onnxruntime` already does this in Python

`optimum.onnxruntime.ORTModelForSeq2SeqLM`/`ORTModelForCausalLM` is pure
Python calling `onnxruntime`'s Python bindings: the encoder/decoder session
objects, the KV-cache dict, and the `generate()` loop are all live Python
objects and bytecode, importable, `inspect`-able, and monkeypatchable at
runtime, and the `.onnx` files it reads sit on disk as plain files next to
it. A compiled extension like `onnx_deploy_py` moves the loop, the
cache-tensor threading, and the session lifetime entirely into compiled
machine code reachable from Python only through the four calls
`onnx_deploy_py` exports (`load_ort`, `Pipeline`, `.generate`,
`.is_seq2seq`) -- there's no Python-level loop to monkeypatch, no
`ORTModelForSeq2SeqLM.generate` to trace through with `inspect`/`dis`, and
nothing about the decode loop's logic visible to `pip show`/source-reading
the way `optimum`'s own `.py` files are. In that narrow sense, yes: a
compiled extension is more opaque than the pure-Python glue, and if the
goal is specifically "make the *generation loop* harder to casually read or
patch," this is a real step up from `optimum`.

Two things it deliberately does **not** do, worth being clear about before
reaching for it as an "obfuscation" tool:

- **It doesn't protect the model weights themselves.** The `.onnx` files
  (and their external-data weight files) still sit on disk, in plain ONNX
  format, next to the extension -- `onnx_deploy_py.Pipeline("some_dir")`
  loads them exactly the same way `optimum` would. Anyone with the
  directory can open them with `onnx.load()`/Netron regardless of which
  loop code drives inference. If the actual goal is hiding *weights*, that
  needs something this tool doesn't have: e.g. weights baked into the
  extension as encrypted/obfuscated data and decrypted only into
  ONNX Runtime's in-memory buffers at load time -- a meaningfully bigger
  feature, not implemented here.
- **It's not a security boundary against a motivated reverse engineer.**
  Compiled C++ is slower and more annoying to read than Python, not opaque
  to it -- a `.so` full of `Ort::Session`/`std::map<std::string, Ort::Value>`
  calls disassembles and Ghidra/IDA-analyzes just fine, and every ABI call
  is a stable, documented, exported symbol by design (that's what makes it
  usable from Python/Rust/Go/etc. in the first place). Treat this as
  "raises the floor of casual inspection," not "DRM."

So: more flexible than the pure-Python glue for keeping the *loop logic*
out of easily-read/patched Python, genuinely (that's `onnx_deploy_py`,
added here) -- but not a substitute for actually protecting model weights
if that's the real requirement.

## What is deliberately out of scope

- **Tokenization.** `Generate()`/`.generate()` take and return `int64_t`
  token ids, not strings. See e.g.
  [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp) as the
  natural thing to link in front of this.
- **Sampling strategies.** Greedy-only (`ArgmaxLastToken`). Top-k/top-p/
  temperature sampling is a small, independent change to that one function.
- **`config.json` parsing.** Generation parameters are passed in explicitly
  (`GenerationConfig`/CLI flags/Python kwargs) rather than read from
  `generation_config.json`, so there is no JSON dependency anywhere in this
  tool -- everything about tensor shapes and layer count comes from the
  loaded `Ort::Session`s' own input/output metadata.
- **Batch size > 1** and **beam search**.
- **The merged `decoder_model_merged.onnx` shape** (see above).
- **Weight obfuscation/encryption** (see previous section).

## Building

```sh
# ORT_HOME only needs to point at *some* ONNX Runtime distribution's headers
# (any recent release works -- the ORT C API is stable). The libonnxruntime
# actually run against is chosen separately, at runtime, via --libort /
# onnx_deploy_load_ort() / onnx_deploy_py.load_ort() -- see below.
cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.19.2
cmake --build build
```

Add `-DONNX_DEPLOY_PYTHON=ON` (needs `pip install nanobind`) to also build
`onnx_deploy_py` under `build/python/`.

## Verifying the flow

CI (`.github/workflows/onnx-deploy.yml`) does exactly this, from a clean
checkout, on every change under `tools/onnx-deploy/`:

1. Downloads two different real ONNX Runtime releases (1.18.1 and 1.19.2).
2. Configures and builds `onnx_deploy_c`/`onnx-deploy`/`onnx_deploy_py`
   against **only the older release's headers** -- and asserts
   `ldd build/libonnx_deploy_c.so` shows no `libonnxruntime` dependency.
3. Generates a tiny hand-built seq2seq export with
   `scripts/make_toy_seq2seq.py` (`onnx` package only -- no
   torch/transformers/optimum). It is not a real language model: see that
   script's docstring for the exact math, chosen so the correct output
   sequence is fully hand-computable (`compute_expected_ids()`) and so a
   broken KV-cache handoff -- including the "cache entry not re-output every
   step" subtlety above -- changes the output instead of just not crashing.
4. Runs `onnx-deploy` against the toy export with `--libort` pointed at the
   **older** release, asserts the exact expected token sequence.
5. **Swaps `--libort` to the newer release, no rebuild, on the same
   compiled binary**, and asserts the identical expected sequence again --
   this is the actual "swappable at runtime" claim, exercised for real, not
   just a design note.
6. Repeats the same swap test through `onnx_deploy_py` (fresh Python
   process per ORT build).
7. Checks that a bad model directory and a bad `--libort` path both fail
   cleanly (`ONNX_DEPLOY_ERROR` / exit 1 with a message), not a crash.

To reproduce locally: download any two ONNX Runtime releases from
<https://github.com/microsoft/onnxruntime/releases> (`onnxruntime-linux-x64-*.tgz`
et al.), then run the same `cmake`/`make_toy_seq2seq.py`/`onnx-deploy`
sequence the workflow does.

## WASM (design only -- not yet built)

Not implemented here: this repo's existing WASM/ONNX Runtime bridge
(`JsModelExecutor`, `scripts/convertmodel/js_model_executor.cpp`, see
`docs/wasm_ort_web.md`) is an Asyncify bridge for exactly **one** awaited JS
call per `Run()` -- built for onnxsim's single constant-fold call, not a
multi-step decode loop. Reusing it for `KvCachePipeline::Generate()` (encoder
once, decoder N times, cache threaded between awaited calls) needs an
embind class that keeps an `ort.InferenceSession` (or two -- encoder and
decoder) alive JS-side across the whole loop, rather than the current
per-call session pattern -- `docs/wasm_ort_web.md` already flags
per-call `InferenceSession.create` as a likely bottleneck for exactly this
reason. "Swappable libort" in WASM terms would mean the JS host chooses
which onnxruntime-web build/execution-provider backs that session, mirroring
the native dlopen swap at the JS/wasm boundary instead of the OS loader.
This needs real design + implementation + browser/Node test infrastructure
beyond what fits here; flagging the shape of the work rather than shipping
unverified code for it (this sandbox has no `emcc` to even compile-check
a WASM attempt against).
