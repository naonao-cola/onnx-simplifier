# Open: onnxsim OOMs simplifying a ~5GB decoder-only transformer submodule

**Status:** root cause found and fixed for the default (`check_n=0`) path -- see
`bench/RESULTS_synthetic_decoder_oom.md`. A synthetic ~8GB model that previously
OOM-killed in this repo's own ~13.3GiB sandbox now completes at ~0.95x its own size
(down from being killed above 1.65x and climbing). `check_n>0` is a separate, still-open
issue the fix does not touch (see that doc's "What's still open"). The original
real-export observation below is unconfirmed against the real model (the mechanism
found is generic to any large-enough external-data model and doesn't depend on this
specific model or export pipeline, but whether the real submodule's export used
`check_n=0` or a nonzero value was never established).

**Model:** the backbone (decoder-only transformer) submodule of a multi-submodule TTS
model, exported standalone via `torch.onnx.export(..., dynamo=False)`. Four submodules
were exported from the same model; three (each <=1.7GB, serialized as a single inline
`.onnx` file) simplified successfully. The backbone (~5.3GB) did not.
**Environment:** a sandbox with a ~15GB RAM cap and (separately) a constrained disk
quota. Absolute numbers are specific to that environment; the qualitative finding
(this submodule OOMs while smaller ones from the same export pipeline do not) is the
useful part.
**Reproduce:** the real model is obtainable (see "How to get the model" below), but no
export script or runnable repro exists yet -- see "Next steps" for what that takes.

## How to get the model

The model behind this observation is [`BreezeBlue/Breeze-TTS-2`](https://hf.co/BreezeBlue/Breeze-TTS-2)
on Hugging Face (~3.4B params total across all submodules; check the model card's
license before redistributing/using the weights). Reproducing the backbone export
needs two things:

1. **Weights** -- download the safetensors shards:
   ```
   huggingface-cli download BreezeBlue/Breeze-TTS-2 --local-dir breeze-model
   ```
2. **Model code** -- the weights load into classes (`BreezeConfig`,
   `BreezeBackboneFactory`, etc.) from [`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts)
   on GitHub (not published to PyPI):
   ```
   git clone https://github.com/breezeblue-ai/breeze-tts
   ```
   Its `requirements.txt` pins `torch==2.9.1`, `transformers==4.57.3`.

No export script for the backbone submodule exists in this repo (it doesn't belong
here -- it's specific to `breeze-tts`'s internal API). To reconstruct one: load only
the `backbone_model.*`-prefixed tensors from the safetensors shards into
`BreezeBackboneFactory.create_backbone(config)`, set `config._attn_implementation =
"eager"`, and trace with `torch.onnx.export(model, (inputs_embeds, position_ids),
..., dynamo=False)`. `dynamo=True` (`torch.export`) cannot trace this model as-is:
transformers' `masking_utils` treats a call with `position_ids` but no
`attention_mask`/`past_key_values` as a possibly-packed-sequence batch and routes mask
construction through a `torch.vmap`-based function that neither `torch.export` nor the
legacy TorchScript tracer can trace here -- a PyTorch/transformers tracing gap
unrelated to onnxsim. Since only a single, unpacked sequence is ever exported per
submodule, packed-sequence detection can be disabled for tracing with
`transformers.masking_utils.find_packed_sequence_indices = lambda position_ids: None`.

## What's known

- `torch.onnx.export`'s legacy (`dynamo=False`) exporter keeps weights inline for
  models under the 2GB protobuf limit, and automatically externalizes larger ones --
  but as one file per initializer (254 small files for this 5.3GB model), not one
  consolidated file.
- The smaller submodules (<=1.7GB, single inline file) simplified without issue in the
  same ~15GB environment, so the failure is specific to this model, not universal to
  `simplify()` on any external-data model.
- **Update -- see `bench/RESULTS_synthetic_decoder_oom.md` for the full writeup.** A
  synthetic, torch/HF-free repro (`bench/decoder_oom_repro.py`) reproduced an
  equivalent OOM, found the real root cause (correcting an earlier, disproven theory
  in this same doc's history -- see that file's own note at the top), and fixed it:
  - **File count vs. total bytes is resolved: file count doesn't matter.** A
    168-file-external-data model and the same model consolidated into one file produced
    byte-identical peak RSS at every size tested.
  - **Root cause:** `onnxsim.cpp`'s `Simplify()` takes its input model by `const&` (to
    guarantee callers who need it preserved get that), so it always deep-copies the
    whole model into a mutable working copy before running the fixed point
    (`onnx::ModelProto sim_model = model;`). For an external-data model whose weights
    dominate its size, that's a second full copy of the tensor payload on top of the
    one already held from loading -- a ~1.9x peak-to-model-size ratio with no need to
    invoke Python, file count, or `check_n` at all. Found by sampling `VmHWM` from
    outside the process (Python-level sampling can't run *during* a C++ call -- the
    GIL blocks it) correlated with `ONNXSIM_DEBUG_PATH_TIMING`'s existing phase
    markers, which narrowed the growth to exactly this one line.
  - **Fixed:** added `SimplifyConsumeInput`, an explicit opt-in variant taking the
    model by mutable reference that *moves* tensor data into the working copy instead
    of copying it (the same move-based ModelProto&lt;-&gt;Graph round trip already used
    elsewhere in this file, and in onnx-optimizer's own "consuming" `optimize()`
    overload). Wired into `SimplifyPath` (the file-to-file entry point the CLI and
    `simplify(path, check_n=0)`'s fast path use) and `C.simplify`'s bytes-based
    binding -- both own a model they discard immediately after the call. A previously
    OOM-killed ~8GB synthetic model now completes at ~0.95x its own size.
  - `check_n>0` is a **separate, still-open issue** the fix above does not touch: that
    path must keep the original model intact for comparison (so it cannot use the new
    consuming variant), and additionally reloads a full model per correctness-check
    trial through onnx's pure-Python reference evaluator when `onnxruntime` isn't
    installed. A model that now succeeds at `check_n=0` can still OOM at `check_n=1`.
  - Whether the real submodule's export pipeline used `check_n=0` (now fixed) or a
    nonzero value (still open) was never established -- see "Next steps" below.

## Next steps, if picking this up

1. ~~Reduce to a runnable repro~~ -- done, see `bench/decoder_oom_repro.py` and
   `bench/RESULTS_synthetic_decoder_oom.md`.
2. ~~Isolate file-count vs. total-bytes~~ -- done: file count does not affect peak
   memory, only total bytes do.
3. ~~Find and fix the root cause of the `check_n=0` OOM~~ -- done: `Simplify()`'s
   const-preserving deep copy, fixed via `SimplifyConsumeInput`. See
   `bench/RESULTS_synthetic_decoder_oom.md`'s "The fix" section for the exact change.
4. Fix `check_n>0`, still open: reloading a full model per correctness-check trial
   through the pure-Python reference evaluator (`onnxsim/model_checking.py`
   ~line 328-329) is the next candidate to profile and fix the same way -- likely
   needs `backend.py`'s reference-evaluator path to work from a path/streamed source
   rather than a fully materialized `ModelProto`, mirroring what `SimplifyConsumeInput`
   did for the simplification path itself.
5. Confirm the fix holds against the real backbone submodule (see "How to get the
   model" above), to check whether real transformer structure (KV-cache concat, RoPE,
   embeddings) changes anything, and to establish whether that export pipeline used
   `check_n=0` or a nonzero value.
6. Consider the CLI's own extra `onnx.save(model_opt, ...)` round trip after
   `simplify()` returns -- now cheap relative to before (no second giant copy already
   happened inside `simplify()`), but still a second write worth revisiting if the CLI
   path specifically needs to shave further memory or time.
