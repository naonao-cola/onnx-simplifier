# Open: onnxsim OOMs simplifying a ~5GB decoder-only transformer submodule

**Status:** likely root cause identified via a synthetic repro, fix not yet applied --
see `bench/RESULTS_synthetic_decoder_oom.md`. The original real-export observation below
is unconfirmed against the real model (no fix has been validated against it), but the
mechanism found is generic (any large enough model) and doesn't depend on this specific
model or export pipeline.

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
  equivalent OOM and found:
  - **File count vs. total bytes is resolved: file count doesn't matter.** A
    168-file-external-data model and the same model consolidated into one file produced
    byte-identical peak RSS at every size tested.
  - `onnxsim.simplify()`'s peak RSS is consistently **~1.9x the model's total weight
    bytes**, even at the default `check_n=0` (no correctness check). This traces to a
    real double materialization in the "fast path" of `onnx_simplifier.py`
    (~line 1281): `C.simplify_path` correctly avoids a Python-side round trip for the
    *input*, but the very next line (~1332) unconditionally does
    `model_opt = onnx.load(fast_out_path)`, fully reloading the *output* into Python
    just to satisfy `simplify()`'s `ModelProto`-returning contract. The onnxsim CLI
    compounds this with a third full pass (`onnx.save(model_opt, ...)` at ~line 2711)
    that most callers of `onnxsim in.onnx out.onnx` never needed.
  - `check_n>0` makes it substantially worse (~2.7x in the repro) because without
    `onnxruntime` installed, each correctness-check trial reloads the entire model
    again via onnx's pure-Python reference evaluator (`onnxsim/model_checking.py`
    ~line 328-329).
  - This ~1.9x-2.7x ratio is large enough on its own, at the real submodule's ~5.3GB
    size in a ~15GB cap, to plausibly fully explain the original OOM without needing
    any real-model-specific structure (KV-cache, RoPE, embeddings) as an explanation --
    though that hasn't been confirmed against the actual model (see "Next steps" below).

## Next steps, if picking this up

1. ~~Reduce to a runnable repro~~ -- done, see `bench/decoder_oom_repro.py` and
   `bench/RESULTS_synthetic_decoder_oom.md`.
2. Capture an actual memory profiler trace (`massif`/`heaptrack`) on the synthetic
   repro to confirm the two-materialization theory at the allocation level, rather than
   the black-box `ru_maxrss` deltas used so far.
3. ~~Isolate file-count vs. total-bytes~~ -- done: file count does not affect peak
   memory, only total bytes do.
4. Design and implement a fix for the double materialization identified above --
   e.g. a way for `simplify()`'s fast path to skip re-loading `fast_out_path` when the
   caller doesn't need the `ModelProto` back (an API-surface change, needs design
   discussion since the documented return contract is `(ModelProto, bool)`), and the
   equivalent for the CLI's own extra `onnx.save(model_opt, ...)` round trip.
5. Confirm the ~1.9x ratio found on the synthetic model holds against the real
   backbone submodule (see "How to get the model" above), to check whether real
   transformer structure (KV-cache concat, RoPE, embeddings) changes the ratio.
