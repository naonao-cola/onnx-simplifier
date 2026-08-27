# Open: onnxsim OOMs simplifying a ~5GB decoder-only transformer submodule

**Status:** unresolved -- this documents an observation from a real export, not a
diagnosed root cause or a fix. Filed as a starting point for whoever picks it up next.

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
  consolidated file. Whether onnxsim's own peak memory during `simplify()` scales with
  the *number* of external-data files (I/O/bookkeeping overhead per file) or just the
  *total bytes* (fewer, larger allocations either way) was not isolated this round.
- The smaller submodules (<=1.7GB, single inline file) simplified without issue in the
  same ~15GB environment, so the failure is specific to this model, not universal to
  `simplify()` on any external-data model.
- No profiling was done -- the OOM was observed (process killed) but no peak-memory
  trace (`/usr/bin/time -v`, `valgrind --tool=massif`, or similar) was captured.

## Next steps, if picking this up

1. Reduce to a runnable repro: either export the real backbone submodule per "How to
   get the model" above, or build a synthetic ONNX model in this scale range (~5GB,
   external data, a decoder-block-style repeated structure) with
   `onnx.helper`/`numpy_helper` (no torch/HF dependency needed) -- the synthetic route
   avoids the `breeze-tts` tracing workaround and is easier to share/CI, at the cost of
   not being certain it reproduces the same failure.
2. Re-run under an actual memory profiler and capture the trace, rather than reasoning
   from wall-clock OOM alone.
3. Isolate file-count vs. total-bytes: consolidate the same model's external data into
   a single file (`onnx.save_model(..., save_as_external_data=True,
   all_tensors_to_one_file=True)`) and compare peak memory against the many-small-files
   version at the same total size.
4. Compare against `bench/RESULTS_pr482_peak_memory.md` (this repo's existing peak-memory
   investigation) for methodology and whatever headroom that work already established.
