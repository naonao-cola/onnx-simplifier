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
**Reproduce:** not yet reduced to a runnable repro script. See "Next steps" below for
what that would take.

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

1. Reduce to a runnable repro: either a synthetic ONNX model in this scale range
   (~5GB, external data, a decoder-block-style repeated structure) built with
   `onnx.helper`/`numpy_helper` (no torch/HF dependency needed), or point at a public
   model of similar size if one exists in `onnxmodelzoo` or similar.
2. Re-run under an actual memory profiler and capture the trace, rather than reasoning
   from wall-clock OOM alone.
3. Isolate file-count vs. total-bytes: consolidate the same model's external data into
   a single file (`onnx.save_model(..., save_as_external_data=True,
   all_tensors_to_one_file=True)`) and compare peak memory against the many-small-files
   version at the same total size.
4. Compare against `bench/RESULTS_pr482_peak_memory.md` (this repo's existing peak-memory
   investigation) for methodology and whatever headroom that work already established.
