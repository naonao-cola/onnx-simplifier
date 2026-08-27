# Investigating the ~5GB decoder-submodule OOM with a synthetic repro

**Follow-up to:** `bench/TODO_large_decoder_submodule_oom.md`
**Reproduce:** `bench/decoder_oom_repro.py` (`gen` / `measure` / `matrix` subcommands)
**Environment:** this repo's own sandbox -- a `memory` cgroup capped at
**13.34 GiB** (`memory.limit_in_bytes` = 14327726080 bytes; read from
`/sys/fs/cgroup/memory/.../memory.limit_in_bytes`), 4 vCPUs, Python 3.11,
onnx 1.22.0, no `onnxruntime` installed. Close enough to the original
report's "~15GB RAM cap" to be a useful stand-in, and small enough that a
several-GB model comfortably exercises the boundary.

## TL;DR

* **File count doesn't matter, total bytes do.** A model with one external-data
  file per initializer (168 files, mimicking `torch.onnx.export`'s legacy
  exporter) and the same model consolidated into a single external file
  produce **byte-identical peak RSS** at both sizes tested. This settles the
  TODO doc's open question -- there is no separate per-file overhead.
* **`onnxsim.simplify()`'s peak RSS is consistently ~1.9x the model's total
  weight bytes**, even on the default, no-correctness-check path
  (`check_n=0`). That ratio is large enough on its own to explain the
  original OOM: it comes from a real, identifiable double materialization in
  `onnx_simplifier.py`'s "fast path" (see below), not from file count, not
  from constant folding, and not from shape inference.
* **`check_n>0` (correctness verification) makes it substantially worse** --
  up to ~2.7x -- because without `onnxruntime` installed, each check trial
  reloads the *entire* model a fresh time through onnx's pure-Python
  reference evaluator.
* At this sandbox's 13.34 GiB cap, a **~4.93 GB** model survives at
  `check_n=0` (peak 9.29 GiB) but **OOMs at `check_n=1`** (peak crosses
  13.2 GiB before being killed). An **~8.02 GB** model OOMs even at
  `check_n=0`. The real submodule was ~5.3 GB in a ~15 GB cap -- a very
  similar margin -- so either a nonzero `check_n` somewhere in that export
  pipeline, or a modest amount of concurrent memory use alongside onnxsim
  (both very plausible for a real multi-submodule export script), is enough
  to tip it over by this same mechanism.

## Results

| layout | check_n | model size | peak RSS | peak / size | outcome |
|---|---:|---:|---:|---:|---|
| many (168 files) | 0 | 4.93 GB | 9.29 GiB | 1.88x | OK |
| single (1 file) | 0 | 4.93 GB | 9.29 GiB | 1.88x | OK (byte-identical to "many") |
| many (168 files) | 1 | 4.93 GB | ≥13.22 GiB | ≥2.68x | **OOM-killed** |
| many (273 files) | 0 | 8.02 GB | ≥13.22 GiB | ≥1.65x | **OOM-killed** |
| many (273 files) | 1 | 8.02 GB | ≥13.21 GiB | ≥1.65x | **OOM-killed** |
| single (1 file) | 0 | 8.02 GB | ≥13.22 GiB | ≥1.65x | **OOM-killed** (byte-identical to "many") |

"≥" marks a run that was killed by the kernel OOM killer (`SIGKILL`) while
still climbing -- its true peak, had the cgroup allowed it, would have been
higher. Reproduce with:

```
python bench/decoder_oom_repro.py matrix /tmp/work --sizes 5,8
```

Each row regenerates its model and measures `onnxsim.simplify()` in a fresh
subprocess (see the methodology note below for why that matters), so a run
takes a few minutes per size.

## The model

A decoder-block-shaped ONNX graph (repeated self-attention + SwiGLU-style MLP
blocks: q/k/v/o projections, softmax attention, gate/up/down MLP), sized by
layer count alone -- no torch or transformers dependency, so it needs neither
the real `BreezeBlue/Breeze-TTS-2` weights nor the `breeze-tts` tracing
workaround the TODO doc describes. `hidden=2048, ffn=5632` gives ~51.4M
params/layer; 24 layers is ~4.93 GB of fp32 weights, 39 layers is ~8.02 GB --
in the same ballpark as the real ~3.4B-param, ~5.3GB backbone submodule.

Generation writes each tensor's bytes straight to its external-data file
rather than building the whole model in memory first via
`onnx.save_model(..., save_as_external_data=True)`. That matters: an earlier
version of the generator that did go through `onnx.save_model` was **itself
OOM-killed** by this same sandbox while generating a ~4.9GB model -- 11.4 GB
anon-RSS observed for 4.9 GB of tensor data (2.3x). That's a real, separate
finding (relevant to any exporter that materializes external data through
onnx's standard Python helper API, `torch.onnx`'s legacy exporter included --
see the TODO doc's note that the legacy exporter is what produced the real
model's 254 external files in the first place), but it's a confound for what
this bench actually measures, so `decoder_oom_repro.py gen` keeps generation
at O(one tensor) of Python memory.

## Root cause: the "fast path" materializes the result twice

`onnxsim.simplify(path, check_n=0)` -- what both `measure()` here and
onnxsim's CLI actually call -- takes the "fast path" in
`onnx_simplifier.py` (~line 1281: `if check_n == 0 and not _shapes_need_model
and ...`). For a file-path input this calls `C.simplify_path(...)`, which is
correctly designed to avoid a Python-side materialize/serialize round trip:
the C++ core reads the external-data model straight from disk and writes the
simplified result straight back to disk (`fast_out_path`), documented
explicitly as avoiding exactly the cost this investigation found:

> every initializer's bytes get materialized into a Python `ModelProto`,
> serialized back to a byte string, and reparsed on the C++ side, all before
> simplification even starts

That comment describes the *input* side correctly. But the very next line
(~1332) is:

```python
model_opt = onnx.load(fast_out_path)
```

`simplify()`'s documented return value is `(model_opt: ModelProto, check_ok:
bool)`, so the fast path -- despite going to the trouble of a pure
file-to-file C++ call for the input -- **always fully loads the entire
output back into Python** (default `load_external_data=True`) before
returning, undoing half of its own optimization. For a multi-GB model this
means:

1. The C++ engine (`C.simplify_path`) holds the model's weights internally
   while simplifying (~1x model size).
2. Python then loads the same result again in full (~1x model size) just to
   satisfy the API's `ModelProto`-returning contract.

Nothing frees the first copy's memory back to the OS before the second copy
is made (RSS is a monotonic high-water mark, and freed heap memory isn't
necessarily returned to the kernel), so the two materializations largely
**add** rather than reuse -- matching the observed ~1.9x ratio almost
exactly, and explaining why it held steady across both a 4.93GB and an 8GB
model (it's proportional to model size, not a fixed overhead).

The onnxsim **CLI** compounds this further: `main()` gets back that same
fully-materialized `model_opt` and immediately calls `onnx.save(model_opt,
..., save_as_external_data=True, ...)` (`onnx_simplifier.py` ~line 2711) to
write it back out to `args.output_model` -- a **third** full pass over the
weights for a workflow (`onnxsim in.onnx out.onnx`) that never needed the
result in Python memory at all. This bench measures the library API only
(`onnxsim.simplify()`, no save), so the CLI's additional cost isn't in the
numbers above, but it's visible directly in the code and is worth keeping in
mind for anyone hitting this via the command line rather than the API.

`check_n>0` makes this worse for an orthogonal reason: without
`onnxruntime` installed, `onnxsim/backend.py`'s `_run_with_reference` falls
back to `onnx.reference.ReferenceEvaluator`, and `model_checking.compare()`
calls it once per `check_n` trial for *both* the original and the simplified
model (`onnxsim/model_checking.py` ~line 328-329) -- each call doing its own
fresh `onnx.load()` of a full-size model. That's on top of the two
materializations above, which is why `check_n=1` pushed the ratio from
~1.88x to ~2.68x on the same 4.93GB model.

## Methodology note: a bug this investigation ran into and fixed

`bench/decoder_oom_repro.py`'s `matrix` command originally called `measure()`
as a plain Python function, in-process, once per (layout, check_n) row.
`resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss` is a **monotonic
high-water mark across every child a process has ever reaped**, not a
per-call value -- so after the first row hit a high peak, every subsequent
row in the same `matrix` invocation silently reported *at least* that value,
regardless of what it actually used. This was caught because two
consecutive rows -- one killed by OOM, one that completed successfully --
reported the exact same peak-RSS figure down to the decimal, which is not
plausible for two different outcomes. Fixed by having `matrix` invoke
`measure` as a fresh top-level subprocess per row, so each one starts from
`RUSAGE_CHILDREN == 0`. All numbers in this document are from the corrected
version. Anyone extending this bench with more in-process measurement loops
should keep this in mind -- it's an easy mistake to reproduce.

## What this doesn't establish

* This isolates onnxsim's own `simplify()` call. It does not reproduce the
  *export* step (`torch.onnx.export`) or confirm the real
  `BreezeBlue/Breeze-TTS-2` backbone hits precisely this code path -- the
  qualitative mechanism (fast-path double materialization, worsened by
  `check_n>0`) is generic to any sufficiently large model, but the exact
  ratio could differ for a real transformer graph (embeddings, KV-cache
  concat, RoPE, causal masking) versus this bench's simpler repeated
  matmul-chain structure.
* No `massif`/`heaptrack`-level trace was captured (the TODO doc's "next
  steps" #2) -- the ~1.9x figure is inferred from black-box `ru_maxrss`
  deltas across model sizes and check_n values, not from a line-level
  allocation profile. It's consistent enough across two model sizes and two
  layouts to trust qualitatively, but a real profiler would be needed to
  confirm the two-materialization explanation at the allocation level rather
  than by elimination.

## Next steps, if picking this up

1. **Fix candidate:** give `simplify()`'s fast path a way to skip the
   `onnx.load(fast_out_path)` re-materialization when the caller doesn't need
   the in-memory `ModelProto` back -- e.g. an `output_path` parameter so
   `simplify()` can leave the result on disk, or a lazy/deferred-load return
   value. This is an API-surface change (the documented return contract is
   `(ModelProto, bool)`), so it needs design discussion, not a silent
   behavior change; not attempted here.
2. Apply the same fix to the CLI's own extra `onnx.save(model_opt, ...)`
   round trip once/if the above lands -- the CLI's dominant use case
   (`onnxsim in.onnx out.onnx`) never needs the intermediate `ModelProto` at
   all.
3. Confirm this against the real backbone submodule (see the TODO doc's "How
   to get the model") to check whether the ~1.9x ratio found here holds for
   an actual transformer export, or whether real-model structure (KV-cache
   concat, RoPE, embeddings) pushes it higher.
4. Capture an actual `massif`/`heaptrack` trace on this bench's synthetic
   model to confirm the two-materialization theory at the allocation level.
