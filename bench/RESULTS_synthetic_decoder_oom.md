# Investigating (and fixing) the ~5GB decoder-submodule OOM

**Follow-up to:** `bench/TODO_large_decoder_submodule_oom.md`
**Reproduce:** `bench/decoder_oom_repro.py` (`gen` / `measure` / `matrix` subcommands)
**Environment:** this repo's own sandbox -- a `memory` cgroup capped at
**13.34 GiB** (`memory.limit_in_bytes` = 14327726080 bytes; read from
`/sys/fs/cgroup/memory/.../memory.limit_in_bytes`), 4 vCPUs, Python 3.11,
onnx 1.22.0, no `onnxruntime` installed. Close enough to the original
report's "~15GB RAM cap" to be a useful stand-in, and small enough that a
several-GB model comfortably exercises the boundary.

**Note on this document's history:** an earlier version of this investigation
(still visible in git history) attributed the peak-memory ratio below to a
redundant *Python-side* reload in `onnx_simplifier.py`'s fast path, and shipped
an `output_path` parameter to skip it. That reload is real, but a follow-up
measurement (`ONNXSIM_DEBUG_PATH_TIMING=1` plus external `/proc/<pid>/status`
RSS sampling, bypassing the GIL-blocking issue that makes in-process sampling
useless during a C++ call) proved it doesn't move the needle: removing it left
the measured peak byte-for-byte identical. The actual cause -- found by
sampling RSS through each phase of `SimplifyPath` and narrowing it down to a
single line -- is inside the **C++ core**, described below. That's now fixed
directly; the `output_path` parameter still exists (it helps a couple of
narrower cases) but is no longer the headline fix.

## TL;DR

* **File count doesn't matter, total bytes do.** A model with one external-data
  file per initializer (168 files, mimicking `torch.onnx.export`'s legacy
  exporter) and the same model consolidated into a single external file
  produce **byte-identical peak RSS**. Settles the TODO doc's open question --
  there is no separate per-file overhead.
* **Root cause, found precisely:** `onnxsim.cpp`'s `Simplify()` takes its
  input model by `const&` (so callers who need their original preserved get
  that guarantee) and therefore always deep-copies it into a mutable working
  copy (`onnx::ModelProto sim_model = model;`) before running the fixed
  point. For a model whose weights dominate its size, that copy alone costs
  another ~1x model size in peak RSS on top of the ~1x already needed to hold
  the loaded input -- explaining a ~1.9x peak-to-model-size ratio with no
  need to invoke Python, file count, or check_n at all.
* **Fixed:** added `SimplifyConsumeInput`, an explicit, opt-in variant that
  takes the model by mutable reference and *moves* its tensor data into the
  working copy (the same move-based ModelProto&lt;-&gt;Graph round trip
  already used elsewhere in this file for the per-round fixed-point Graph,
  and in onnx-optimizer's own "consuming" `Optimizer::optimize()` overload)
  instead of copying it. Wired into `SimplifyPath` (the file-to-file entry
  point both the CLI and `onnxsim.simplify(path, check_n=0)`'s fast path use)
  and into `C.simplify`'s in-memory/bytes entry point -- both own a model
  they discard immediately after the call, which is exactly when this is
  safe. **Verified fix:** the previously-OOMing ~8.02 GB model now completes
  at 7.6 GiB peak (~0.97x model size, i.e. essentially just the cost of
  holding it once) instead of being killed at >13.3 GiB.
* **`check_n>0` is a separate, still-open issue**, unaffected by this fix (it
  runs a different code path that must keep the original model around for
  comparison, and additionally reloads full models per correctness-check
  trial through onnx's pure-Python reference evaluator when `onnxruntime`
  isn't installed). A model that now succeeds at `check_n=0` can still OOM at
  `check_n=1`.

## Results

All measurements are `onnxsim.simplify(path, check_n=0)` (the CLI's and the
fast path's exact call), single-file external-data layout (file count doesn't
affect any of this, see above).

| model size | peak RSS, before fix | peak RSS, after fix | ratio, before | ratio, after | outcome, before | outcome, after |
|---|---:|---:|---:|---:|---|---|
| 4.93 GB | 9.29 GiB | 4.74 GiB | 1.88x | 0.96x | OK | OK |
| 8.02 GB | ≥13.22 GiB | 7.61 GiB | ≥1.65x | 0.95x | **OOM-killed** | OK |

"≥" marks a run killed by the OOM killer (`SIGKILL`) while still climbing --
its true peak, had the cgroup allowed it, would have been higher, so the
before/after gap at 8.02 GB is a lower bound. `check_n=1` on the same 4.93 GB
model still OOMs after the fix (peak reaches ≥13.27 GiB before being killed,
essentially unchanged from before the fix) -- expected, since that path is
untouched (see "What's still open" below).

Reproduce the fixed numbers with:

```
python bench/decoder_oom_repro.py matrix /tmp/work --sizes 5,8
```

(the "many vs single" file-count comparison in that script's output is
unaffected by this fix and still shows byte-identical peaks at every size.)

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

## Root cause, precisely

Sampling `/proc/<pid>/status`'s `VmHWM` from an external shell loop every
100ms (Python-level sampling can't run *during* a C++ call: the GIL blocks it,
as `bench/peak_memory.py`'s own methodology notes already point out) against
`onnxsim.simplify(path, check_n=0)` with `ONNXSIM_DEBUG_PATH_TIMING=1` shows
RSS growing in two distinct, contiguous phases on the 4.93GB model:

```
SimplifyPath: loadModel  21161.7ms   <- RSS: 0 -> ~4.85 GB  (one copy of the model)
SimplifyPath: Simplify   11333.6ms   <- RSS: ~4.85 -> ~9.77 GB (a SECOND ~4.9GB copy)
SimplifyPath: ByteSizeLong    0.2ms  <- flat
SimplifyPath: saveModel   7780.7ms   <- flat (external-data write, no extra RSS)
```

The peak is reached **and never exceeded again** by the end of the `Simplify`
phase -- confirmed by comparing `getrusage(RUSAGE_SELF).ru_maxrss` sampled
immediately after `C.simplify_path` returns to the same value sampled at the
very end of the Python call: identical, so nothing afterward (not
`model_checking.compare`, not any Python-side reload) contributes to the
peak. That ruled out this document's original theory (a Python-side
`onnx.load(fast_out_path)` reload) directly: removing it experimentally
produced a byte-for-byte identical peak, which is only possible if the peak
was already set before that reload ever ran.

`Simplify()` (`onnxsim.cpp`) takes its model by `const&`:

```cpp
onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model, ...)
```

so that callers who need their original model preserved (e.g. anything
needing a before/after comparison) get that guarantee. But the fixed point
underneath mutates in place, so the function makes a working copy up front:

```cpp
// The fixed points mutate in place, so make one working copy of the (const)
// input model and simplify it in place.
onnx::ModelProto sim_model = model;
```

For an external-data model whose weights dominate its size, this is a second
full deep copy of the whole tensor payload -- `model` (~1x, from `loadModel`)
plus `sim_model` (another ~1x, from this line) simultaneously resident. That
alone explains the ~1.9x peak-to-model-size ratio, independent of Python,
file count, or `check_n`.

The only other use of the original `model` parameter anywhere after this line
is `RecordSimplifyDiffMetadata(sim_model, model)` near the end, which computes
a structural diff (removed/changed nodes, dropped doc strings) -- confirmed
by reading `model_info.cpp`'s implementation to never touch tensor byte data,
only shapes, node lists, op types and doc strings.

## The fix

`onnxsim.cpp` already uses a move-based ModelProto&lt;-&gt;Graph round trip
elsewhere -- `OptAndShape`'s per-round resident Graph calls
`onnx::ExportModelProto(&out, g, /*consume_tensor_data=*/true)`, and
onnx-optimizer's own `optimize.h` has an analogous "consuming" overload of
`Optimizer::optimize()` with the exact comment: *"This roughly halves the
memory traffic of one optimize() call for weight-heavy models."* The fix
applies that same, already-proven pattern one level higher, to the `sim_model`
copy itself:

1. Extracted `Simplify()`'s body into a private `SimplifyImpl`, parameterized
   by an optional `onnx::ModelProto* mutable_model`. When null, `sim_model`
   is built exactly as before (a plain deep copy) -- so the existing
   `Simplify()` entry point is byte-for-byte unchanged, and every other
   caller (Rust bindings, C API, WASM, every existing test) is unaffected.
2. Added `SimplifyConsumeInput(executor, onnx::ModelProto& model, ...)`: when
   `mutable_model` is non-null, `sim_model` is instead built via
   `ImportModelProto(*mutable_model)` (the *mutable* overload, which moves
   each initializer's raw bytes out of `model`) followed by
   `ExportModelProto(&sim_model, g, /*consume_tensor_data=*/true)` (moves
   them into `sim_model`) -- the same Import/Export pair `OptAndShape`
   already uses, just called once at the top instead of once per fixed-point
   round. `model`'s structure survives intact (needed for the diff above);
   only its tensor bytes move.
3. Wired into the two callers that own a model they discard right after the
   call and never read again beforehand -- exactly the condition
   `SimplifyConsumeInput`'s doc comment requires:
   - `SimplifyPath` (`onnxsim.cpp`): `model = SimplifyConsumeInput(executor, model, ...)`
     -- the file-to-file entry point both the CLI and `simplify(path,
     check_n=0)`'s Python fast path call.
   - `C.simplify`'s bytes-based binding (`cpp2py_export.cc`): its `model` is
     parsed fresh from the input bytes and only ever used for this one call.
   No other call site was touched, and none needed to be: this is a new,
   separately-named function, not an overload of `Simplify` that could
   silently change behavior for an existing caller via overload resolution.

See `onnxsim/onnxsim.h`/`onnxsim.cpp`/`cpp2py_export.cc` for the actual diff.

### Why `output_path` (this doc's original fix) is now secondary

The `output_path` parameter added to `onnxsim.simplify()` (a Python-level
change to skip reloading the result after the fast path's C++ call) is still
in the codebase and still does what it says, but measuring it against the
*fixed* C++ core shows it no longer has anything left to save: with the C++
fix in place, `output_path` and no `output_path` produce the same peak
(4856.9 vs 4857.3 MiB on the 4.93GB model -- noise). That's expected: once
the C++ side needs only ~1x model size, there's no second large peak left for
a Python-side reload to add on top of. `output_path` still helps in the two
cases the C++ fix doesn't reach:

* `check_n > 0` (goes through the slow path, `Simplify()`'s plain overload,
  which must still deep-copy to preserve the original for comparison) --
  `output_path` at least avoids re-loading the *already-materialized* result
  a second time before returning.
* A caller who wants the saved file but supplies an in-memory `ModelProto`
  rather than a path (`output_path` requires a path input, so this doesn't
  apply, but see the parameter's docstring for the exact conditions).

## What's still open

* **`check_n > 0` is not fixed.** `model_checking.compare()` still reloads a
  full model per correctness-check trial via onnx's pure-Python reference
  evaluator when `onnxruntime` isn't installed (`onnxsim/model_checking.py`
  ~line 328-329), on top of `Simplify()`'s own const-preserving path (which
  correctly cannot use `SimplifyConsumeInput`, since check_n > 0 needs the
  original model intact for the comparison). A model that now succeeds at
  `check_n=0` can still OOM at `check_n=1` -- confirmed on the 4.93GB model
  above. If the real backbone submodule's export pipeline used a nonzero
  `check_n`, this fix alone would not have resolved the original report.
* **The CLI's own extra save.** `main()` still calls `onnx.save(model_opt,
  ...)` after `simplify()` returns, which is now cheap relative to before (no
  second giant copy already happened inside `simplify()`) but is still a
  second write of a potentially large model. Not urgent given the fix above
  already addresses the actual OOM, but worth revisiting if the CLI path
  specifically needs to shave further memory or time.
* **Not validated against the real backbone submodule** (see the TODO doc's
  "How to get the model") -- the mechanism found here is generic to any
  external-data model whose weights dominate its size, but a real
  transformer's additional structure (KV-cache concat, RoPE, embeddings)
  hasn't been checked against this exact fix.
* **No `massif`/`heaptrack` trace was captured** -- the root cause was
  isolated by external `/proc/<pid>/status` RSS sampling correlated with the
  existing `ONNXSIM_DEBUG_PATH_TIMING` phase markers, and confirmed by
  reading the relevant source directly (finding the exact `sim_model = model`
  line and tracing its only other later use), not by a line-level allocator
  profile. A real profiler run would be a good independent confirmation but
  wasn't necessary to find or fix this.

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
`RUSAGE_CHILDREN == 0`. Anyone extending this bench with more in-process
measurement loops should keep this in mind -- it's an easy mistake to
reproduce.
