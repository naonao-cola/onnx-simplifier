# Peak-memory impact of PR #482

**PR:** [#482 — *Defer loading external tensor data to reduce memory usage*](https://github.com/onnxsim/onnxsim/pull/482)
**Compared:** `b625e26` (parent, pre-PR) vs `7e500fc` (merge commit)
**Reproduce:** `bench/peak_memory.py` (see its docstring)

## TL;DR

For a model whose weights (~**1.0 GiB**) live in an external-data file:

| Scenario | Metric | Before | After | Reduction |
|---|---|---:|---:|---:|
| **CLI** (`onnxsim in out`) | **Peak RSS** | **8073 MiB** | **7049 MiB** | **−1024 MiB (−12.7%)** |
| CLI | RSS held after load | 1081 MiB | 57 MiB | −1024 MiB (19×) |
| Library API (`simplify(path)`) | Peak RSS | 7048 MiB | 7048 MiB | 0 (unchanged) |
| Library API | RSS held after load | 1092 MiB | 68 MiB | −1024 MiB (16×) |

**The peak reduction equals one full copy of the external weights.** For the
CLI it scales with model size: a model with *W* bytes of external weights peaks
roughly *W* lower after the PR. For an *N*-GiB model that is ~*N* GiB less peak
RAM.

## What the PR does

Before, both the CLI and `simplify()` called `onnx.load(path)`, which
immediately reads the external `.data` file into every tensor's `raw_data`. The
fully-materialized weights then sit in RAM through every graph-metadata phase
(input-shape overwrite, unused-output / initializer pruning, unhashable-tensor
detection, doc-string snapshot) even though none of those phases read raw tensor
bytes.

After, the model is loaded with `load_external_data=False` (weights left on
disk), and the external data is materialized only right before the model is
serialized for the C++ simplifier. In addition the CLI hands `simplify()` the
**path** instead of a loaded `ModelProto`, so the weights never enter `main`'s
process image at all — `simplify` owns the single copy it loads.

## Why the CLI peak drops but the API peak does not

Peak RSS of a full run occurs *inside the C++ simplifier*, which must
deserialize the serialized model — so at the peak the weights are resident
regardless of the PR. Both versions reach that point, so the **instantaneous
peak of `simplify()` alone is unchanged** (7048 MiB either way).

The CLI, however, previously kept a **second** full copy of the weights alive:
`main` loaded the whole model *and* `simplify` processed it. The PR removes that
second copy (`main` now keeps only a ~57 MiB metadata skeleton), so the process
peak drops by exactly one weights-worth (~1 GiB).

Measured checkpoints make this concrete:

```
CLI  old:  after_load = 1081 MiB   ->  peak = 8073 MiB
CLI  new:  after_load =   57 MiB   ->  peak = 7049 MiB   (main holds no weights)
API  old:  after_load = 1092 MiB   ->  peak = 7048 MiB
API  new:  after_load =   68 MiB   ->  peak = 7048 MiB   (single copy either way)
```

## The other, always-present win: sustained working set

Even where the instantaneous peak is unchanged (the API path), the memory the
model occupies **throughout the Python-side graph transformations** falls from
~1092 MiB to ~68 MiB — a **16× smaller working set** for the whole duration of
those phases. Consequences:

* Lower average memory pressure while the (potentially long) optimization phases
  run.
* If any pre-serialization phase raises (bad input shape, unhashable tensor,
  etc.), the old code had already spent ~1 GiB loading weights it never used;
  the new code spends ~0.
* `model_info.ModelInfo` now reports model size from external-data **metadata**,
  so a multi-GB model can be measured without loading it (and a subgraph
  double-counting bug in the old size calc is fixed) — see the PR's tests.

## Method & caveats

* Model: 16-layer MatMul chain, 4096×4096 fp32 weights (~1.0 GiB), saved as a
  single external `.data` file; the `.onnx` file itself is ~2 KB.
* The two variants share one compiled `onnxsim_cpp2py_export` extension (from
  the released 0.6.5 wheel). PR #482 changed **only Python**, so the C++
  simplification stage is identical for both and cancels out of every delta.
  Absolute peak values therefore reflect the released C++ backend; the
  *reductions* are the PR's effect.
* Peak = `ru_maxrss` (`resource.getrusage`), the process high-water mark. A
  2 ms RSS sampler was also used but under-reads the peak because the C++ call
  holds the GIL and starves the sampler — `ru_maxrss` is the authoritative peak.
* Host: Linux x86-64, Python 3.11, onnx 1.22, onnxruntime 1.28, 15 GiB RAM.
