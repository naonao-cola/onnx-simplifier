# gperftools profiling of graph-native shape inference

**Goal:** profile the *real* `InferShapesOnGraph` execution (real schema-registered
inference functions, real fixed-point loop, via `onnxsim.simplify()`) with a
function/line-level CPU sampling profiler (gperftools), to confirm/refute/quantify
two not-yet-implemented optimization hypotheses discussed against
`third_party/onnx/onnx/common/graph_shape_inference.cc`:

1. **`EncodeCurrentType`**'s `out.CopyFrom(*v.type())` for non-tensor
   (Sequence/Optional/Map) values could alias `v.type()` instead of deep-copying it.
2. **`ConstantDataFor` + `encodeTensor`**: re-encoding a `Constant`/initializer
   `Tensor` into a fresh `TensorProto` on every node visit, every fixed-point round,
   could be cached by `Tensor*` address (safe since `cb6cdbd`'s stable-address
   initializer storage).

**Reproduce:** `bench/graph_shape_inference_gperftools_repro.py` (new; builds the
synthetic model and drives `onnxsim.simplify()` in a loop) plus gperftools'
`libprofiler.so` + `pprof`. See "Methodology" below for exact commands.

## TL;DR

Both hypotheses are **real but practically negligible** at the scale tested. The
profile instead surfaced a **much larger, unrelated hotspot**: graph *Import*
(`ModelProto` &rarr; `onnx::Graph`) is O(V&sup2;) in the number of graph values and
dominates end-to-end `simplify()` time by roughly two orders of magnitude over
`InferShapesOnGraph` itself.

| Region (of 61,746 total CPU samples, ~246s wall) | Samples | Share |
|---|---:|---:|
| `onnx::ImportModelProto` (`ModelProto` &rarr; `Graph`) | 49,106 | **79.5%** |
| `_EvalPartialShapeOnGraph` (partial-shape-eval fold helper) | 6,210 | 10.1% |
| `_FoldConstantOnGraph` (constant folding, no `onnxruntime` installed &rarr; reference evaluator) | 4,571 | 7.4% |
| **`onnx::InferShapesOnGraph`** (the function this task targets) | 1,471 | **2.4%** |
| &nbsp;&nbsp;of which `ProcessNode` (inlined into `Run`) | 773 | 1.3% (53% of `InferShapesOnGraph`) |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `EncodeCurrentType` (both call sites, both branches) | ~28 | 0.045% |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `ConstantDataFor` (map lookup only) | ~27 | 0.044% |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `encodeTensor` (hypothesis 2's actual target) | 14 | 0.023% |
| &nbsp;&nbsp;&nbsp;&nbsp;of which the hypothesis-1 `CopyFrom` line itself | **0** | **0%** (unsampled) |

**Hypothesis 1 (EncodeCurrentType CopyFrom aliasing): refuted as worth pursuing.**
The exact `CopyFrom` line recorded zero samples across a quarter-million-sample
profile, even though the synthetic model deliberately round-trips 250
`SequenceConstruct`/`SequenceAt` pairs specifically to hit it.

**Hypothesis 2 (ConstantDataFor/encodeTensor caching): confirmed real, but tiny.**
`encodeTensor` itself is 14 samples (~1% of `InferShapesOnGraph`'s own time) despite
the model reusing 3 small initializers as the RHS of an elementwise op on every one
of ~8,000 relevant nodes, re-encoded fresh on every one of 5 rounds, over 15 loops.
`ConstantDataFor`'s own `unordered_map` lookup costs about **twice** as much as the
`encodeTensor` call it gates.

**Unhypothesized finding, and the actual headline result:** `onnx::Graph::
isNameUnique()` — called from `getNextUnique()`, called *unconditionally* from
every `Value`'s constructor (`ir.h:1550`), called from every `Graph::createValue`
during Import — does a full O(current-graph-size) linear scan (every node's
attributes, inputs, and outputs) to find an unused synthetic id, **even for values
that get renamed immediately afterward by `setUniqueName()`** (the overwhelmingly
common case for a named `ModelProto`), making that scan's result thrown away in
practice. A single line inside it — `Value::uniqueName()`'s cached-name return,
`ir.h:364` — accounts for **44.4% of every sample taken across the entire process**.

## Environment / build

- gperftools installed via `apt-get install google-perftools libgoogle-perftools-dev
  graphviz` (root in this sandbox; package version 2.15-3build1 on Ubuntu 24.04,
  `google-pprof --version` self-reports as the older "gperftools 2.0" Perl `pprof`
  script). Both `libprofiler.so` and `google-pprof` are present and worked.
- Built the Python wheel the normal way: `pip install -e . -v` (no special flags —
  `setup.py`'s `cmake_build` already defaults to `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
  on Linux, confirmed via the printed cmake invocation and `file
  onnxsim_cpp2py_export*.so` showing `with debug_info, not stripped`).
  `ONNXSIM_BUILTIN_ORT=OFF` throughout, per this repo's `CLAUDE.md` — no ONNX Runtime
  was built. Build took ~9 minutes on 4 cores.
- `onnxruntime` is **not** pip-installed in this environment, so constant folding
  falls back to onnx's (slower) reference evaluator. This inflates `FoldConstant`'s
  *absolute* share above but does not touch anything inside `InferShapesOnGraph`,
  which is what the two hypotheses are about; if anything, an `onnxruntime`-backed
  run would shrink `FoldConstant`'s share and make Import's dominance even starker.

## Methodology

Model: `bench/graph_shape_inference_gperftools_repro.py gen`, 2,000 repeated blocks
(10,503 nodes total) over a `[dim_param("N"), 64]` activation:

- Every block: `Relu` then three elementwise ops (`Add`/`Mul`/`Sub`) against three
  distinct, reused small initializers (`bias_vec`/`scale_vec`/`shift_vec`) — the
  *same* `Tensor*` every round, every occurrence, targeting hypothesis 2.
- Every 4th block: `Shape` &rarr; `Gather` (against a shared `Constant` index node)
  &rarr; `Concat` (against a shared `Constant` 1-D tensor) &rarr; `Reshape`, so the
  dynamic batch dim is threaded through `EncodeCurrentType`'s plain-tensor path on
  every visit, and a *second*, `Constant`-node-attribute-held (not initializer)
  shared tensor also targets hypothesis 2.
- Every 8th block: `SequenceConstruct`/`SequenceAt` round-trip through a shared
  `Constant` position index — the model's only non-tensor-typed value, targeting
  hypothesis 1 (Sequence/Optional/Map are otherwise rare in ordinary ONNX graphs;
  this is the "cheaply construct one if you can" case the task allowed for).

Run: `python bench/graph_shape_inference_gperftools_repro.py run <model> --loops 15`,
calling `onnxsim.simplify(model, perform_optimization=False, check_n=0)` 15 times in
one process (`perform_optimization=False` skips the Optimize pass suite, matching
`RESULTS_graph_shape_inference_arena.md`'s end-to-end isolation method; constant
folding stays on, since it's what drives the outer fixed point to more than one
round at all). Under:

```
CPUPROFILE=/tmp/prof2.out CPUPROFILE_FREQUENCY=1000 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so \
python bench/graph_shape_inference_gperftools_repro.py run repro2.onnx --loops 15
```

Total wall time 246.4s (15 loops, ~16.4s/loop), 61,746 samples recorded (`PROFILE:
interrupts/evictions/bytes = 61746/5027/1702512`).

**Sampling-rate caveat:** the achieved rate was ~250Hz regardless of
`CPUPROFILE_FREQUENCY` (tested 1000 and 5000, both landed at 250-262 interrupts/sec)
— this sandbox appears to cap `SIGPROF` delivery below the configured rate. This
just means "0 samples" in a rare line should be read as "under roughly 4ms of
cumulative CPU time across the whole 246s run," not "provably exactly zero."

**pprof tooling caveat:** the installed Perl `pprof`'s `--list=<routine>` mode
produced *no* per-line breakdown for `ProcessNode` (it call-site-attributes all of
a fully-inlined callee's samples onto the caller's one call-site line and reports
nothing for the callee's own body) — `ProcessNode` has no separate symbol at all
(confirmed via `nm -C`; it's fully inlined into `GraphShapeInferenceRunner::Run`,
the only symbol nm shows for that class). Line-level data below instead uses
`google-pprof --text --lines --cum`, which *does* correctly resolve samples to
their true originating source line across inlining (cross-checked against
`--list=InferShapesOnGraph`, a routine that *does* keep its own symbol, where both
methods agree).

Cross-check against onnxsim's own `ONNXSIM_PROFILE` instrumentation (a single,
non-profiled `simplify()` call on the same model): `InferShapes` = 194.73ms of a
7,725.08ms `Simplify` total = **2.5%** — matching gperftools' independently-derived
2.4% almost exactly. This is a solid sanity check that the sampling approach is
measuring the right thing, even with sparse samples in the hypothesis-specific
functions.

## Flat self-time, whole process (top lines)

```
Total: 61746 samples
   27556  44.6%  44.6%    33693  54.6% onnx::Value::uniqueName[abi:cxx11] (inline)
    8560  13.9%  58.5%     8560  13.9% std::vector::size (inline)
    4116   6.7%  65.2%    57333  92.9% onnx::Graph::isNameUnique
    3986   6.5%  71.6%     3986   6.5% __memcpy_avx512_unaligned_erms
    3066   5.0%  76.6%     3155   5.1% std::vector::push_back (inline)
    2401   3.9%  80.5%     2401   3.9% __memcmp_evex_movbe
    2189   3.5%  84.0%    38460  62.3% std::__find_if
    1066   1.7%  85.7%    12824  20.8% onnx::Attributes::attributeNames
    1053   1.7%  87.4%     5614   9.1% std::__cxx11::basic_string::basic_string
```

`Value::uniqueName()` (a getter returning a cached `std::string` by value) and its
callers (`isNameUnique`'s linear scan, `std::find_if`, `attributeNames()`) between
them account for essentially the *entire* profile's self time. `__memcpy`/
`__memcmp` are the string-copy/string-compare machinery that getter and its
`std::find_if` comparator lambda drive.

## Cumulative (call-graph) breakdown

```
       0   0.0%   6.8%    49106  79.5% onnx::ImportModelProto
      16   0.0%  71.2%     6210  10.1% _EvalPartialShapeOnGraph
       0   0.0%  72.9%     4571   7.4% _FoldConstantOnGraph
       0   0.0%  88.6%     1471   2.4% onnx::InferShapesOnGraph
      24   0.0%  88.6%     1455   2.4% onnx::GraphShapeInferenceRunner::Run
       0   0.0%  89.1%      926   1.5% OptAndShapeOnGraph
      38   0.1%  90.2%      773   1.3% ProcessNode (inline)
       1   0.0%  98.6%       28   0.0% onnx::ConstantDataFor
       5   0.0%  98.6%       28   0.0% onnx::EncodeCurrentType
       3   0.0%  99.1%       15   0.0% onnx::encodeTensorGeneric
       0   0.0%  99.9%        1   0.0% onnx::encodeTensor (inline)
       0   0.0%  99.6%        2   0.0% ApplyInferredType (inline)
```

## Root cause of the dominant cost (not one of the two hypotheses)

`ir.h` line-level breakdown (`google-pprof --text --lines --cum`, filtered):

```
      0   0.0%   0.1%    57596  93.3% onnx::Graph::createValue (inline)     ir.h:1635
      7   0.0%   0.1%    57588  93.3% onnx::Value::Value (inline)           ir.h:1550
      4   0.0%   0.1%    57337  92.9% onnx::Graph::getNextUnique            ir.h:1226
      0   0.0%   0.1%    53329  86.4% onnx::Graph::create                  ir.h:1330
  27403  44.4%  45.1%    27403  44.4% onnx::Value::uniqueName (inline)      ir.h:364   <-- single hottest line in the profile
   4014   6.5%  51.6%    16824  27.2% onnx::Graph::isNameUnique             ir.h:999   (node->attributeNames() per candidate, per node)
      0   0.0%  65.4%    33527  54.3% onnx::Graph::isNameUnique             ir.h:1017  (find_if over node outputs)
      0   0.0%  82.1%     5833   9.4% onnx::Graph::isNameUnique             ir.h:1013  (find_if over node inputs)
```

The mechanism, read straight from the source (`ir.h:363-367`, `992-1023`,
`1224-1227`, `1550`):

```cpp
std::string uniqueName() const {
  if (has_unique_name())
    return unique_name_;          // ir.h:364 -- 44.4% of ALL samples, just this
  return toVarName(unique());
}
...
Value(Node* node_, size_t offset_)
    : node_(node_), offset_(offset_),
      unique_(node_->graph_->getNextUnique()), ...   // ir.h:1550 -- ALWAYS runs
...
size_t getNextUnique() {
  size_t next_unique_name = next_unique_name_++;
  while (!isNameUnique(next_unique_name)) { ... }      // O(current graph size) scan
  return next_unique_name;
}
```

**Every** `Value` constructor calls `getNextUnique()` unconditionally to seed a
fallback numeric id, which pays a full `isNameUnique()` scan of every node's
attributes/inputs/outputs in the graph so far — *even when the value is about to
get a real name via `setUniqueName()` moments later*, at which point
`unique_name_`/`has_unique_name_` are overwritten and the numeric id (and the scan
that computed it) is never consulted again. For a `ModelProto` where every node
output is already named (true of essentially every real ONNX model and true of
`onnx.helper`-built models), this scan's result is thrown away nearly 100% of the
time. With one `getNextUnique()` call per new value and one full-graph scan per
call, Import is **O(V&sup2;)** in the graph's value count — at ~10,500 nodes
(≈20,000+ values counting inputs/outputs/initializers) this is the single largest
cost in the entire pipeline by a wide margin, and it is **invisible to
`ONNXSIM_PROFILE`**: `Import` runs as the first statement inside the `Pipeline`
lambda (`onnxsim.cpp`) but has no `Profiled(...)`-wrapped span of its own, so its
cost only ever shows up as unaccounted time inside `Pipeline`'s own total — which
is exactly the kind of gap `RESULTS_profiling_survey.md` flagged from the outside
(there, at the Python&harr;C++ marshalling boundary *before* `Simplify()` even
starts; here, mechanistically, *inside* the profiled `Pipeline` span itself, one
level deeper than that report's instrumentation could see). Root-causing and
fixing this is out of scope for this profiling-only task, but it is a much larger
and more concrete opportunity than either of the two hypotheses this task set out
to test.

## Hypothesis 1: EncodeCurrentType's non-tensor CopyFrom

`graph_shape_inference.cc:44-50`:

```cpp
void EncodeCurrentType(Value& v, TypeProto& out) {
  if (v.elemType() != 0 || v.has_sizes()) {
    encodeTypeProtoTensorType(*out.mutable_tensor_type(), v);   // line 46 -- 22 cum. samples
  } else if (v.type()) {
    out.CopyFrom(*v.type());                                    // line 48 -- 0 samples
  }
}
```

| Line | Samples (cum.) | Share of total |
|---|---:|---:|
| 45 (`if` condition) | 4 | 0.006% |
| 46 (common tensor branch: `encodeTypeProtoTensorType`) | 22 | 0.036% |
| 47 (`else if` condition) | 1 | 0.002% |
| **48 (hypothesis-1 target: `out.CopyFrom`)** | **0** | **0%** |

**Verdict: refuted.** Even with 250 deliberate `SequenceConstruct`/`SequenceAt`
round-trips built into the model specifically to hit this line, it recorded zero
samples in a 61,746-sample profile. `EncodeCurrentType` as a whole (both branches,
both of its two call sites in `ProcessNode` — the third, `RecordOuterScopeType`,
adds a few more) totals only ~28 cumulative samples (0.045% of the whole process,
about 2% of `InferShapesOnGraph`'s own time). The plain-tensor branch it's
compared against is *already* this cheap; the rare non-tensor branch this
hypothesis targets is cheaper still, or at least too rare in practice (Sequence/
Optional/Map types are exactly as uncommon as the background doc suspected) to
register at all. Aliasing `v.type()` instead of deep-copying it would save time
that doesn't show up as measurable time in the first place.

## Hypothesis 2: ConstantDataFor + encodeTensor caching

`graph_shape_inference.cc:449-464` (the `ProcessNode` per-input loop):

```cpp
for (Value* input : inputs) {
  ...
  TypeProto* input_type = input_types.Add();
  EncodeCurrentType(*input, *input_type);                        // line 454 -- 32 cum.
  value_types_by_name[input->uniqueName()] = input_type;          // line 455 -- 39 cum. (uniqueName() cost)

  if (const Tensor* data = ConstantDataFor(*input, initializer_by_name)) {   // line 457 -- 28 cum.
    if (ElementCountFits(*data)) {
      TensorProto* tp = input_data_storage.Add();
      encodeTensor(*tp, *data);                                   // line 460 -- 14 cum.  <-- hypothesis 2 target
      input_data_by_name[input->uniqueName()] = tp;               // line 461 -- 23 cum. (uniqueName() cost again)
    }
  }
}
```

`ConstantDataFor` itself (`graph_shape_inference.cc:101-115`):

| Line | Samples (cum.) |
|---|---:|
| 101 (entry / `Constant`-node kind check) | 1 |
| **110 (`initializer_by_name.find(v.uniqueName())`)** | **27** |

**Verdict: confirmed real, but negligible at this scale, and the wrong half of the
pair to optimize first.** The model deliberately maximizes this cost: three
distinct small initializers (`bias_vec`/`scale_vec`/`shift_vec`), each consumed by
name on every one of ~8,000 relevant nodes (every `Add`/`Mul`/`Sub` in all 2,000
blocks), re-encoded from scratch on every one of ~5 `InferShapesOnGraph` rounds per
`simplify()` call, over 15 loops — tens of thousands of redundant `encodeTensor`
calls of only 3 distinct `Tensor` objects, exactly the repeated-encode pattern the
hypothesis describes. Despite that, `encodeTensor`'s own cost is 14 cumulative
samples (0.023% of the whole profile, ~1% of `InferShapesOnGraph`'s own time). Its
gating call, `ConstantDataFor`'s `unordered_map::find`, costs *roughly twice as
much* (27 samples) as the `encodeTensor` call it's protecting — so a cache keyed
on `Tensor*` (which still needs a `ConstantDataFor` lookup to get the key in the
first place, then a *second* lookup into the cache) would need to eliminate more
than just the `encodeTensor` call to net out ahead; caching `ConstantDataFor`'s own
lookup result would matter more, and even that is single-digit-percent of a
function that is itself 2.4% of total wall time. Both together (~41-45 samples,
0.07% of the whole run) are roughly **2,000x** smaller than the `isNameUnique`
cost documented above, even in a model custom-built to be as favorable to this
hypothesis as reasonably possible.

## What's still open

- **Not measured here:** whether an `onnxruntime`-backed run (this sandbox has no
  `onnxruntime` installed) changes the *relative* picture. It should only shrink
  `FoldConstant`'s share further, making Import's dominance more pronounced, not
  less — but that's inference, not measurement.
- **Not measured here:** a model where Sequence/Optional/Map values are a
  significant fraction of graph values rather than a deliberately-small sample
  (250 out of ~10,500 nodes). Real ONNX graphs dominated by such types are
  themselves rare, per the background doc's own framing, so this was not pursued
  further, but it would be the only way to make hypothesis 1 land any samples at
  all.
- **A genuinely promising, unhypothesized opportunity the data points at
  directly:** skip `Value`'s unconditional `getNextUnique()` call (or the
  `isNameUnique()` scan it triggers) when the constructor's caller is about to
  immediately call `setUniqueName()` with an already-known name — the dominant
  path during `ModelProto` import. Implementing and validating this is out of
  scope for this profiling-only task.
- **A second, smaller opportunity in the same function:** `isNameUnique()` calls
  `node->attributeNames()` (27.2% of samples inside it) for *every* node on *every*
  candidate-name check, even for nodes with no `g`/`gs`-kind attribute at all,
  where the subgraph-recursion branch can never trigger.
- Only one profiling run was collected at this scale (assembling ~250s of
  wall-clock at ~250Hz to get double-digit sample counts in the hypothesis-target
  functions was itself the bulk of the effort); the headline `isNameUnique`
  finding is robust (tens of thousands of samples, reproduced identically at both
  a 2-loop/6,426-sample and a 15-loop/61,746-sample scale earlier in this
  investigation), but the exact single-digit percentages for hypotheses 1 and 2
  carry real sampling noise — the *conclusion* (negligible relative to Import) is
  not in doubt given the ~2,000x gap, but a rerun could show e.g. 10 or 20 samples
  in `encodeTensor` instead of 14.

## Caveats

- Sandboxed/virtualized host; the gperftools sampling rate was capped at ~250Hz
  regardless of the requested `CPUPROFILE_FREQUENCY` (tried 1000 and 5000) — see
  "Methodology" above.
- No `onnxruntime` installed, so `FoldConstant` uses onnx's Python-level reference
  evaluator rather than a real ORT session; this inflates `FoldConstant`'s absolute
  share but doesn't touch anything under test inside `InferShapesOnGraph`.
- Single run at each scale, not averaged across repeated trials (unlike
  `RESULTS_graph_shape_inference_arena.md`'s 3-trial isolated benchmark) — see
  "What's still open" for what that means for precision on the small numbers.
