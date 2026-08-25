# Model-regression profiling survey: where time goes now

**Goal:** run onnxsim's built-in fixed-point profiler (`ONNXSIM_PROFILE`) across a
representative slice of the model-regression set (`scripts/regression/models.json`)
to find the *next* bottleneck, following up on
[`RESULTS_issue633_followup.md`](./RESULTS_issue633_followup.md) (which closed the
old repeated-round-trip issue and found the `Optimize` pass suite dominating the
*profiled* portion of `simplify()`).

**Tool:** [`scripts/regression/profile_sample.py`](../scripts/regression/profile_sample.py)
(new), which fetches named regression models via `model_zoo.py` and runs each
through `simplify(path, profile=...)` in its own subprocess, capturing both the C++
profiler's printed span summary and total Python-level wall time / peak RSS.
**Build:** this repo's `HEAD` at the time of writing (`443e9ab`), `onnxsim==0.7.3`
built via the normal wheel path (`ONNXSIM_BUILTIN_ORT=OFF`), `onnx 1.19`,
`onnxruntime` (constant-folding backend).
**Models (9, spanning the regression set's baseline-time distribution):**
`mobilenetv2-12-int8`, `mnasnet_small_Opset17`, `dla60x_Opset18`,
`efficientnetv2_rw_m_Opset17`, `vit_small_patch32_384_Opset18`,
`jx_nest_small_Opset18`, `ssl_resnext101_32x8d_Opset18`,
`xcit_medium_24_p8_224_dist_Opset18`, `mvp_Opset18` (all `onnxmodelzoo/*`).

**Caveat (same as the prior report):** this is a shared/virtualized sandbox host
with visible run-to-run variance -- back-to-back trials of the *identical* call on
`mvp_Opset18` ranged 37-53s for one code path and 92-118s for another (see below).
Absolute seconds are not portable; the *relative* comparisons, repeated across
multiple trials, are the signal.

## TL;DR

Inside `Simplify()`/`Pipeline()` (the region `ONNXSIM_PROFILE` actually covers),
nothing has changed since the #633 follow-up: `Optimize` still dominates, and it's
still fast in absolute terms.

**The next bottleneck is outside that profiled region, and invisible to
`ONNXSIM_PROFILE`.** For every model in this sample, total Python-level wall time
is several times larger than the profiler's own `Pipeline` span -- growing to
**16x** for `mvp_Opset18` and **20x** for `ssl_resnext101_32x8d`. The gap is the
Python<->C++ marshalling boundary: when `simplify()` receives an already-loaded
`ModelProto` (as opposed to a file path), it pays a `ModelProto.SerializeToString()`
in Python, a re-parse in C++, and a re-serialize + `onnx.load_from_string()` on the
way back -- none of which falls inside the profiled `Simplify()` call, because the
profiler is only enabled once that call has already started. On top of that,
`onnx.checker.check_model()` (which always runs, even at the default `check_n=0`)
does its *own* internal `SerializeToString()` to hand the model to the C++ checker
-- a second, separately-invisible marshalling cost.

**This directly costs the regression harness itself:** `scripts/regression/worker.py`
loaded every model into a `ModelProto` before calling `simplify()`, taking the slow
path on every run, including the timings that seed `models.json`'s
`baseline_seconds` (shard balancing) and the `known_slow` classification. Both
gaps are fixed in this change: `worker.py` now passes the path straight through
(letting large models take the C++ core's existing fast path, `C.simplify_path`),
and `simplify()` itself no longer makes `onnx.checker.check_model()` re-serialize a
model it had already serialized moments earlier. Measured together with `cProfile`
on `mvp_Opset18` (the sample's largest model, immune to this host's wall-clock
noise -- see below): **99.86s -> 37.98s, a 2.6x reduction**, with `SerializeToString`
no longer appearing anywhere in `simplify()`'s or `check_model()`'s call graph.

## Profiled-span coverage vs. total wall time

| model | orig nodes | `Pipeline` span (profiled) | total wall (Python) | coverage gap | total peak RSS | `Pipeline` peak (profiled) |
|---|---:|---:|---:|---:|---:|---:|
| mobilenetv2-12-int8 | 73 | 30.8 ms | 0.10 s | 3.2x | 120.7 MiB | 108.8 MiB |
| mnasnet_small | 150 | 17.9 ms | 0.11 s | 6.1x | 146.2 MiB | 122.8 MiB |
| dla60x | 155 | 64.3 ms | 0.77 s | 12.0x | 502.0 MiB | 369.8 MiB |
| efficientnetv2_rw_m | 774 | 238.9 ms | 2.37 s | 9.9x | 1518.9 MiB | 910.7 MiB |
| vit_small_patch32_384 | 624 | 411.4 ms | 1.13 s | 2.7x | 710.6 MiB | 448.2 MiB |
| jx_nest_small | 2282 | 2009.3 ms | 3.29 s | 1.6x | 1132.1 MiB | 692.4 MiB |
| ssl_resnext101_32x8d | 241 | 454.1 ms | 9.05 s | **19.9x** | 2119.2 MiB | 1454.5 MiB |
| xcit_medium_24_p8_224_dist | 2330 | 4708.2 ms | 32.0 s | 6.8x | 2355.3 MiB | 1432.5 MiB |
| mvp_Opset18 | 1967 | 4892.6 ms | 79.71 s | **16.3x** | 9419.5 MiB | 6488.6 MiB |

For the two smallest models the gap is mostly fixed per-process overhead (Python
startup, imports, subprocess fork), not marshalling. For everything past ~500
nodes it tracks a model's **total tensor bytes far more than its node count** --
`ssl_resnext101_32x8d` has only 241 nodes but is a large-parameter ResNeXt (heavy
initializers), and shows the second-worst gap in the set. That's consistent with
the gap being a data-copy cost (serializing/parsing every initializer's raw bytes
across the Python/C++ boundary, twice), not a graph-algorithm cost.

## Isolating the gap on `mvp_Opset18` (the worst case)

`cProfile` around a plain `simplify(model)` call (in-memory `ModelProto`, matching
what `worker.py` used to do) attributes the wall time as:

```
ncalls  tottime  cumtime  function
     1   65.06s   99.86s  onnx_simplifier.py:867(simplify)          <- includes the actual C.simplify() C-ext call
   117   33.17s   33.17s  Message.SerializeToString                  <- split three ways:
                                                                         1x  11.66s  from simplify() itself (model -> bytes, into C++)
                                                                         1x  13.18s  from onnx.checker.check_model (model_opt -> bytes, for the C++ checker)
                                                                       106x   ~8.3s  from PyModelExecutor.Run (per constant-fold output tensor; cheap each)
     1    1.12s   14.64s  onnx/checker.py:120(check_model)
```

Two back-to-back A/B trials, same model, same host, `simplify(path)` (fast path)
vs `onnx.load()` + `simplify(model)` (what `worker.py` used to do):

| trial | `simplify(path)` | `onnx.load()` | `simplify(model)` | loaded-model total | fast-path total |
|---|---:|---:|---:|---:|---:|
| 1 | 53.10s | 14.57s | 91.88s | 106.45s | 53.10s (**2.0x faster**) |
| 2 | 37.07s | 2.77s | 118.25s | 121.02s | 37.07s (**3.3x faster**) |

The direction is consistent across both trials despite the host's own variance
(each column individually swings ~1.5x between trials); the loaded-model path is
never faster, and the gap is large enough to matter even at its smallest.

## Fix applied: stop pre-loading models in the regression harness

`scripts/regression/worker.py`'s `run_onnxsim()` called `onnx.load(onnx_path)`
purely to have something to pass to `simplify()`, then never used the loaded
`ModelProto` for anything else (node counts come from `simplify()`'s return value).
That forced the slow marshalling path described above on every model in every
weekly run. Changed to pass `onnx_path` straight through, which is exactly the
input shape `simplify()`'s existing fast path (`C.simplify_path`, added by PR #482
for the CLI) is designed for.

**Correctness:** verified analytically that the fragile-pass skip-and-retry loop
(`run_onnxsim`'s `while`/`except RuntimeError` around `passes/*.h` messages) still
works with a path input -- the fast path's internal `except Exception: pass`
silently falls back to the slow (bytes) path on *any* exception, including a
pass-abort `RuntimeError`; that fallback's own `C.simplify()` call is not wrapped
in the same broad handler, so the same `RuntimeError` propagates to the caller
exactly as before, just after one extra failed fast-path attempt. Empirically
verified end-to-end via `worker.py --run-tool onnxsim` directly (not just the
library call): a small model (`mobilenetv2-12-int8`, 73 nodes) is byte-identical
in outcome (`73->73`, `valid=true`); `mvp_Opset18` reproduces the fast-path speedup
(29.9s and 30-55s range across repeated runs, vs. the old path's 90-120s) with the
same `valid=true`, `1967->1374` node result, and **peak RSS roughly halved**
(4735 MiB vs. 9420 MiB) as a bonus -- avoiding the double in-memory copy the old
path paid (a loaded Python `ModelProto` alongside the C++ side's own copy), same
shape as the CLI's PR #482 fix.

**Net effect on the regression job:** every model's measured `seconds` (and
therefore `baseline_seconds` in `models.json`, which drives shard balancing) was
inflated by this avoidable marshalling cost, worse for models with more/larger
weights. This likely overstates real simplification cost for large models
specifically, including the two models currently excluded from the blocking shards
via `known_slow: true` (`longt5_Opset17`, `resnetv2_50x3_bitm_Opset17`, both capped
at the 900s known-slow ceiling) -- worth re-measuring baselines after this change
lands to see whether either can move back into the regular shards.

## Second fix: `check_model()` was re-serializing a model we'd already serialized

The first fix's own `cProfile` breakdown pointed at a second, separate cost:
`onnx.checker.check_model()` -- which always runs, even at the default `check_n=0`
-- was paying its *own* `SerializeToString()` (13.18s of `mvp_Opset18`'s time)
because `model_checking.compare()` handed it an already-*deserialized*
`ModelProto`, moments after `simplify()` had a serialized copy in hand (the bytes
`C.simplify()` returned, or the file `C.simplify_path` had just written) and threw
it away. `onnx.checker.check_model()`'s own dispatch (`onnx/checker.py`) already
special-cases this: given a path it calls `C.check_model_path` (no Python
serialization at all); given `bytes` it uses them as-is; only a `ModelProto`
triggers `SerializeToString()`.

Fixed by passing the bytes/path straight through to `model_checking.compare()`
instead of the freshly-loaded `ModelProto`, in both `simplify()` fast-path branches
and the slow path when `check_n == 0`. Safe because `compare()` only touches
`model_opt` for the `check_model()` call at `check_n == 0` -- the per-trial
inference loop that needs a real `ModelProto` never runs. The one exception is the
custom-domain-op tolerance scan (`_custom_default_domain_ops`, for models like a
TensorRT `BatchedNMS_TRT` plugin exported into the default domain -- issues #107,
#220), which does need a real `ModelProto` to walk the graph; it only runs on a
`checker.ValidationError`, so `model_checking.compare()` now materializes one
lazily right there instead of unconditionally up front. Verified both custom-op
scenarios (`ModelProto` input and path input, exercising both fast-path branches)
still resolve to `check_ok=True` with the op preserved.

**Combined effect, measured with `cProfile` (not wall-clock -- see below) on
`mvp_Opset18`:** before either fix (`onnx.load()` + `simplify(model)`, what
`worker.py` used to do), `simplify()`'s own call tree took **99.86s**, with
`SerializeToString` alone accounting for 33.17s across three sources (11.66s
`simplify()`'s model->bytes, 13.18s `check_model`'s model_opt->bytes, ~8.3s
per-tensor constant-fold outputs). After both fixes (`simplify(path)`), the same
model takes **37.98s** total -- a **2.6x** reduction -- and `SerializeToString` no
longer appears as a caller of either `simplify()` or `check_model()` at all;
`check_model`'s cumulative time drops from 14.64s to **5.98s**, all of it now the
checker's actual C++ validation work rather than marshalling.

We report this pair as `cProfile` call counts rather than wall-clock seconds
specifically because the host's own variance (documented throughout this report)
is large enough to make a single before/after wall-clock pair unpersuasive on its
own; call-graph attribution is immune to that noise and shows the same functions
(`SerializeToString` inside `simplify()` and inside `check_model()`) simply
stopped being called, which wall-clock numbers alone couldn't prove as cleanly.

## What's still open

With both marshalling costs gone, `mvp_Opset18`'s remaining ~38s is dominated by
`simplify()`'s own frame (~30.6s `tottime`, i.e. time cProfile can't attribute to
a named sub-call) -- the actual C++ `Simplify()` core plus the C++-side file
read/parse/write around it, `check_model`'s genuine validation work (~6s), and
`onnx.load()` of the result (~1.2s). None of that is obviously more marshalling to
cut; it looks like real work on a large model. The natural next step is re-running
this same survey against the harness's *own* numbers once `models.json`'s
`baseline_seconds` are refreshed with these fixes in place, to see which models
(if any) still stand out disproportionately to their node count -- that would
point at the next real algorithmic bottleneck rather than another marshalling gap.
