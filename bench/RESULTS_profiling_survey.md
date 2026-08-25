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
`baseline_seconds` (shard balancing) and the `known_slow` classification. Fixed in
this change (see below) by passing the path straight through, which lets large
models take the C++ core's existing fast path (`C.simplify_path`) instead.

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

## What's still open

`onnx.checker.check_model()`'s own internal `SerializeToString()` (~13s of
`mvp_Opset18`'s ~53s fast-path time in the trial above) is a separate cost, paid
unconditionally by `simplify()` even at `check_n=0`, and is itself invisible to
`ONNXSIM_PROFILE` since it runs entirely in Python via `model_checking.compare()`,
never inside the profiled `Simplify()` call. Unlike the marshalling gap above, this
one isn't obviously avoidable -- the checker call is real validation, not
incidental overhead -- so it wasn't touched here. It's the natural next lever once
the harness's own numbers reflect the fix above rather than being dominated by it:
worth its own profiling pass (does the checker's cost scale with tensor bytes the
same way, and is there a cheaper validation mode for the "just want structural
sanity, not a byte-for-byte round trip" case `simplify()` needs here) once it's
the largest remaining unaccounted-for cost rather than the second-largest.
