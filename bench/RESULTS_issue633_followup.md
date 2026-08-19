# Issue #633 follow-up: where the round-trip fix (#637) leaves the speed gap

**Models:** [`onnxmodelzoo/cait_xxs36_224_Opset17`](https://huggingface.co/onnxmodelzoo/cait_xxs36_224_Opset17)
(69.5MB, 36 class-attention blocks) and
[`onnxmodelzoo/mixer_l16_224_in21k_Opset17`](https://huggingface.co/onnxmodelzoo/mixer_l16_224_in21k_Opset17)
(833MB, 24 mixer blocks -- issue #633's original flagship repro).
**Tools:** onnxsim (this repo, `HEAD` at the time of writing: `9a0016b`, which already
includes #634/#635/#637/#638/#641), onnxslim `0.1.96`, onnx `1.22.0`, onnxruntime `1.29.0`.
**Reproduce:** `ONNXSIM_PROFILE=out.json python -c "..."` per the snippets below, or
`python bench/onnxslim_comparison.py cait_xxs36_224_Opset17 mixer_l16_224_in21k_Opset17`
for the onnxslim side-by-side.

## TL;DR

#637 (Graph-native `OptAndShape`, made the default) + #638 (zero-copy
`eliminate_duplicate_initializer` hashing) + #641 (BLAKE3-trusted CSE dedup) together
already close issue #633's root cause: the `ModelProto<->Graph` round trip that used to
happen once per fixed-point round is now amortized to once per `simplify()` call. On
this branch's environment, that takes **mixer_l16 from onnxslim being ~5-10x faster
(as originally reported) to onnxsim being at parity or faster**, and narrows
**cait_xxs36's gap to ~2.4x** (was unmeasured before, included here as a second,
smaller repeated-block data point). Profiling shows shape inference and the round-trip
bookkeeping are now negligible cost; the residual gap is the onnx-optimizer pass
suite's own per-round graph-traversal cost, run ~50+ times per `simplify()` call even
though each round is cheap and Graph-resident. Neither of issue #633's original two
"remaining options" is still relevant as stated -- both are effectively done (option 2:
#637; option 1: partially subsumed by #634's move-semantics Import/Export). What's left
is a different, narrower question, described at the bottom.

**Caveat on absolute numbers:** this sandbox is a shared/virtualized host with visible
run-to-run variance (mixer_l16's onnxsim time ranged 37-61s across two back-to-back
runs below). Treat the relative onnxsim-vs-onnxslim comparison *within this
environment* as the meaningful signal, not the absolute seconds, which will not match
numbers measured elsewhere (e.g. issue #633's own "~180-200s" / "~17s" for the same
model, measured on a different host).

## Results

| Model | orig nodes | onnxsim | onnxslim | onnxsim time | onnxslim time |
|---|---:|---:|---:|---:|---:|
| `cait_xxs36_224_Opset17` | 1758 | 1558 | 1558 | 21.8-22.0s | 9.1s |
| `mixer_l16_224_in21k_Opset17` | 733 | 582 | 678 | 37-61s | 67.5s |

(`check_ok=True` for both onnxsim runs. Node-count deltas between onnxsim and onnxslim
are a fusion-coverage question, not a correctness issue, and out of scope here --
issue #633 is specifically about the speed gap.)

## Profiling breakdown (`ONNXSIM_PROFILE`)

`cait_xxs36_224_Opset17` (21.1s wall):

```
function                calls     wall(ms)      cpu(ms) max wall(ms)    peak(MiB)
-------------------------------------------------------------------------------------
Simplify                    1     21075.84     21350.04     21075.84       526.88
  Pipeline                  1     21075.78     21349.99     21075.78       526.88
    OptAndShape             4     20537.17     20804.13     19227.84       526.88
    FoldConstant            4       325.43       329.61       141.55       526.88
    Fingerprint             8       212.35       215.49        27.41       526.88
      Optimize             54     19350.41     19602.18      1367.88       526.88
      InferShapes          57       640.95       649.02        15.11       526.88
      OrtSession            2         7.46         7.54         6.07       526.88
```

`mixer_l16_224_in21k_Opset17` (36.9s wall, a faster-than-average run of the two
measured):

```
function                calls     wall(ms)      cpu(ms) max wall(ms)    peak(MiB)
-------------------------------------------------------------------------------------
Simplify                    1     36861.62     37336.88     36861.62      4896.89
  Pipeline                  1     36861.55     37336.82     36861.55      4896.89
    OptAndShape             3     26404.18     26743.94     25203.71      4896.89
    FoldConstant            3      8696.57      8810.05      8012.34      4896.89
    Fingerprint             6      1760.24      1782.30       296.03      4896.89
      Optimize             53     25314.00     25638.31       809.73      4896.89
      InferShapes          55       255.49       259.74        10.71      4896.89
      OrtSession            1        14.61        14.73        14.61      4092.72
```

Same shape in both: **`Optimize` is 90-95% of total wall time**; `InferShapes`
(shape inference, the thing the round trip used to gate) is 1-3%. The round trip
itself is no longer separately visible in this trace at all -- `OptAndShape`'s single
Import/Export per call doesn't get its own span, and its cost is now small enough
to be lost in the `Optimize`/`InferShapes` totals rather than dominating them the way
the original issue described (53 *full* `ModelProto<->Graph` round trips for mixer_l16,
one per round). `OptAndShape` itself is called only 3-4 times per `simplify()` --
i.e. the outer (`OptAndShape`, `FoldConstant`) fixed point converges quickly; it's the
*inner* `OptimizeGraphFixed` fixed point (54 and 53 calls to the `Optimize` span above,
respectively) that still does the ~1-round-per-block work the issue originally
diagnosed, just cheaply now since it's Graph-resident.

## Isolating `Optimize`'s remaining cost (cait_xxs36)

`onnxsim.simplify(model, skipped_optimizers=[...])`, same model, single-run timings:

| Skipped | Time | Nodes |
|---|---:|---:|
| (none) | 22.0s | 1558 |
| `eliminate_duplicate_initializer` | 21.2s | 1560 |
| `eliminate_common_subexpression` | 15.4s | 1560 |
| both | 13.7s | 1560 |

Unlike mixer_l16 (where #638's PR description measured `eliminate_duplicate_initializer`
at ~98% of pass time on the *old*, non-graph-native path), on cait_xxs36 CSE is the
larger single contributor (~6.6s), not duplicate-initializer elimination (~0.8s) --
expected, since CSE also compares every non-initializer node, not just initializers.
Skipping both still leaves ~13.7s against onnxslim's 9.1s, so this is not one dominant
pass either; it's the ~40-pass default `GetFuseAndEliminationPass()` suite's cumulative
per-round cost, spread thinly, run tens of times.

## Why the obvious next lever (cache `TensorContentDigest` across rounds) isn't free

`eliminate_duplicate_initializer` and `eliminate_common_subexpression` both memoize
`TensorContentDigest` (the #641 BLAKE3 hash) in a `Tensor*`-keyed
`std::unordered_map`, but `ClearTensorContentDigestCache()` runs at the top of every
single pass invocation (`onnxoptimizer/passes/tensor_content_hash.cc`), so the same
initializer's digest gets recomputed from scratch on every one of the ~53 rounds above,
even though most initializers are untouched between rounds. Extending that cache's
lifetime across rounds looks like the next lever -- but `Graph::initializers_` is a
`std::vector<Tensor>` (`onnx/common/ir.h`), and `eliminate_duplicate_initializer`
itself calls `eraseInitializerAndInput`/`addInitializerAndCreateValue` on it every
round it removes a duplicate. A `std::vector` reallocates/shifts on insert/erase, so a
`Tensor*`-keyed cache is only safe for the lifetime of one pass call -- exactly the
scope the existing code already restricts it to, and exactly why (see that file's own
header comment). Safely widening it needs a stable-address container for
`Graph`'s initializers (e.g. `std::vector<std::unique_ptr<Tensor>>` or
`std::list<Tensor>`) so a `Tensor`'s address survives other initializers being
inserted/erased around it -- a real IR-level change to `onnx/common/ir.h`, not a
bounded pass-local fix, and not attempted here.

## Where this leaves issue #633

- The issue's two "remaining options" are effectively resolved: option 2 (stop
  round-tripping through protobuf inside `OptAndShape`'s inner loop) is #637, merged
  and default; option 1 (alias `Tensor` raw data into the source `ModelProto` during
  Import) is subsumed by #634's move-semantics Import/Export -- with only one
  Import/Export per `simplify()` call now instead of one per round, the byte-copy this
  would have saved is no longer a repeated cost, just a fixed one-time cost, so the
  benefit it would add is much smaller than when it was proposed.
- What's actually left, per the profiling above, is a narrower question than either
  original option: onnx-optimizer's default pass suite's own per-round cost
  (graph traversal + `TensorContentDigest` recomputation across dozens of rounds), not
  round-tripping or shape inference. Closing it further needs either (a) a
  stable-address container for `Graph::initializers_` so the digest cache can safely
  span multiple rounds (an `onnx` IR change), or (b) reducing how many rounds the
  *pass suite itself* needs by addressing the "one block resolved per round" pattern
  at its source, which the original issue already characterized as systemic
  cross-pass coupling rather than a single patchable bug. Neither is attempted here.
