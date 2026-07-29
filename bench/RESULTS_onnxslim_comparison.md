# onnxsim vs onnxslim on `swin_s_Opset18` and `FasterRCNN-10`

**Models:** [`onnxmodelzoo/swin_s_Opset18`](https://huggingface.co/onnxmodelzoo/swin_s_Opset18)
and [`onnxmodelzoo/FasterRCNN-10`](https://huggingface.co/onnxmodelzoo/FasterRCNN-10)
(Hugging Face).
**Reproduce:** `python bench/onnxslim_comparison.py swin_s_Opset18 FasterRCNN-10`
**Tools:** onnxsim (this repo), onnxslim `0.1.94`, onnxruntime `1.28`, onnx `1.22`.

## TL;DR

Comparing the two simplifiers surfaced a concrete constant-folding gap in
onnxsim, now fixed in `onnxsim/onnxsim.cpp` (`IsDeterministic`):

| Model | orig | onnxsim (before) | **onnxsim (after fix)** | onnxslim |
|---|---:|---:|---:|---:|
| `swin_s_Opset18` | 12830 | 1295 | **1082** | 1058 |
| `FasterRCNN-10`  | 6370  | 2824 | **2824** | 2622 |

The fix removes **213 nodes (−16 %)** from Swin-S and closes almost the entire
gap to onnxslim (1082 vs 1058), with the simplified model **numerically
identical** to the original (max abs diff `0.0` over random inputs). FasterRCNN
is unchanged (no regression). Genuinely non-deterministic ops (`RandomNormal`,
`RandomUniform`, `Multinomial`, `Bernoulli`, `Dropout`, …) are still never
folded.

## Root cause

onnxsim's constant folder only folds a node when every input is constant **and**
the op is considered deterministic. Determinism was decided by
`schema->GetNodeDeterminism() == Deterministic`.

That test is wrong for a whole class of deterministic ops. ONNX describes some
ops through a **function body**, and `GetNodeDeterminism()` infers a function
op's determinism from its constituent ops — reporting `NonDeterministic` for any
constituent that merely *carries a subgraph* (`Loop`/`If`/`Scan`), and
`Unknown` for context-dependent functions. `Range` is the canonical victim: its
body is a `Loop` (opset < 27) or a context-dependent function (opset ≥ 27), so
its schema determinism is **not** `Deterministic` even though `Range` is a pure
function of its inputs.

Because folding propagates constness through the graph, a single unfoldable
`Range` **poisons an entire otherwise-constant subgraph**. In Swin-S the static
attention-mask construction

```
Range → Slice → Reshape → Expand → Unsqueeze → Concat        (× windows)
ScatterND → ScatterND → … (chained, initializer-fed)
```

is fully determined by the (static) input size and the model's initializers, so
it *should* collapse to constants — but the leading `Range` blocked the whole
chain. onnxslim folds it; onnxsim did not.

## The fix

`IsDeterministic` now folds every default-domain op **except** an explicit
denylist of the genuinely non-deterministic (random/stateful) generators —
`RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`,
`Multinomial`, `Bernoulli`, `Dropout`, `SequenceMap`. This mirrors the set ONNX
annotates with `SetNodeDeterminism(NonDeterministic)` on the operator's *own*
schema (as opposed to the value it *infers* for function bodies), and matches
the behaviour onnxsim had before it switched to the schema attribute. Subgraph
carriers (`If`/`Loop`/`Scan`) remain excluded by the existing `!HasSubgraph`
guard at the call site.

## Swin-S: where the 213 nodes went

Op-type counts after simplification (ops that changed):

| op | orig | onnxsim before | onnxsim after | onnxslim |
|---|---:|---:|---:|---:|
| Unsqueeze | 988 | 63 | **0** | 0 |
| Expand | 297 | 54 | **0** | 0 |
| Range | 198 | 3 | **0** | 0 |
| Where | 220 | 6 | **0** | 0 |
| Equal | 220 | 3 | **0** | 0 |
| ScatterND | 99 | 27 | **0** | 0 |
| Sub | 86 | 3 | **0** | 0 |
| Concat | 525 | 74 | **47** | 47 |
| Slice | 676 | 133 | **124** | 100 |
| Reshape | 440 | 181 | **166** | 262 |

The remaining ~24-node difference vs onnxslim is mostly onnxslim rewriting the
3-D transformer linears `MatMul(x, W) + b` into `Reshape → Gemm → Reshape`
(1 → 97 `Gemm`), a representation change that trades node count for a fused-bias
GEMM kernel; it is orthogonal to the folding fix here.

## Verification

* **Numerical equivalence (Swin-S):** original vs simplified, random input,
  onnxruntime with all graph optimizations disabled → **max abs diff `0.0`**.
* **No regression (FasterRCNN):** identical output (2824 nodes) before and after
  the fix; `onnxsim.simplify(..)` still returns `check_ok=True`.
* **Random ops preserved:** `RandomUniform`/`RandomNormal` graphs are returned
  unfolded.
* **Tests:** `tests/test_constant_fold_determinism.py` (new, torch-free) plus the
  existing `tests/test_simple.py` suite pass.

## Side note: FasterRCNN-10 fails onnxruntime's load-time shape inference

Independently of this change, onnxsim's `FasterRCNN-10` output (both before and
after the fix) fails to load in onnxruntime with a static
`Reshape [ShapeInferenceError]` on the dynamic `3×height×width` input, whereas
onnxslim's output loads. onnxsim's own correctness check passes because it runs
with concrete test shapes. This is a separate, pre-existing issue for
dynamic-shape detection models and is **not** addressed here.
