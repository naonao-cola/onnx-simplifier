# Does a standalone Transpose template survive real-graph composition?

PR #1758 (`docs/axera-transpose-real-shapes.md`) verified all 23 distinct
real ResNet18-step Transpose `(shape, perm)` pairs standalone, byte-exact on
device, and explicitly flagged composition as untested. This is that test.

## The real neighbours, read from the graph itself

Not assumed from `scripts/axera/legalize.py`'s tap-construction code (which
builds per-tap slices before simplification/CSE collapses them into the
batched forms actually seen in the compiled step) -- read directly from
`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx` with `onnx.shape_inference`,
by locating a real `Transpose` node of each target shape/perm and walking its
producer/consumer edges:

- **Weight family** (`[1,64,64,9]`, `[1,128,128,9]`, ... perm `0,2,1,3`, 16 of
  the step's 41 Transpose instances): `Reshape(w[Cout,Cin,3,3] -> [1,Cin,Cin,9])
  -> Transpose(perm 0,2,1,3) -> Reshape(-> [1,1,Cin,9*Cin])`. `w` is a real
  **trainable graph input** (`resnetv15_stage1_conv3_weight`, confirmed in
  `model.graph.input`, not an initializer), not a constant -- so this chain
  has to be compiled with a live input, the same as production use.
- **Activation family** (`[16,1,R,C]` perm `0,1,3,2`, e.g. `[16,1,576,3136]`,
  25 of the 41 instances): `Reshape(mul_out[16,1,64,9*C] -> [16,1,9*C/R,C]) ->
  Transpose(perm 0,1,3,2) -> MatMul(other[16,1,64,C], transposed)`, i.e. the
  Transpose feeds directly into the tap `MatMul`.

## Composition fuses the whole chain into one opaque unit

Both chains were built standalone with Pulsar2 7.0-lite (their real shapes,
uniform +/-0.9 calibration -- the actual activation/weight range was not
available, so this is an assumption, stated as in every other doc here) and
compared to PR #1758's committed standalone-Transpose templates:

| chain | composed | standalone template | 
| --- | --- | --- |
| weight `[1,64,64,9]` | 1 `neu mode` node, 2440 B MCode, 80 B params | 1 `neu mode` node, 2728 B MCode, 160 B params |
| activation `[16,1,576,3136]` | 1 `neu mode` node, 134,176 B MCode, 10,496 B params | 1 `neu mode` node, 132,104 B MCode, 31,680 B params |

Both composed builds are **a single fused `neu mode` node** -- `Reshape` and
`Transpose` (and `MatMul`, for the activation chain) do not survive as
separate ops in the compiled graph at all, the same fusion PR #1758's own
templates already showed for `Reshape -> Transpose -> Reshape` in isolation.
There is consequently no "the Transpose's own bytes" region to find inside a
composed build and compare against the standalone template -- the comparison
above is deliberately the whole compiled unit vs. the whole compiled unit,
and they already disagree at that level: different MCode length, different
`npu_params` length, and (via `scripts/axera/mcode.py`'s `segments()`)
completely different segment sizes -- the activation chain's compute segment
is 50,112 bytes composed vs. 9,120 bytes standalone, not a patch away. This
is a rewrite, not a concatenation, matching `docs/axera-compose.md`'s
(PR #1732, toy-scale `Gather` chain) finding, now confirmed for a
structurally different op at real training scale.

## But the composed result is numerically correct

Both composed models were run on the AX8850 (`axcl-vm`, serialized via the
shared device lock, a standalone-template control run before and after,
clean):

- **Weight chain**: max error **0.0** against `numpy` (36,864 outputs). Pure
  data movement -- no quantized compute engine touches these values -- so
  there is no rounding to allow for, the same reason PR #1758's standalone
  Transposes were also exact.
- **Activation chain**: max error 0.669 against a peak magnitude of ~79
  (589,824 outputs), 0.85% relative. This chain includes a real quantized
  `MatMul`; that is ordinary int8 accumulation noise for this hardware, not
  evidence of anything composition-specific going wrong -- it is the same
  order of error the project's other real `MatMul`/`Gemm` checks report.

## What this means for whole-step assembly

PR #1758's 41/41 standalone coverage is real, but it is a **disconnected
fact** from whole-step generation, not a reusable building block the way
Gather's index-retargeting is for Gather (`scripts/axera/memory_emit.py`):
Pulsar2 does not leave an op-shaped seam in its compiled output to patch.
What standalone verification *does* still buy:

1. **A compile-ability pre-check.** None of the 23 real shapes failed to
   compile standalone (unlike the stem `Gather`, which needed chunking for an
   OCM budget failure -- `docs/axera-stem-gather.md`). That is one class of
   whole-step-compile risk already ruled out for every Transpose in this
   step.
2. **A numerically-verified oracle.** The standalone templates are a trusted
   reference for what a correct compile of that exact shape/perm computes,
   useful for validating a future whole-step compile op-by-op even though the
   whole step still needs its own from-scratch Pulsar2 invocation.

Whole-step Transpose "coverage" in the sense of a byte-patchable artifact
remains at 0%, unchanged from before this check -- this closes the open
question PR #1758 flagged, rather than raising the coverage number.

## Reproduction

```
scripts/axera/transpose_compose_real_check.py build WORK_ROOT
scripts/axera/transpose_compose_real_check.py check WORK_ROOT
```

`tests/test_axera_transpose_compose_real_check.py` checks the two committed
fixtures (`scripts/axera/fixtures/transpose_compose_real/`) round-trip and
that the two chain-builder functions reproduce the real step's node types,
perms, and shapes -- no Docker/device required.
