# Does a Conv weight-code table survive real-graph composition?

`docs/axera-conv-weight-learn-stem.md` and its siblings validated
`emitter.py`'s bit-permutation weight-code encoding, standalone, at 11 real
ResNet18 Conv shapes. `docs/axera-transpose-compose-real.md` (PR #1763) and
`docs/axera-gather-compose-real-scale.md`/`docs/axera-gather-aggregate-real.md`
already checked whether standalone-verified coverage survives being embedded
with real neighbours, for Transpose and Gather. This is that check for Conv --
never done before.

**Short answer: composition fuses the whole chain into one opaque node, same
as every other op checked. But the *terminal* Conv's weight-code table
(the part `emitter.py`'s bit-permutation learns) survives byte-for-byte, at a
fixed offset, independent of which weights are used. The requantisation
scaffold does not survive the same way -- patching it via the existing
formula, using scales read from the composed build's own quantisation
metadata, produced numerically wrong output.** Neither purely positive nor
purely negative; both halves are load-bearing findings.

## The real neighbours

All 20 real Conv weights in the actual training-step graph
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`) are graph *inputs*
(trainable), not initializers -- confirmed directly by walking every real
`Conv` node's `w`/`b` inputs against `model.graph.initializer` and
`model.graph.input`. **`emitter.py`'s weight-table machinery cannot reach any
of them in that exact compiled artifact, regardless of composition**, for the
same reason `docs/axera-mcode-training-graph-coverage.md` already
established for a different architecture: a trainable weight has no static
`npu_params` region to write into. This closes that direction definitively,
with zero ambiguity, before any build was needed.

What this technique legitimately targets instead -- explicit in
`docs/axera-conv-weight-learn-stem.md`'s own scope section -- is a
**standalone, frozen-weight deployment artifact refreshed from a checkpoint**.
So the composition test below builds an ordinary frozen-weight inference
chain at the real shape and real residual-block topology, the same way the
Transpose and Gather composition checks did, not the live training graph.

The real topology, read from `step.onnx` by walking producer/consumer edges
from a stage-1 `Conv` node (`resnetv15_stage1_conv1_fwd`): `Conv -> Add(skip)
-> Relu -> Conv`, the standard ResNet basic-block residual pattern.

## Build: a real 4-op chain, weights swapped as a control

Two chains, `Conv(64,64,3x3) -> Add -> Relu -> Conv(64,64,3x3)` at the real
`[16,64,56,56]` shape, using the exact reference/holdout weight pairs already
committed by PR #1769 (`scripts/axera/fixtures/conv_weight_learn/`):

* **chain1**: `reference` weights first (interior, Add+Relu fused onto its
  output), `holdout` weights last (terminal, feeds only the graph output).
* **chain2**: the same two weight sets, positions swapped.

Both built cleanly with Pulsar2 7.0-lite (AX650, MinMax calibration,
identical seed-0 calibration data for `x`/`skip` in both), each in well
under a minute -- no OCM/size issue, unlike the stem Gather.

## Finding 1: fuses into one node, same as Transpose and Gather

```
composed graph node types: ['neu mode']
composed graph node count: 1
```

No surprise, given precedent. `npu_params` is 76,676 bytes -- close to but
not exactly 2x the standalone 38,656-byte table (76,676 vs. 77,312), a
~0.8% difference, much smaller than Transpose's composition delta (its
compute segment grew from 9,120 to 50,112 bytes, >5x).

## Finding 2: the terminal Conv's weight-code table is byte-exact, at a fixed offset

Using `emitter.emit_table` to predict what the standalone reference table
looks like for a given weight set, then sliding that prediction across the
composed table and scoring Hamming distance over only the weight-code bytes
(masked via the learned `origin` map -- excludes scaffold/constant bytes):

```
chain1: terminal (holdout) weight -> shift=37380, mismatches=0   / 36864
chain1: interior (reference) weight -> best mismatches=18344     / 36864  (no match anywhere)
chain2: terminal (reference) weight -> shift=37380, mismatches=0 / 36864
chain2: interior (holdout) weight -> best mismatches=18339       / 36864  (no match anywhere)
```

The **same offset, 37380, in both chains**, regardless of which weight set
ended up terminal -- this is a property of chain *position*, not of which
specific weights are used. ~18,300/36,864 mismatches (~50%) for the interior
weight in both chains is what an *unrelated* byte pattern looks like at this
granularity (no real match, not a partial one) -- confirming the interior
Conv's own table genuinely uses a different encoding, not the same one at a
different offset. Reproducible, no Docker required, from
`scripts/axera/conv_compose_real_check.py::check()`.

**Interpretation**: a Conv whose output is consumed by nothing else the
compiler fuses into it (the last op in a subgraph, or any Conv you make a
compile boundary's terminus) keeps its standalone weight-code layout exactly.
A Conv with a fused epilogue (here, `Add`+`Relu`) does not. This is a
materially better result than Transpose's (whose entire chain rewrote, with
no exception) and refines rather than contradicts the general "composition
rewrites" finding from `docs/axera-compose.md`.

## Finding 3: the scaffold does not patch correctly, even for the terminal Conv

The natural next step -- patch a genuinely new weight set into the located
terminal region and check correctness on the card -- did not fully work.

Full pipeline: `emitter.codes_of` + `emitter.emit_table` for the weight-code
bytes (confirmed correct independently, see below),
`conv_weight_learn.requant_block_biased` for the scaffold, with
`x_scale`/`x_zero`/`y_scale`/`y_zero` read via
`conv_weight_learn.build_scales` from the **composed** build's own
`out/quant/quant_axmodel.json` (the terminal Conv's real input there is `r1`,
the `Relu` output -- an internal tensor, not a graph boundary in the
original unfused graph, though the json still reports scale/zero for it by
name).

Two independent weight sets were tried (the first accidentally reused the
same RNG seed as the already-published `holdout` fixture and came out
byte-identical to it -- caught and corrected with a second, verified-distinct
seed; both gave the same result):

```
CONTROL (chain1, unmodified)            : max_err 0.188  mean_err 0.0136
PATCHED (fresh weights, terminal region): max_err 5.29-5.46  mean_err 0.787-0.788
HEALTH check after                      : ok (device stayed healthy both times)
```

Not a calibration-range problem: only ~3e-7 of the true (numpy) output
elements fell outside the composed build's representable `y` range (`[-4.28,
4.24]`, vs. the true output's own `[-4.02, 4.43]`) -- the Gather-style
out-of-range failure mode from `docs/axera-compose.md`/`docs/axera-gather-
compose-real-scale.md` does not explain this.

The weight-code write itself was independently re-verified correct for the
*exact* patched table used in this device test (zero mismatches against the
predicted encoding, same method as Finding 2) -- so the bug is isolated to
the **scaffold computation**, not the code layout. The most likely cause,
not confirmed: the composed build's json-reported scale for an *internal*,
never-materialized-as-a-boundary tensor (`r1`) may not be the same
fixed-point representation the fused kernel's epilogue actually computes
against -- `requant_block_biased`'s formula was derived and validated
entirely against a standalone graph-boundary input's scale.

## What this means

* **Whole-step assembly**: unreachable regardless of any of the above --
  every real Conv weight in the training step is a graph input, not a
  patchable initializer, full stop.
* **Multi-layer frozen-deployment assembly** (the technique's actual scope):
  partially reusable. The weight-code *layout* for a terminal Conv is
  composition-invariant and directly locatable; a full correctness patch
  (weights + scaffold) is not yet safe for a non-terminal position, and even
  for the terminal position the scaffold formula needs more than the
  standalone recipe once the Conv's real input is an internal, fused tensor.
* **Not covered**: mcode (only `npu_params` was checked here); shapes other
  than `Conv(64,64,3,3)`; chains longer than 4 ops; whether the interior
  Conv's different encoding has its own decodable structure (not attempted);
  fixing the scaffold-for-internal-tensor gap.

## Reproduction

```
scripts/axera/conv_compose_real_check.py   # check() reproduces Finding 2 on both fixtures
```

`tests/test_axera_conv_compose_real_check.py` covers the same, no Docker or
device required, from the two committed fixtures
(`scripts/axera/fixtures/conv_compose_real/chain{1,2}.axmodel.gz`) plus the
existing `conv_weight_learn` fixtures. Finding 3's device numbers are recorded
here as text (not automated -- they need the AX8850 and a Pulsar2 build of
the patched model, both outside CI's reach), consistent with how prior
compose-check docs in this project record their own device results.
