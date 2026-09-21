# How a Gather composes with the ops after it (AX650)

`docs/axera-memory-op-generator.md` retargets standalone one-op models. A
training step is one `neu mode` node whose MCode holds every op, and a Relu
consumer already changed a standalone Transpose's MCode by 72 bytes. This note
measures what happens when the Gather is one segment of a chain, and whether its
indices can still be retargeted. Code: `scripts/axera/compose_emit.py`,
`tests/test_axera_compose_emit.py`.

## Setup

Pulsar2 7.0-lite, AX650, MinMax calibration on four uniform +/-0.9 samples per
input. Chains are prefixes of the ResNet18 conv-backward pattern, all float32
with `x[1,1,4,16]` and eight indices on axis 3:

| chain | ops | graph inputs | output |
| --- | --- | --- | --- |
| 1 | Gather | x | [1,1,4,8] |
| 2 | + Reshape [1,1,8,4] | x | [1,1,8,4] |
| 3 | + MatMul with live `w[4,6]` | x, w | [1,1,8,6] |
| 4 | + Transpose (0,1,3,2) | x, w | [1,1,6,8] |
| 5 | + Add with live `b[1,1,6,8]` | x, w, b | [1,1,6,8] |

Each chain was built with even indices (A = `0,2,...,14`) and with a shuffled
vector with duplicates (B = `15,3,9,0,0,7,12,1`). Chains 3 and 5 were built a
second time with A to measure rebuild noise. Scratch builds are in
`/home/takecheeze/npu-scratch/t_compose`.

## How the MCode composes

It is rewritten at every step, not extended.

- MCode size: 2632, 2632, 3456, 3944, 4472 bytes for chains 1 to 5. The Reshape
  costs nothing in size; MatMul adds 824 bytes, Transpose 488, Add 528.
- Every chain has the same five-segment shape under `mcode.segments`, but the
  segment lengths change each step (for example chain 3: 480/32/1248/480/352,
  chain 4: 64/480/1376/480/672). Comparing whole segments between chain k and
  chain k+1 finds none carried over, except one for chain 1 versus 3. The older
  chain's MCode is not a prefix or subsequence of the newer one.
- Chain 1 is the standalone `[1,1,4,16]` Gather: its MCode matches the
  `memory_emit` template except for 2 bytes, both inside the 301-325 noise
  window. Chain 2 has the same size but is a different program. So the
  `memory_emit` templates hold only when the Gather is the whole graph.

## Rebuild noise is much larger inside a composed graph

Building the same graph twice (same indices, same data) changes:

- chain 3: 1078 MCode bytes; chain 5: 1501 MCode bytes;
- `npu_params`: 0 bytes in both cases.

The standalone Gather's noise window was 301-325 (25 bytes). Composed graphs
also vary near 204-280, 775-885 and later. This means a compiler-built
reference cannot be checked byte for byte against a committed fixture, and the
noise footprint has to be treated as scratch, not signal. The device outputs of
a rebuild had the same maximum error as the fixture's on all eight test draws
(chains 3 and 5), so the noise did not change behavior in these tests.

## Where the index words are

Still the first N uint32 words of `npu_params`, contiguous, for every chain
(the eight-word sequence was found at word offset 0 in all twelve builds). What
follows differs from the standalone Gather:

- chain 1 and 2: ten zero words (72 bytes total), same as standalone;
- chains 3 and 4: 220 bytes; after the indices come six copies of 112.0
  (`0x42E00000`), zeros, six copies of about 0.00389 (`0x3B7EF026`), and zeros;
- chain 5: 244 bytes, with one more constant (about 0.00033) for the Add.

These are quantization constants. They depend on the calibration data, and, for
the Gather output, on which elements the indices select.

## A vs B: what changes with the indices

Chains 1 and 2: 6 MCode bytes, all in the known noise window, and only the
index words in `npu_params`. Same as standalone.

Chains 3 to 5: the params differ outside the index words (chain 3: bytes 96-118;
chain 5 also 160-162). Those are the scale words, because the Gather output and
downstream tensors are calibrated from different elements. Every A-vs-B MCode
difference byte is also a byte that changes between rebuilds of the same graph
(checked for chains 3 and 5), so none of it separates from noise.

So a fresh build with other indices has different calibrated ranges. For the
MatMul output, A calibrated to `[-1.397, 1.787]` and B to `[-1.221, 1.443]`.

## What retargeting does

`emit_gather_in_graph(chain, out, indices=...)` rewrites only the eight index
words. The MCode and the scale words are the reference's. The result is the
reference model with a different Gather selection and **the reference's
calibrated ranges**.

Device check (AX8850, `axcl-vm`, one lock-serialized session, control run and a
health run per chain; inputs uniform +/-0.85):

- Constant-input ramp (`x = c`, `w = 0.85`, so `m = 3.4c`), chain 3. The output
  clips at 1.786 for both the fixture and the emitted-B model, and at 1.442 for
  Pulsar2's own native B build. So the emitted model has exactly A's range
  (1.787) and not B's (1.443). Retargeting keeps the reference scales, as
  designed. Below the clip all three agree with `3.4c` to 0.01.
- Eight random draws per model, maximum absolute error against numpy for the
  requested indices. Most draws sit at the int8 floor (0.008-0.017); a few are
  large (0.07-0.5). Large errors occurred in the fixture, in emitted B and in
  native B alike.
- The large errors are calibration-range saturation. Predicting them from the
  reference's calibrated `[min, max]` (of the MatMul output, and for chain 5 also
  the Add output) matches the observed large-error draws exactly for the
  chain-3 native and emitted B models and the chain-5 fixture. The other three
  cells differ by one marginal draw each, where the range is exceeded by a small
  amount and the error stays under my 0.05 cutoff. An element-level look at the
  worst chain-3 emitted-B draw showed a single output clipped at -1.399 where
  numpy gives -1.722, i.e. at A's calibrated minimum.
- Emitted B on chain 3 has large errors on the same two draws as native B.

All four chains (2 to 5) were emitted and run with two index vectors each (B and
the odd indices) and returned outputs of the right size.

## Consequences

1. Index-only retargeting works inside a composed graph. The index words are
   where they were, and MCode and other scales can be left alone.
2. It is not equivalent to a rebuild. The emitted model has the reference's
   calibrated ranges. Any tensor that exceeds them saturates, exactly like a
   native build that saw narrower data. The reference must therefore be
   calibrated on data whose every downstream range covers the indices you will
   retarget to. For the ResNet18 im2col taps each index vector covers every
   position once per tap, so the gathered range equals the input range; the
   ranges of the later tensors depend on the calibration data and weights.
3. The MCode noise footprint is not fixed to 301-325 once the graph is composed,
   so the strict normalized-MCode check used for standalone templates cannot be
   applied to compiler-built references. `compose_emit` checks graph signature,
   table size, segment layout, and zero padding instead.
4. Inputs must stay within the calibration range.

## Not covered

- Only `x[1,1,4,16]`, eight indices, and these five ops. Other shapes need their
  own fixtures. A real step has hundreds of ops and many Gathers; none of that
  was tried.
- The Transpose and Reshape MCode is still not generated, only carried inside a
  reference.
- The saturation account is a fit on 16 draws (two chains); it is consistent
  with the data but not proven per element beyond the one draw inspected.
- Weights (`w`, `b`) were live graph inputs here. A frozen weight would be in
  `npu_params` and was not tried.
