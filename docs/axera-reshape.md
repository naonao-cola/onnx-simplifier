# AX650 Reshape: when it is free, and what is generatable

Reshape is the largest memory-op group in the ResNet18 training step (170 of
1104 nodes). This records what Pulsar2 7.0-lite (AX650) does with it, measured
by compiling one-op-plus-consumer graphs, and what
`scripts/axera/reshape_emit.py` can and cannot generate.

**Short version.** A Reshape either costs nothing (Pulsar2 fuses it into its
neighbour and only the tensor dims change) or becomes a shape-specific DMA
program that is not decoded here. Which one happens depends on the exact
(input shape, output shape) pair, and no closed-form rule was found. Of the
ResNet18 step's Reshape families, only the bias flatten `[1,C] -> [C]` is free;
every convolution and weight Reshape needs real MCode. The emitter covers the
free pairs only.

Method: float32 graphs, `Relu` as the neighbouring op, MinMax calibration with
four uniform +/-0.9 samples, `pulsar2 build --target_hardware AX650`, compared
by byte-diffing the `*_neu` MCode with the known 301-325 noise window ignored.
Scratch builds are in `/home/takecheeze/npu-scratch/t_reshape`.

## 1. Which Reshapes compile to nothing?

**Standalone `Reshape` does not compile.** `x -> Reshape -> y` fails in every
case tried (`[1,4,4,9]`-style shapes, flatten, weight reshapes) with
`NPUBackendError: ZeroDivisionError('division by zero')`. It needs a
neighbouring op, so all measurements use `Relu`.

With a `Relu` neighbour, 50 shape pairs were measured (25 fused, 25 not).

**Fused.** The MCode of `Reshape -> Relu` equals, outside the noise window, the
MCode of a plain `Relu` compiled at the *output* shape. For `Relu -> Reshape` it
equals `Relu` at the *input* shape. `npu_params` (40 zero bytes) and
`npu_dyn_params` (empty) are unchanged, and the `extra_data` metadata blob is
identical. Only the graph input dims (or output dims and `outputs_info`) and
their `value_info` entry differ. The `Relu -> Reshape` order was measured on 10
pairs and agreed with `Reshape -> Relu` on every one (8 fused, 2 not).

Fused pairs include, as `[input] -> [output]`:

- `[1,8,4,4] -> [1,1,8,16]` and the reverse, `[16,4,4,4] -> [16,1,4,16]`
- `[1,C,H,W] -> [1,1,C,H*W]` for `(C,H,W)` in (4,4,4), (8,4,8), (8,8,8),
  (8,16,16), (8,2,2), (16,8,8), (4,4,8), (4,4,16), (4,4,32), (3,4,4), (6,4,4)
- `[1,1,8,16] -> [1,8,2,8]`, `[1,8,16]`, `[1,2,4,16]`; `[1,8,4,4] -> [1,4,8,4]`
- `[1,1,8,48] -> [1,8,48]` and `[1,8,6,8]`, and `[1,8,6,8] -> [1,1,8,48]`
- bias flatten `[1,8] -> [8]`, `[1,64] -> [64]`, `[1,512] -> [512]`

**Not fused** (Reshape adds real MCode, from +96 bytes to about 3x the Relu program):

- any pair touching a trailing dim 3, 5, 7, 9, 12 or 56, e.g. `[8,8,3,3] ->
  [1,8,8,9]`, `[1,4,4,9] -> [1,1,4,36]`, `[1,8,7,7] -> [1,1,8,49]`,
  `[1,4,4,12] -> [1,1,4,48]`, `[1,1,8,12] -> [1,8,4,3]`
- `[1,1,6,16] -> [1,2,3,16]`, although the last dim is 16 and unchanged
- every real ResNet18 shape except the bias flatten (table below)

**No rule.** Every discriminating sweep broke the obvious candidates:

- "last dim is a power of two": `[1,1,8,48] -> [1,8,6,8]` fuses (48 is not a
  power of two, and the split gives 6).
- "no dim is split into non-powers of two": `[1,1,8,48] -> [1,8,6,8]` splits
  48 into 6x8 and fuses, but `[1,1,6,16] -> [1,2,3,16]` splits 6 into 2x3 and
  does not, and `[1,4,4,12] -> [1,1,4,48]` does not.
- "batch dimension changes": `[1,4,4,9] -> [1,1,4,36]` keeps batch 1 and does
  not fuse; `[16,4,4,4] -> [16,1,4,16]` fuses.

The likely cause is memory-layout tiling (fusion holds when source and target
address the same tiles), but I did not decode the layout, so the tables are
lookups of measured pairs, not a predicate. Extrapolating would be a guess.

### ResNet18 step families

Reshape at the shapes the step actually uses, `Reshape -> Relu`, MCode bytes
compared with `Relu` at the output shape:

| Reshape | Reshape+Relu | Relu at output | fused |
| --- | --- | --- | --- |
| `[1,64] -> [64]`, `[1,512] -> [512]` (bias, 17 in step) | same | same | yes |
| `[16,64,56,56] -> [16,1,64,3136]` | 6344 | 3360 | no |
| `[16,1,64,3136] -> [16,64,56,56]` | 6664 | 3648 | no |
| `[16,256,14,14] -> [16,1,256,196]` | 3560 | 3304 | no |
| `[16,512,7,7] -> [16,1,512,49]` | 3240 | 2952 | no |
| `[16,1,128,784] -> [16,128,28,28]` | 4936 | 2816 | no |
| `[16,1,64,28224] -> [16,1,576,3136]` | 32104 | 11104 | no |
| `[16,1,512,441] -> [16,1,4608,49]` | 6632 | 3264 | no |
| `[16,1,256,1764] -> [16,1,2304,196]` | 10824 | 4416 | no |
| `[16,1,128,7056] -> [16,1,1152,784]` | 19816 | 9376 | no |
| `[64,64,3,3] -> [1,64,64,9]` | 2376 | 2080 | no |
| `[1,64,64,9] -> [1,1,64,576]` | 2344 | 1952 | no |
| `[1,64,576] -> [64,64,3,3]` | 2664 | 2144 | no |
| `[512,512,3,3] -> [1,512,512,9]` | 6344 | 3392 | no |

So only the 17 bias flattens of the step's 170 Reshapes are free in this
sense. The rest are real DMA programs. (The step's weights are graph inputs
and its Reshapes are between other op types, not `Relu`; a different neighbour
was not measured, so this is the best available evidence, not a proof for the
real graph.)

## 2. Can the fused ones be emitted by patching metadata?

Yes, and that is what `emit_fused_reshape_axmodel(input_shape, output_shape,
output_path, position="before"|"after")` does. It loads the committed `Relu`
compiled at the reference shape (`fixtures/reshape/relu_<dims>.axmodel.gz`),
validates its structure (single `neu mode` node, `x`/`y`, float32, 40 zero
bytes of `npu_params`, empty `npu_dyn_params`, `outputs_info` matching), and
rewrites only the dims listed above.

- **Oracle equality.** For all 33 fused pairs measured (25 before, 8 after) the
  emitted model equals the compiler-built `Reshape`+`Relu` model as a whole
  protobuf, with the 301-325 noise window neutralised. Six compiler-built
  oracles are committed under `fixtures/reshape/oracle_*` and checked in
  `tests/test_axera_reshape_emit.py`.
- **Device.** All 33 emitted models ran on the AX8850 in `axcl-vm` (V3.6.5
  firmware) and matched `relu(x).reshape(out)` in flat order, maximum absolute
  error 0.0035, inputs uniform in +/-0.8 (inside the calibration range). A run
  of an untouched compiler-built model came first; no device fault occurred.
- **Refusals.** Pairs measured not fused raise "compiles to a real DMA program".
  Pairs never measured (including any not in the tables) raise "unmeasured".
  Shape/type errors and unknown `position` values are rejected.

Limits: `Relu` is the only neighbour measured; float32 only; the fused tables
are 33 measured pairs, not a rule; and the emitted model is a one-op program,
not part of a whole training-step MCode.

## 3. How does a real Reshape program vary with shape?

Where Reshape is not fused, the MCode is a DMA program whose size and content
depend on the whole shape pair:

- **Output rank and size-1 dims do not matter.** `[1,4,4,9] -> [1,1,4,36]` and
  `[1,4,4,9] -> [1,4,36]` give the same MCode (only noise-window differences).
- **Everything else does.** Same-length blobs differ by 274 (`[1,8,4,9]` vs
  `[1,8,4,5]`) to 558 bytes (`[1,8,4,9]` vs `[1,4,4,12]`) outside the noise
  window. Lengths are not a simple function of element count:
  `[1,8,7,7] -> [1,1,8,49]` is 2144 bytes but `[1,8,4,9] -> [1,1,8,36]` is 2368.
- **Determinism.** Rebuilding two non-fused models (`[1,4,4,9]` and
  `[1,8,7,7]`) changed only 4-5 bytes, all inside the noise window, so these
  differences are signal.
- **Large tensors carry a table.** For `[16,1,64,28224] -> [16,1,576,3136]`,
  `npu_params` is 12,800 bytes versus 10,240 for the plain `Relu`; its words
  (e.g. 451,584, 903,168, 1,806,336) look like byte offsets and resemble the
  Gather tail tables in
  [`axera-memory-op-generator.md`](axera-memory-op-generator.md). Smaller
  non-fused cases have the ordinary 40 zero bytes and keep everything in MCode.

I did not decode these programs, so there is no emitter for non-fused Reshape.
Doing so needs the same kind of shape sweep that was done for Gather, plus the
instruction encoding of the copy loops, and the result would still be
one-op-plus-consumer, not a whole-graph MCode.

## What is and is not generatable

| case | generatable | evidence |
| --- | --- | --- |
| fused Reshape pair in the measured tables (33 pairs) | yes, by relabelling | oracle equality, device run |
| bias flatten `[1,C] -> [C]` (ResNet18 step: 17 ops) | yes | oracle, device |
| non-fused Reshape (every ResNet18 conv/weight family) | no | MCode is a real DMA program |
| a pair not measured | no, rejected | no closed-form rule |
| standalone Reshape (no neighbour) | no | Pulsar2 fails to compile it |
| a neighbour other than `Relu` | not measured | out of scope |
