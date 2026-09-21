# AX650 non-fused Reshape: how much of the DMA program is predictable?

[`axera-reshape.md`](axera-reshape.md) found that only the bias flatten fuses;
every convolution and weight Reshape in the ResNet18 step becomes a real DMA
program. This asks whether those programs share an encoding with Transpose
([`axera-transpose-mcode.md`](axera-transpose-mcode.md)) and how much of them can
be generated.

**Short version.**

- The Reshape programs do **not** share an instruction skeleton with Transpose.
- Which program layout a shape gets changes at almost every dimension step (the
  mechanism is not known). Inside one layout the shape enters as a few affine
  immediates and two tensor-size words, so **one narrow family is
  byte-exact**: the weight fold `[Co,8,3,3] -> [1,Co,8,9]` (then `Relu`) for a
  measured set of 14 even `Co` values. Nothing else was reproduced.
- The ResNet18 families (`[C,C,3,3]`, `Cin = Cout`, and the activation folds) change
  tens of independent blocks per step and were **not** reproduced. The
  new emitter does not cover the ResNet18 step.

Method: float32 `Reshape -> Relu` (standalone Reshape does not compile),
MinMax calibration with four uniform +/-0.9 samples, `pulsar2 build
--target_hardware AX650` (Pulsar2 7.0-lite), comparing `*_neu` MCode with the
301-325 noise window ignored. 220 builds, harness
`scripts/axera/reshape_dma_sweep.py`, scratch
`/home/takecheeze/npu-scratch/t_reshape_dma`. The plain `Relu` at each output
shape was built alongside; a pair is "fused" if the two MCodes are equal.

## 1. Is the encoding shared with Transpose?

No. Take the 8-byte substrings that appear in at least 60% of the 174 untiled
Transpose segment-2 streams (604 of them), and drop those that also occur in any
of the `Relu` programs (leaving 133 that are specific to Transpose). Each
Transpose stream holds 121.8 of the 133 on average. Across the 49 Reshape
programs the mean is 0.5 and the maximum 1. In the other direction, 379 8-byte
substrings recur in at least five Reshape streams and are absent from every
`Relu`; only 1 of those 379 occurs in a Transpose stream.

The whole-stream overlap with the Transpose template is about the same for the
Reshape programs (mean 0.42) as for the plain `Relu` programs (0.43), and a
Gather control gives 0.32, so it says nothing beyond the shared instruction
vocabulary. The Transpose decode in `transpose_decode.py` therefore does not
transfer.

One thing does carry over: the loader tail (after the last stream segment)
holds the tensor byte size as a little-endian uint16, the same kind of size
immediate the Transpose sweep found in its stream. See section 4.

## 2. What the sweeps show (Cin = Cout families and the activation folds)

Segment sizes and the number of independent differing blocks (`difflib` over the
segment-2 bytes) between consecutive `C` values, for the three ResNet18 weight
families with `Cin = Cout = C`:

| family | C=8..32 | C=40..64 | blocks changed per step |
| --- | --- | --- | --- |
| fold `[C,C,3,3] -> [1,C,C,9]` | segs 1312/32/32 | 1248/256/32 | 43, 17, 19, 82, 26, 38 |
| flatten `[1,C,C,9] -> [1,1,C,9C]` | 1248, 1152, 1248, 1280 (C=8,16,24,32) | 1216/256/32 | 93, 81, 88, 85, 38, 85 |
| unfold `[1,C,9C] -> [C,C,3,3]` | 1184/32/384 | 1184/256/384 | 45, 16, 72, 76, 26, 91 |

(C = 8, 16, 24, 32, 40, 48, 64.) Even between equal-length programs at C=40
and C=64 the streams differ in 398-519 bytes. There is no small set of fields
to patch.

Activation folds and one dimension varied at a time:

- `[1,C,7,7] -> [1,1,C,49]`, C = 8..64 step 4: MCode length 1152, 1248 or
  1280 and a different layout at almost every step (four steps replace 90-116 bytes
  and insert 44-181 more: C = 16 to 20, 24 to 28, 28 to 32, 36 to 40). The only quiet step is C=52 to 56 (3 bytes: offsets
  549, 812, 937).
- `[N,8,7,7] -> [N,1,8,49]`, N = 2..8: lengths 1344, 1184, 1216, 1344, 1344,
  1344, 1344; N=2 to 3 changes 372 bytes and N=4 to 5 changes 427.
- `[1,8,k,k] -> [1,1,8,k*k]`, k = 4..16: fused for k = 4, 8 and 16, a real DMA
  program for the others, with 444-625 differing bytes between the programs of
  different k that share a layout.

Same pattern as Transpose: a discrete choice of layout (segment sizes and
instruction form) that depends on the whole shape, then immediates that follow
the shape once the layout is fixed. The layout is not predicted here.

## 3. The one clean family: `[Co,8,3,3] -> [1,Co,8,9]`

With `Cin` fixed at 8 and only `Co` varied (`Reshape -> Relu`), the program for
`Co` = 40, 44, 48, 52 and 56 differs in exactly these bytes of the `*_neu` blob
(offsets are absolute; segment 2 starts at 380):

| offset | value | check at Co = 40, 44, 48, 52, 56 |
| --- | --- | --- |
| 936 | `Co - 1` | 39, 43, 47, 51, 55 |
| 1192 | low byte of `8*Co - 1` | 63, 95, 127, 159, 191 |
| 1212 | `Co / 2` | 20, 22, 24, 26, 28 |
| 1344 | `Co` | 40, 44, 48, 52, 56 |
| 2000-2001 | `4*8*9*Co` as uint16 LE (tensor byte size) | 11520, 12672, 13824, 14976, 16128 |
| 2160-2161 | the same value | same |

`npu_params` (40 zero bytes) and `npu_dyn_params` (empty) are unchanged, and the
only other differences between builds are `outputs_info`, the graph dims, and
the noise window. The high byte of `8*Co - 1` is a constant 1 for `Co` in
40..63.

**Validation.** `scripts/axera/reshape_dma_emit.py` retargets the committed
Co=40 build with those formulas. For every `Co` from 30 to 80 that was built:

| Co | prediction vs Pulsar2 build |
| --- | --- |
| 40, 44, 48, 52, 56 | exact (these fitted the formulas) |
| **34, 38, 42, 46, 50, 54, 58, 62, 64** | **exact, held out** |
| 36 | 3 bytes differ (offsets 1620-1622: layout change) |
| 60 | 268 bytes differ (layout change) |
| every odd Co built (41..61): 53 and 57 differ by 5 bytes, the rest by 450-455 | different instruction form at 1209-1213 |
| 32 | 389 bytes differ |
| 66, 68, 70, 72, 80 | 472 bytes differ |

So the exact set is `{34, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 62, 64}`.
It is not an interval (36 and 60 fail between exact neighbours), so
`EXACT_CO` is an explicit set and `emit_weight_fold_axmodel` refuses everything
else, including any `Cin != 8`. Four compiler-built held-out models (Co = 34,
42, 58, 62) are committed as oracles and the tests check that the emitted
model equals each one outside the noise window.

**Device.** With the shared device lock, a control run of the compiler-built
Co=40 model first, then the emitted models for Co = 34, 46, 58, 62 and 64 on
the AX8850 in `axcl-vm`: each matched `relu(x).reshape(1, Co, 8, 9)` in flat
order with maximum absolute error 0.0035, inputs uniform in +/-0.8 (inside
the calibration range). A health run of the control model afterwards was clean.

Why this family is regular while the square weight fold is not is not
established. `Cin = 8` means a single 8-element group on the inner dimension;
the square fold has `Cin` groups that scale with `C`. That is a guess about the
cause, not something measured.

## 4. Where the shape appears

- **Segment 2** holds a handful of immediates that are the dimensions or simple
  functions of them: `Co - 1`, `Co`, `Co/2`, and `8*Co - 1` for this family.
- **The loader tail** holds the tensor byte size as uint16 LE at two places
  (2000 and 2160 in this family), so it also changes with the shape. Whether the
  runtime checks these words was not tested; the emitter writes them so that the
  result equals the compiler-built model outside the noise window.
- Both are affine in the shape; what is not known is which layout a shape gets.

## 5. What is and is not generatable

| case | generatable | evidence |
| --- | --- | --- |
| `[Co,8,3,3] -> [1,Co,8,9]` + Relu, Co in the 14-value exact set | yes | 5 fit + 9 held-out exact vs Pulsar2 builds, 5 device runs |
| the same, any other Co | no (rejected) | 36, 60, odd Co and Co >= 66 change layout |
| `Cin != 8` weight folds, including the ResNet18 `[C,C,3,3]` | no | 17-90 blocks change per step |
| flatten / unfold weight Reshapes | no | same |
| activation folds `[N,C,H,W] -> [N,1,C,H*W]` | no (only the fused pairs of `reshape_emit.py`) | layout re-selected at nearly every C or N |
| tiled Reshapes at training size (batch 16, 64-512 channels) | no | not attempted beyond the sizes in `axera-reshape.md` |
| a neighbour other than `Relu` | not measured | out of scope |

## Not tested

- The tiled regime (batch 16, thousands of elements per row) for any family,
  and whether the tail tables there (12,800 bytes of `npu_params` for
  `[16,1,64,28224] -> [16,1,576,3136]`) follow the Gather byte-offset pattern.
- What selects the layout. The sweeps only bound it: even `Co` alone is not
  enough (36, 60), and being a multiple of 8 is not needed (42, 46 are exact).
- Both neighbour orders: `Relu -> Reshape` was not swept, only
  `Reshape -> Relu`.
- Any Reshape between other op types, which is where the step's Reshapes
  actually sit; a different neighbour changes the MCode (a `Relu` consumer moved
  a Transpose's MCode by 72 bytes).

## Reproduce

```
uv run --no-project --with onnx --with numpy python \
    scripts/axera/reshape_dma_sweep.py build CASES.json WORK_ROOT
```

`CASES.json` is a list of `[[in_shape], [out_shape]]` pairs; the emitter check is
`tests/test_axera_reshape_dma_emit.py`.
