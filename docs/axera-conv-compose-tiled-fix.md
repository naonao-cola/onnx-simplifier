# Does the tiled scaffold layout fix `docs/axera-conv-compose-real.md`'s composition bug? No -- but a different, real bug was found and fixed along the way, and still isn't the cause.

`docs/axera-conv-compose-real.md` (PR #1783) found that patching a new weight
set into the terminal `Conv(64,64,3,3)` position of a real composed ResNet18
residual-block chain (`Conv -> Add(skip) -> Relu -> Conv`) gave wrong device
output (`max_err 5.29-5.46, mean_err 0.787-0.788` against a `max_err 0.188,
mean_err 0.0136` control) even though the weight-code table was independently
confirmed byte-exact. It isolated the bug to "the scaffold/scale computation
for an internal (non-boundary) tensor" but did not fix it. `docs/axera-conv-
scaffold-arithmetic.md` (PR #1784), landed after, found that the requant
scaffold's apparent "scattered" layout at wide `Cout` is really a per-tile
arithmetic placement, not something that needs bit-permutation learning. This
checks whether that finding explains and fixes PR #1783's bug. Short answer:
no, on both counts below.

## The shape does not tile

`conv_scaffold_arithmetic.tile_width(cin=64, k=3, cout=64) == 64 == cout` --
one tile, the whole channel range. That module's own docstring says this
exact case ("`Cout<=tile_width` (64/64, 1x1 downsample)") "stays one
contiguous span exactly as the already-merged emitters found". Applying the
tiled formula here is mathematically identical to the contiguous formula
`conv_bias_requant.py` (used by both PR #1769's working stage-1 emitter and
PR #1783's failed composition patch) already computes. Confirmed by a single
function call, no build needed -- this rules out the tiling hypothesis for
this specific bug outright.

## A real, independent bug: PR #1783's patch used the wrong weight's scale

Comparing PR #1783's actual patched `npu_params` bytes (still on disk in its
build scratch, `/home/takecheeze/npu-scratch/t_conv_compose_real/
patched_terminal.axmodel`) against the per-channel `M` formula
(`M = x_scale * weight_scale_channel / y_scale`, `docs/axera-conv-weight-
learn-stem.md`'s already-decoded formula) with two different candidate
`weight_scale_channel` sources:

```
M via the NEW held-out weight's own scale (emitter.weight_scales(w_new)):
  [0.00072325 0.00082013 0.00088207 ...]   -- does NOT match the patched bytes

M via the composed build's own quant_axmodel.json (the ORIGINAL, already-
baked-in weight's scale, read by tensor name "w2"):
  [0.00090652 0.00084514 0.0008344  ...]   -- matches the patched bytes
  bit-for-bit, all 8 checked channels
```

The patch computed the scaffold's `M` (and, by the same mechanism, its
quantized-bias term) from whatever weight Pulsar2 had *already compiled* at
that position, not from the new weight actually being written in. `emitter.
weight_scales()` -- the function every working emitter in this project
(`conv_weight_learn.py`, `conv_bias_requant.py`) already uses to get a weight
scale with no build required -- was the correct call; the ad-hoc device-test
script apparently read it from the build's own json instead, an easy mistake
since `conv_weight_learn.build_scales()` makes that one-liner available right
next to the correct API.

This is real: `wrong_m()`/`test_wrong_m_reproduces_the_documented_bug_bit_for_
bit` in `conv_compose_tiled_fix.py` reproduce PR #1783's exact buggy values,
no device access needed. `a1` was also checked as a possible second source of
the same class of bug (a stale/aliased scale) and came up clean -- see below.

## But fixing it did not fix the device output

Built the correction: `conv_weight_learn.emit_biased(standalone_table,
origin, w_new, b_new, x_scale, x_zero=0.0, y_scale, y_zero=128.0,
block_at=36864)` -- which internally calls `emitter.weight_scales(w_new)`,
the correct source -- spliced into a fresh copy of the composed reference at
the same `TERMINAL_OFFSET=37380` PR #1783 located. Verified locally before
any device run: the emitted table's `M` values now match `x_scale *
emitter.weight_scales(w_new) / y_scale` exactly (not the buggy formula), and
the weight-code region is still 0/36864 mismatches (unaffected by the
scaffold fix, as expected).

Device run (AX8850, `axcl-vm`, serialized via the shared lock, real batch-16
input/skip from the composed build's own `quant/debug/io/float/{x,skip}.npy`,
ground truth from a plain float64 numpy convolution of the real chain):

```
CONTROL (chain1 unmodified, terminal=holdout weights): max_err=0.0869 mean_err=0.013545
FIXED   (terminal=w_new, corrected M via weight_scales(w_new)): max_err=5.3250 mean_err=0.788294
HEALTH  (control again, device stayed healthy)       : max_err=0.0869 mean_err=0.013545
```

`FIXED`'s error is statistically indistinguishable from PR #1783's own
buggy-scaffold numbers (`max_err 5.29-5.46, mean_err 0.787-0.788`). **The
w_scale-source bug is real, confirmed, and independently worth fixing in any
future patch script -- but it was not the cause of the wrong device output.**
Something else is.

## What's still open

The `a1`/`r1` aliasing angle -- PR #1783's own leading theory, that an
internal tensor's json-reported scale might not be reliable -- was checked
directly against the composed build's `quant_axmodel.json` and ruled out: `r1`
(the canonical entry) and `a1` (the name `Conv`'s own tensor_config uses for
the same input) both resolve to the identical hash (`1433399308`) and the
identical `{scale: 0.018685635179281235, zero_point: 0.0}` everywhere they
appear, via the json's own `OVERLAPPED`/`ACTIVATED`/`dominator` alias
mechanism. No divergence -- this specific sub-theory is retired.

What remains unconfirmed, and is now the most plausible open explanation:
`requant_block_biased`'s formula (`README.md`'s `M_channel = input_scale *
weight_scale_channel / output_scale`, causally confirmed by a hand-patch
experiment) was derived and validated entirely against a standalone graph-
*boundary* input. The scale *value* for the internal tensor `r1` is confirmed
correct and unambiguous (see above) -- but whether the fused kernel's
epilogue actually consumes that value the same way a boundary-input Conv's
does (same fixed-point width, same zero-point convention, no additional
offset from whatever `Add`+`Relu` computed immediately before it) is not
decoded here. Closing this would need instrumenting or bisecting the fused
compute segment itself, the same class of open problem `docs/axera-dma-
queue.md` and the `teng2` decode attempts already found resistant to shape
sweeps -- this may be the same wall, on a different op.

## Reproduction

```
scripts/axera/conv_compose_tiled_fix.py check
```

No Docker or device required -- `tests/test_axera_conv_compose_tiled_fix.py`
covers all four claims above from the committed fixtures
(`scripts/axera/fixtures/conv_compose_tiled_fix/{w_new,b_new}.npy.gz`, the
same corrected-seed held-out weight set PR #1783 used) plus PR #1769's and
PR #1783's existing fixtures. The device numbers are recorded here as text,
the same way PR #1783 recorded its own (outside CI's reach).
