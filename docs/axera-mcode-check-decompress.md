# `mcode.check()` decompresses segments; the 1x1 downsample runs on the device again

This follows up `docs/axera-mcode-segments-fix.md` (#1856), which left two items open:

1. `mcode.check()` did not use the LZ77 codec, so it passed the two blobs `patch_scales` had corrupted.
2. The link between those token-byte overwrites and the `0x8030070C` fault had not been re-tested on the
   device.

Both are closed here.

## The check

`check()` now also calls `mcode.codec_violations(mc)`. For every segment with a key 5 (a compressed
segment), it checks the framing that every native fixture has:

| rule | message (prefixed `segment at POS:`) |
| --- | --- |
| key 5 fits in the slot (`8 * key 2`) | `key 5 stream of N bytes overruns its L-byte slot` |
| the slot is key 5 rounded up to 32 | `slot is L bytes, key 5 = N pads to P` |
| the padding after the stream is zero | `padding after the N-byte stream is not zero` |
| the stream opens with a literal token and then `a7 00 00` | `stream does not open with the a7 00 00 header` |
| `short_unit_codec.decode` accepts it (no overrun, no offset beyond history) | `does not decompress (...)` |
| it decodes to whole 8-byte records | `decompresses to N bytes, not whole 8-byte records` |
| every decoded word is a record (`[verb >= 0xa0][00]...`) or zero | `K decompressed word(s) are not records (first at record R); a patch likely overwrote an LZ77 token` |

If `segments()` cannot read the tail table, the check reports that once and skips the codec rules.
The structural rules already report that case.

**Fixture sweep.** All 1,063 committed blobs were checked: `.axmodel` `_neu` initializers and `.mcode`
files.

- 1,061 are clean.
- The only 2 that fail are the known-corrupt `patch_scales` outputs
  (`conv_learn_256to512/c1x1_holdout_emitted`, `conv_256to512_tiled_fix/c1x1_emitted_scaled`).
- Each gets exactly one message:
  `segment at 920: 2 decompressed word(s) are not records (first at record 190); a patch likely overwrote an LZ77 token`.

**A test that pinned the old behaviour.** `test_axera_mcode_patcher_audit.py`'s
`test_the_1x1_holdout_emissions_re_decode_their_segments` asserted `mcode.check(out) == []` for those two
blobs. It now asserts that the check reports them, and that the reference and the native build still
pass.

`tests/test_axera_mcode_check_decompress.py` covers:

- native blobs of several kinds pass;
- the two corrupt blobs fail;
- synthetic corruptions are caught: a bad back-reference offset, non-zero padding, a broken `a7` header;
- a value edit through `short_unit_codec.replace_segment` stays clean;
- the fixed 1x1 scale patch below passes.

## Device retest: `Conv(x[16,256,14,14], w[512,256,1,1], stride 2)`

No Pulsar2 builds were run. The models are:

- **Reference and native holdout:** #1777's committed builds.
- **`npu_params`:** `conv_256to512_tiled_fix.emit_holdout_table` on #1790's committed holdout weights. It
  is byte-identical to #1790's oracle.
- **MCode:** `patch_scales.patch_model` with the builds' quant tables, now fixed by #1856.

The fixed `patch_scales` makes 8 writes:

- The 4 float32 `y` scales at 3167/3175/3183/3191 are real changes.
- The 4 bfloat16 1/x_s slots at 2230..2254 are unchanged (`0d43 -> 0d43`).
- The 4 `w` matches on token bytes that corrupted the old emission are now skipped.

The result passes the new `check()`. It decodes to the native build's segment lengths
`[1536, 1536, 4352, 2176, 5760]`, and no changed byte is a token byte.

**Device runs.** These ran on the AX8850 in `axcl-vm`:

- Each run took `flock /tmp/axcl-device.lock`.
- The card was at 69C.
- The input was uniform(-0.9, 0.9), seed 9999, the same as #1777/#1790.
- The run order was control, native, fixed, scaffold-only, health, then control, `+zp`, health.
- The controls and health runs match each other bit-exactly.
- No run faulted.

| model | vs numpy | vs native device output |
| --- | --- | --- |
| control (reference build, its own weights) | max 0.0225, mean 0.0046 | |
| native holdout build | max 0.0690, mean 0.0047 | |
| **fixed `patch_scales` emission** | max 0.146, mean 0.113 | max 0.146, mean 0.113; every output differs |
| scaffold-only (no MCode patch; #1790's clean run) | max 0.173, mean 0.111 | max 0.169, mean 0.111 |
| **fixed + zero points / 1/x_s via the codec** | **max 0.0690, mean 0.0047** | max 0.0162 (1 LSB) on 0.013% of outputs |

**The `0x8030070C` fault is gone.** The scale-patched emission that faulted in #1790 now runs. Its only
change from the faulting version is that the four token bytes are no longer overwritten. That confirms
the cause #1856 inferred statically.

**It is still not numerically right, and the remaining error is fully explained.**

- A linear fit of the fixed output against numpy gives slope 0.99983 and offset +0.1134. The scale is now
  right: scaffold-only has slope 0.974.
- The offset equals (132 - 125) x y_scale = 0.1135. The MCode still declares the reference build's
  output zero point.
- `patch_scales` patches scales only.

Decoding both MCodes and diffing the records gives the full list of what differs from the native build:

| segment / record | field | emitted | native |
| --- | --- | --- | --- |
| 0/58, 1/61, 2/134 | `0x03d0`, `0x03d0`, `0x1b10` (x zero point) | 128 | 127 |
| 2/479 | `0x1a90` (y zero point) | 125 | 132 |
| 2/140..147 | `0x0f50`..`0x0fc0`, float32 1/x_s | `430daaad` | `430daab7` |
| 0/187..190 | a group of `a2` records | reordered | |

`patch_scales` does find the 1/x_s slot. It only matches the bfloat16 top half, which does not change
here, so the float32 low mantissa stays stale.

**Fixing the remaining records.** Those records were edited on decoded records and re-encoded with
`short_unit_codec.replace_segment`:

- 12 record writes.
- `check()` stays clean.
- The only remaining difference from native in decoded records is the reordered group in segment 0.

That variant (`+zp` in the table) matches numpy as closely as the native build does. It differs from the
native device output by one output LSB on 0.013% of values, which is the scaffold's bias rounding.

## What this changes

- **`mcode.check()` now catches the whole class that #1856 found:** a patch that lands on an LZ77 token.
  It no longer needs a device run to surface it.
- **`Conv(512,256,1,1)` runs on the device** with the full pipeline. Its remaining error is a stale output
  zero point, which is outside `patch_scales`' scope.
- **Recalibrating these values correctly requires writing records through the codec.** The zero points
  and the full float32 1/x_s are decoded record values. Writing them through `short_unit_codec` gives
  output indistinguishable from the native build. A committed tool that does this (a zero-point/float32
  extension of `patch_scales` that works on decoded records) was not added here. The retest's variant
  was a scratch script that matched records by field and old value.

The scratch scripts (`emit.py`, `emit_zp.py`, `device_run.py`, `compare.py`, `record_diff.py`) are under
`/home/takecheeze/npu-scratch/t_mcode_check_decompress`, not committed.
