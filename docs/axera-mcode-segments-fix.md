# `mcode.segments` was 4 bytes late in 424 blobs; auditing our MCode patchers against the codec

This follows up #1850, which showed that MCode segments are LZ77 token streams
(`scripts/axera/short_unit_codec.py`, `docs/axera-short-unit-encoding.md`). That work left two items:

1. `mcode.segments` placed segment starts 4 bytes too late in 424 of the 1,063 fixture blobs, and the codec
   compensated with a 0/4-byte shift.
2. Four non-record words turned up in two segments. Both came from blobs that our own patchers wrote.

No Pulsar2 builds or device runs were done for this work. Everything below comes from committed fixtures
and the no-device test suite.

## Root cause: the tail vector's alignment padding

An MCode blob is laid out as:

```
[ FlatBuffers header | seg 0 | seg 1 | ... | seg N-1 | pad | tail vector + tables ]
                     ^ 8-byte aligned                       ^ 4-byte aligned
```

- Each segment is 8-byte aligned.
- The FlatBuffers vector that holds the segment tables only needs 4-byte alignment.
- When the vector lands at an offset that is 4 mod 8, the builder puts **4 zero bytes** between the last
  segment and the vector.

`segments()` found the header end by tiling backwards from the vector:
`header = vec - 8 * sum(key 2 words)`. In padded blobs that range swallowed the padding, so every start
came out 4 bytes late:

| group | fixture blobs | `vec % 8` | old start vs true stream start |
| --- | --- | --- | --- |
| unpadded | 639 | 0 | exact |
| padded | 424 | 4 | +4 |

For example, `adam_update_fp32` has its vector at 1828, so segments start at 288, not 292.

A late start cut each stream's opening `t a7 00 00 XX` apart:

- `t` is the LZ77 literal-run token and `a7 00 00 XX` is the segment header record.
- The old boundary fell between `a7 00 00` and `XX`, so each segment "started" on `XX`.
- The first 4 bytes of the next segment were counted as the tail of the previous one.

The heuristic tokenizer's `A` form ("a lone trailer single abutting the next segment's a7") is that literal
token seen from the wrong side of the boundary.

**Fix.** `segments()` now subtracts the padding. It asserts that the padding is 0 or 4 bytes and all zero.
`tail_padding(mc)` exposes it. `check()` rule 1 compares against `vec - padding`.
`short_unit_codec.stream_start` no longer shifts. It now only confirms that the header record sits where
`segments()` says, and it raises otherwise.

After the fix, every segment in all 1,063 blobs starts exactly on its stream header: a literal token then
`a7 00 00` for compressed segments, `a7 00 00` for raw ones. `tests/test_axera_mcode_segments_fix.py` pins:

- that property for every `.mcode.gz` fixture;
- both padding groups;
- two padded examples (`adam_update_fp32` at 288, `reshape_gather_bwd` at 280);
- two unpadded examples (`attn_qkv_softmax` at 280, `resnet18_int8` at 320);
- rejection of non-zero padding.

### What depended on the old offsets

Nothing that uses absolute offsets changes: fault-injection results, patch offsets and learned maps are
all unaffected. What changes is anything that is relative to a segment or labelled by one. These
expectations moved with the fix; each has a comment in the diff:

| where | old | new | why |
| --- | --- | --- | --- |
| `test_axera_teng2_fault_injection` segment 2 start/end | 380 / 1786 | 376 / 1738 (segment is `[376, 1784)`) | padded blob |
| `test_axera_teng2_fault_injection_add` header | 340 | 336 | padded blob |
| `test_axera_training_graph_setup_sequence` layout | starts 292/356/... | 288/352/... | padded blob |
| `test_axera_training_setup_block_generalizes` V-index | `loss_head_kd` 3, `reshape_gather_bwd` 3, `w2v2fe` 4 | 4, 4, 5 | padded blobs; the opening `a7` record now decodes inside the segment |
| `test_axera_mcode_validator` `_A_COUNTS` | | +1 for 8 padded fixtures | `A` is the opening literal token |
| `test_axera_mcode_validator` `reshape_mul_gap` status quo | 93.8%, `(284,285)` gap | 93.9%, `(488,490)` gap | boundary moved |
| `reshape_calibration_isolation.FIELD_OFFSETS` | 537/550/555/825 | 541/554/559/829 | all 7 builds are padded blobs |

The two fault-injection docs now carry correction notes:

- **Relu.** The "one functional byte, the segment's very first byte" at offset 380 is actually the `XX` of
  the opening `a7 00 00 XX` record, not the segment's first byte.
- **Add.** The `cv3:2320..2323` results belong to the next segment.

## Patcher audit

**Method.**

1. A pytest plugin wrapped every `emit*`/`patch*`/`retarget*` function in the 27 patcher/emitter modules
   under `scripts/axera/`, plus `onnx.save`. The full `tests/test_axera_*.py` suite ran under it with
   Docker and the device disabled.
2. For each MCode returned, and each `_neu` initializer saved, the plugin checked:
   - codec well-formedness: segments tile; each compressed segment decodes exactly `key 5` bytes to whole
     records opening with `a7 00 00`; `key 2` padding is zero; every word is `[verb >= 0xa0][00]...` or zero;
   - decoded segment lengths against the input or template;
   - whether any changed byte is an LZ77 token byte rather than literal payload.
3. Patchers that tests exercise only on synthetic bytes were checked separately on their committed
   reference fixtures.
4. All 1,063 committed blobs were decoded as well. Exactly two fail, and they are the two #1850 found.

| patcher | fixtures / calls checked | result | action |
| --- | --- | --- | --- |
| `patch_scales.find_scale_slots` / `patch_model` | `conv_256to512_tiled_fix/c1x1_emitted_scaled`, `conv_learn_256to512/c1x1_holdout_emitted` | **corrupting** (see below) | skips matches that overlap a token byte |
| `emitter.emit_mcode` (via `emit_axmodel`) | suite uses synthetic streams only; `conv_learn_256to512` references | none observed, but it can write a learned field onto a token byte | refuses (`ValueError`) a field on a token byte; unparseable streams unchecked |
| `tiny_emit.emit_conv_reg8_group`, length-changing configs | `conv_dilation3` ↔ `_rebuild0`; 9 of 20 calls | **corrupting**: +/-2 bytes inside a compressed segment, `key 5` stale, tail vector 2 mod 4 | now reported by `mcode.check()`; tests re-pinned; no codec route (below) |
| `tiny_emit.emit_gemm_reg8_group`, cross-form | `gemm_1x512x1000_tb0` family; 5 of 41 calls | **corrupting**, same way | same |
| `tiny_emit.emit_gemm_reg8_group`, same-form | 8 calls | edits token bytes, but each segment decodes to the donor's records | none |
| `tiny_emit.retarget_tail_vector` | every call above | repoints the header only; cannot make a length change well-formed | docstring warning |
| `tiny_emit`: `emit_neg`, `emit_matmul_reg8_quad`, `patch_output_quad`, `patch_mul_*`, `patch_conv_zp_x`, `patch_site_a`, `patch_matmul_a_scale`, `patch_reshape_gather_scales` | 91 calls | clean: literal payload only, decoded lengths unchanged | none |
| `elementwise_scale_emit.retarget` / `emit` | 29 retargets, 27 saved models | clean | none |
| `conv_learn_wide.patch_mcode_scale_zero_form_a` | `conv256_reference` | clean: all 17 offsets are literal payload | none |
| `conv_learn_256to512` / `conv_256to512_tiled_fix` 3x3 | `c3x3_holdout_emitted`, `c3x3_emitted_scaled` | clean: MCode edits are literal only | none |
| `elementwise_emit`, `compose_emit`, `memory_emit`, `reshape_emit`, `reshape_dma_emit`, `transpose_emit`, `stem_gather_compose_emit`, `gather_{aggregate,compose}_real_check`, `teng2_cluster_patch(_npoint)`, `tinygrad_ax_backend`, `conv_weight_learn_stem` | 173 saved models | clean; `transpose_emit` edits raw segments and tables only | none |
| `memory_emit`, one save | `test_emit_gather_last_axis_rejects_altered_table_tail_and_mcode` | malformed on purpose (negative test) | none |
| `teng2_fault_injection(_add).patch_byte` | `add_1x16x8x8_ref` | corrupting by design | none; see the note in `axera-teng2-fault-injection-add.md` |

### The two #1850 blobs: `patch_scales` overwrote LZ77 tokens

Both blobs are the same ResNet18 `Conv(512,256,1,1)` emission. `patch_scales.find_scale_slots` searches
the raw MCode bytes for a scale's float32/bfloat16 encodings.

- In this reference, the bfloat16 `83 3a` (about 0.0010) occurs 4 times: at 724, 1340, 2916 and 3150.
- Each of those is a **back-reference token and its offset byte**, not a stored scale.
- The patch rewrote each `0x83` to `0x8b`, which lengthens the copy from 6 to 14 bytes.

The effect on the output:

- Segments 0, 1 and 2 decode 8, 8 and 16 bytes longer.
- 277, 233 and 440 decoded bytes differ from the reference.
- Segment 1 ends in 2 non-record words.
- For comparison, a native build of the same holdout differs from the reference in 9, 1 and 34 decoded
  bytes, with identical lengths.
- `mcode.check()` passes all of it.

`docs/axera-conv-256to512-tiled-fix.md` had already isolated the device fault (`0x8030070C`) to the
`patch_scales` step. The scaffold-only emission runs cleanly. That doc measured "1,548 differing bytes from
~608" against a rebuild and put it down to scheduling divergence. The re-decoded segments are a concrete
static cause, but this has not been re-run on the device.

With the fix, `find_scale_slots` finds no slot for this pair. The float32 copies at 3167, 3175, 3183 and
3191 sit in literal payload and are still found. The 3x3 shape's emission only touched literals, which
matches its wrong-but-not-faulting device result. `tests/test_axera_mcode_patcher_audit.py` pins all of
this. The two committed fixtures are kept as evidence.

### Length-changing `tiny_emit` emitters

`emit_conv_reg8_group` (P4 short form ↔ long form) and `emit_gemm_reg8_group` (cross-anchor-form splices)
insert or remove 2 bytes of the compressed stream. `retarget_tail_vector` then repoints the header's
relative offset at the moved vector. Under the real format that is not enough:

- the segment's `key 5` stream length, and possibly `key 2`, go stale;
- the vector ends up 2 mod 4, and a FlatBuffers vector must be 4-byte aligned;
- a 2-byte insertion inside an LZ77 stream shifts every back-reference distance that spans it.

After this change, `mcode.segments` rejects the misaligned vector. The 11 tests that asserted these
outputs were "now clean" pin the real state instead: exactly one `... is not 4-byte aligned` error. Their
class names are kept for history and their docstrings mark them as superseded. None of these outputs had
been run on a device.

They were not routed through the codec. The "P4 short form" and "long form" were identified in compressed
bytes, and the 2-byte difference looks like a literal run versus a back-reference encoding of the same
records, not two record layouts. A codec port would need that re-derived on decoded records first. Then:

1. decode the segment;
2. edit the records;
3. `encode`;
4. rewrite `key 5`/`key 2` and re-lay out the blob.

`short_unit_codec.replace_segment` only handles the case where the padded size is unchanged. That is a new
emitter, not a minimal fix.

## Still open

- **`mcode.check()` still doesn't use the codec.** It is structural and heuristic, so it passes the two
  corrupt blobs above. Adding "every compressed segment decodes `key 5` bytes to whole records" would
  catch this whole class. It's left out here to keep this change minimal.
- **No device re-run.** The link between token-byte overwrites and `0x8030070C` is supported by the
  isolation in `docs/axera-conv-256to512-tiled-fix.md` and the fault-injection split above, but it has not
  been re-tested.
