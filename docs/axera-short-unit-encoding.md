# AX650 MCode "short units": LZ77 segment compression

**Short answer:** the variable-length short units are not a separate
numeric encoding. They are the tokens of a byte-oriented LZ77 compressor
that Pulsar2 applies to each MCode segment. Once a segment is decompressed,
it is a flat array of the familiar 8-byte records
`[verb][00][field][bank][value32 LE]`, and every value appears there as a
whole little-endian word:

- zero points, as integers;
- scales, `1/s`, and scale ratios, as float32 on all eight lanes.

`scripts/axera/short_unit_codec.py` decodes and re-encodes these segments,
and `tests/test_axera_short_unit_codec.py` covers it with no Docker or
device. The work used only our own compiled builds (the committed fixtures),
analysed black-box. No new builds were needed.

## The rule

A segment's token stream is a sequence of two kinds of token. `t` is the
token byte.

| token | meaning |
| --- | --- |
| `t < 0x80` | literal run: copy the next `t + 1` bytes through (1..128) |
| `t >= 0x80`, `b` | back-reference, see below |

A back-reference covers two or three bytes:

- **Length** is `(t & 0x1f) + 3`, so 3..33. When `t & 0x1f == 0x1f`, one
  more byte `e` follows `b` and the length is `34 + e`, up to 289.
- **Offset** is `(b << 2) | ((t >> 5) & 3)`, a 10-bit window of 1..1023
  bytes back in the decompressed output.

So a two-byte back-reference packs its fields as follows:

| field | bits | source |
| --- | --- | --- |
| flag | 1 | bit 7 of `t` |
| offset, low 2 bits | 2 | bits 5-6 of `t` |
| length − 3 | 5 | bits 0-4 of `t` |
| offset, high 8 bits | 8 | `b` |

**Framing.** A segment is compressed exactly when its tail table (see
`mcode.tail_tables`) has key 5.

- Key 5 is the token stream's exact byte length.
- The stream is zero-padded to a multiple of 32 bytes, and key 2 records
  that padded size in 8-byte words.
- Every stream starts with a literal run whose first record is the
  segment's header, `a7 00 00 XX 00 00 00 00`. `XX` is constant per model.

`mcode.segments` tiles segments back from the tail vector. In 424 of the
1,063 blobs this places every segment start 4 bytes late, so a segment
appeared to begin `XX 00 00 00 00 a1 ...` with its `NN a7 00 00` sitting at
the end of the previous one. The offset is always either 0 or 4, and it is
the same for every segment in a blob. `short_unit_codec.stream_start` finds
the true start.

**How this relates to the old short-unit model** (`scripts/axera/README.md`,
`mcode.py`):

- `[p][p+1 bytes]` is a literal run.
- `[tag][reg]` is a back-reference. The "tag" `0x81..0x9f` is a token byte
  with offset low bits 00. The "register byte" is `b`, which is almost always
  even because offsets are mostly multiples of 8 (whole records back).
- The `0x9f` "extra byte" is the length extension.
- Bare pairs are back-to-back back-references.
- Verb-looking bytes `a1`/`c1`/`e1` inside the stream are back-references
  with offset low bits 1/2/3.

The census's zero-point reflow (`docs/axera-teng-register-census.md`,
Relu 128 → 64) reads as follows:

| stream | tokens | offset |
| --- | --- | --- |
| before | `84 22`, then `01 10 1b`, then `83 4a` | 296 |
| after | `84 22`, then `01 10 1b`, then `83 e6` | 920 |

In both, `84 22` copies 7 bytes, `01 10 1b` is the literal register address
`0x1b10`, and the final token copies the 6 bytes that start with the zero
point from an earlier record holding the same value. Only the source changes:
the new value's earlier occurrence sits elsewhere, 920 bytes back. The value
byte never appears in the unit at all. That is why a zero-point change
"re-encodes the stream from that point on".

## Evidence

**Derivation.** The rule was read off one build: `div_s1`'s TENG segment and
segment 0. Two observations fixed it:

- Copying `(t & 0x1f) + 3` bytes at `b * 4` turned the middle of the segment
  into perfectly stride-8 register records.
- Tokens `e1 82`, `e1 65`, `e1 69` fit only with the offset's low bits taken
  from `t`'s bits 5-6.

The framing was read from `toy_training_step`'s segment heads
(`24 a7 00 00 1e 00 00 00 00 ...`).

**Held out: the whole fixture corpus.** This is every `*.axmodel*` and
`*.mcode*` under `scripts/axera/fixtures`: 1,063 blobs. Four `bsp_matmul/v1_*`
files use another container and were skipped.

| check | result |
| --- | --- |
| compressed segments decoded, each consuming exactly `key 5` bytes | **3,346 / 3,346** |
| decoded bytes that are a whole number of 8-byte records | 3,346 / 3,346 |
| decoded segments starting with the `a7 00 00` header record | 3,346 / 3,346 |
| 8-byte words that are not `[verb >= 0xa0][00]...` or zero padding | 4, see below |
| tokens decoded | 3,774,826 back-references, 3,513,532 literal runs |
| offset low bits 0 / 1 / 2 / 3 | 3,736,507 / 20,311 / 1,591 / 16,417 |
| back-references using the length extension | 1,071,538 (extension byte up to 255) |
| largest offset, longest literal run | 1023, 128 (the window and field limits) |
| `key 2 × 8 == round_up(key 5, 32)`, padding all zero | every compressed segment |

In total, 17.2 MB of token streams decode to 268.8 MB of records.

The four non-record words are in 2 segments, and both belong to blobs that
our own in-place patchers emitted, not to native builds:

- `conv_256to512_tiled_fix/c1x1_emitted_scaled`
- `conv_learn_256to512/c1x1_holdout_emitted`

Their reference and native counterparts decode cleanly. Those patchers edited
compressed bytes without a decoder, so they are worth re-checking.

**Known values reproduce exactly.** 67 builds have committed quantization
parameters: the census's Div/ReduceSum/Sqrt, the binary-op Add/Sub/Mul/Div,
and the elementwise-emit Relu/Sqrt templates and oracles. For each, every
product `s_x^a s_y^b s_z^c` with exponents in {-1, 0, 1} was checked against
the decoded records as float32, within 1 ulp:

- **All 67 builds** carry `1/s_x` and `s_y` on all eight lane registers
  `0x0f50..0x0fc0`.
- **Values the census could only see as "short units"** are ordinary lane
  records:
  - Div `s_y`, and `s_x/(s_y·s_z)` (e.g. 102.0 in `div_s1`, 72.857 in
    `div_r3`);
  - ReduceSum `s_x/s_y`;
  - Sqrt `s_x`.
- **Binary ops:**
  - Add/Sub/Div `1/s_z` is an eight-lane float. #1839 had found it "does not
    appear as a float".
  - Mul's divisor `s_y/(s_x·s_z)` sits at `0x0fd0..0x1000`.
- **Zero points are whole records.**
  - Div: `zp_x` and `zp_y` are written to `0x1b10` (e.g. 128/204 in
    `div_s1`, 128/219 in `div_r3`, 64/146 in `div_asym`), and a second copy
    is written to `0x1a90`.
  - Relu `x0_y0` vs `x128_y128`: `0x1b10` goes 0 → 128. The nonzero build
    also adds register writes, starting with `0x1eb0 = 128`, and its TENG
    program grows by 32 bytes.

**A whole program's calibration dependence is now visible.** `add_c2` and
`add_c3` have the same structure: 2,912 decoded bytes each. Their TENG
programs differ in just 28 records:

- three eight-lane floats, `1/s_x`, `1/s_z` and `s_y`;
- two swapped small counts, `0x03d0` and `0x02b0`.

That is the entire compressed-side calibration footprint of Add. The Q15
`npu_params` words the census found are outside MCode.

## Encoder

`encode` round-trips every stream: `decode(encode(raw)) == raw` for 3,240
of 3,240 segments tried. The 106 segments above 20 KB decoded were skipped
only to save time.

It stays inside the envelope that native streams never leave (0 exceptions
in 3.77 M native back-references):

- offset ≥ 25;
- length ≥ 4;
- no copy overlapping its own output (length ≤ offset);
- at least 12 literal bytes at the end.

A copy outside this envelope decodes in software, but it has never been
shown to decode on the device, so the encoder does not emit one.

Parsing is greedy: take the longest copy, with ties going to the nearest
offset. Against Pulsar2's own bytes:

| result | segments |
| --- | --- |
| byte-identical | **2,454 / 3,240 (75.7 %)** |
| same length, different choice | 780 |
| shorter than native | 6 |
| longer than native | 0 |

Where it differs, it is in which of several equal-length earlier copies a
back-reference points at, typically the last one before the tail literals.

`replace_segment(mc, i, raw)` re-encodes segment `i`. It writes the new stream
and its zero padding in place and rewrites key 5, but only when the result
pads to the same 32-byte slot. Otherwise it raises, because the blob would
need re-laying out. Tests exercise it offline:

- **Zero-point edit.** `div_s1`'s `zp_x` goes 128 → 100. This is the
  "value already occurs earlier" case that broke in-place literal patching
  in `docs/axera-elementwise-scale-emit.md`.
- **Step recalibration, fits.** `step_recalib/toyf_A1` → `toyf_Dmom100`:
  their TENG programs decode to identical lengths (43,712 bytes) and differ
  in 128 whole records. Re-encoding A1's slot with Dmom100's content fits,
  and it decodes back exactly.
- **Step recalibration, refused.** `toyf_Bx2` differs in 570 records, and
  its stream is 17,324 bytes, 108 over A1's 17,216-byte slot. Native Bx2 is
  17,324 bytes too.

## What this unblocks

1. **Zero points.** Readable now, as whole integer records.

   - A nonzero → nonzero change is a value edit: decode, edit, re-encode,
     replace.
   - Zero ↔ nonzero adds or removes register writes, e.g. Relu's
     `0x1eb0`. That is a structural change, so it remains a template-key
     question, not an encoding one.

2. **Binary-op scale ratios.** Unblocked for the refusal's stated reason. The
   ratio-dependent fields are ordinary float records, not variable-length
   units, so for a fixed structure an emitter needs only to write the new
   `1/s_x`, `1/s_z` and `s_y` (or Mul's divisor) and re-encode. Two limits
   remain:

   - The structure itself still depends on calibration: `add_c1`, with
     `s_x == s_z`, lacks an eight-record rescale block that `add_c2`/`add_c3`
     have.
   - Add/Sub's Q15 `npu_params` words need writing too.

3. **Whole-step recalibration.** Item 1 of `docs/axera-step-recalibrate.md`'s
   "What would turn this into a working recalibrator" was an encoder for
   TENG's compressed register writes. That is this codec: decode, rewrite
   records, re-encode, re-pad and re-count. The Dmom100 variant is
   re-encoded into the reference's slot.

   Still missing:

   - re-laying out a blob when a stream outgrows its 32-byte-padded slot
     (Bx2), which means shifting later segments and the tail table;
   - that doc's (d) range-dependent tables and (b′) field map. Those are
     content questions, not encoding.

## Not explained / not verified

- **Nothing re-encoded has run on a device.** The encoder stays inside the
  native envelope, and `replace_segment` changes only the stream, its padding
  and key 5. Whether the runtime checks anything else (for example a
  checksum, or table keys 0/1/3/4, whose meaning is unknown) is untested.
- **Native candidate choice.** Pulsar2 sometimes prefers a farther copy among
  equal-length ones, especially just before the tail literals. The exact
  match-finder is not reproduced, which leaves 24 % of re-encodes differing
  from native at equal length.
- **The 4-byte tiling offset.** Why `mcode.segments` is off by 4 in some
  blobs is a FlatBuffers-layout detail that was not chased. The codec detects
  it from the header record.
- **Re-layout.** Growing a segment past its padded slot is not implemented.
