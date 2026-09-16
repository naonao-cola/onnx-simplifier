"""First semantic mcode decode work on a training-graph fixture, applied
to `adam_update_fp32.mcode.gz` -- and a correction to how "no mcode
coverage for training graphs" should be read.

**Correction, stated up front.** Training-graph mcode is NOT untouched
territory at the general-grammar level: `scripts/axera/README.md`'s
"Training-graph streams: first distill-step coverage, one live pair
decoded by fault" section (and its follow-ups) already ran the full
structural validator (verb/tag closure, the quintet form, terminal
pairs, trailer singles) against `adam_update_fp32` and six sibling
training/backward fixtures, including real device fault-vs-inert
probes -- e.g. the README's "Adam seg2 `0b 32` + `20`" trailer, which
this file re-locates and confirms present at absolute offset 1184-1187
in `adam_update_fp32.mcode.gz` (segment 2, immediately before that
segment's zero padding), exactly as already documented. `mcode.check()`
already returns `[]` for this fixture. What genuinely has never been
done before this file: decoding what any specific *field* in a
training-graph mcode stream carries -- the kind of work this session
did extensively for Gemm/Conv/MatMul's forward-inference mcode (output-
scale quads, site A/B/C, zero-point literals, shape-derived fields).
That semantic layer is what this file starts.

## Segment layout

`adam_update_fp32.mcode.gz` (2288 bytes, `mcode.check()` == `[]`) has 5
segments: `(292,64)`, `(356,32)`, `(388,832)`, `(1220,32)`, `(1252,576)`.
Segment 2 (832 bytes) is the largest, dense with `S`/`raw`/`B` records
(120/79/40, plus 15 `V`) -- not characterized further here, genuinely
open. Segment 4 (576 bytes) is 70 `V` (verb) records plus 4 `raw` bytes
-- a pure op-program, the same shape this project's Conv work has
already found meaning in.

## Confirmed: the universal setup quad generalizes to a training graph

The cross-op boilerplate quad (`1c 00 00 ff ff a1 00 <id>` x4, stride
8, confirmed previously for Mul/Gemm/Conv/MatMul inference builds,
`tests/test_axera_universal_setup_quad.py`) is present here too, at
offsets `[631, 639, 647, 655]` with the same incrementing id byte
(`0x20, 0x30, 0x40, 0x50`). A fifth op family, and the first
training-graph / FP32-override build, confirms it.

## Confirmed negative: no INT8 scale/zero-point quad anywhere

A systematic scan for any 4-byte value repeating exactly 4 times at
stride 6-9 anywhere past offset 300 finds nothing resembling Mul's
site A/B/C or the output-scale quad (`<f32> a1 00 <id>` families) --
every stride-8 hit beyond the universal quad itself resolves to a
verb's own fixed `bank` byte recurring across a run of ascending-field
verbs (see below), not an independent scale-shaped field. Consistent
with this graph's FP32 `quant.layer_configs` override (README, "Backward-
graph ops" section): there is no MinMax INT8 activation scale being
computed here for the reciprocal-scale mechanism to carry.

## The real finding: segment 4's "setup sequence" is not Conv-specific

`tests/test_axera_conv_k9_segment34_content.py` (PR #1550, merged
*earlier this same session*) decoded Conv's `k=9` kernel-orientation
segment 4 as a one-shot 39-record setup sequence (ascending `field`
writes across banks 2, 4, 3 in turn: bank 2 fields `0x40..0xd0`, bank 4
fields `0x50..0xc0`, bank 3 fields `0x30..0xe0`, each run bracketed by
an `(161,80,1)`-style verb) followed by a repeating 11-verb block.

**That exact setup block recurs, byte-for-byte identical in `(verb,
field, bank)` triples, in this completely unrelated fixture** -- a
training-graph Adam weight-update op, FP32-precision override, not a
Conv, not INT8-quantized, not `k=9`-shaped at all. Comparing
`adam_update_fp32`'s segment 4 `V` records against Conv's own decoded
sequence: from `(168,48,2)` through `(161,96,1)` (35 consecutive
records), the two are identical. Only the first 3-4 records before
that point differ (Conv's segment opens with an extra `(167,0,10)`
header record Adam's does not have). This is strong, independent
cross-op-family, cross-quantization-mode evidence that the setup block
is a genuinely universal per-op-program preamble, not something tied
to Conv, to `k=9`'s regime shift, or to INT8 quantization -- a real
generalization of a finding from earlier in this same session, found
by an entirely different fixture.

## What differs: Adam's own post-setup content is not a clean fixed-count repeat

Past the shared setup block and its own `(163,0,0)`-shaped terminator,
`adam_update_fp32`'s remaining ~32 `V` records form two full variant
blocks (13 and 16 records) plus a third, truncated one (3 records) at
the segment's end -- not byte-identical repeats the way Conv's `k=9`
block repeats its own fixed 11-verb sequence 4 times. Plausibly
per-tensor handling specific to this op (Adam touches several live
tensors: weight, gradient, and this op's `Sub`), but not decoded to
that meaning here -- recorded as a real, reproducible difference from
Conv's own repeat structure, left open.

## What remains fully open

Segment 2's dense `S`/`raw`/`B` content (120/79/40 records) -- almost
certainly where this op's actual arithmetic content lives -- has no
characterization here at all. Segment 0/1/3's smaller boilerplate-
shaped regions are untouched. This file establishes the entry point
(what's already boilerplate, what's genuinely new, and one real
cross-fixture generalization), not a decode of the op's semantics.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


UNIVERSAL_QUAD_BODY = bytes.fromhex("1c0000ffffa100")

# The 35-record shared setup block, in (verb, field, bank) triples --
# identical between adam_update_fp32's segment 4 and
# conv_4c4c_8x8_k9x1's own segment 4 (tests/test_axera_conv_k9_segment34_content.py,
# PR #1550, this session).
SHARED_SETUP_BLOCK = [
    (168, 48, 2),
    (161, 64, 2),
    (161, 80, 2),
    (161, 96, 2),
    (161, 112, 2),
    (161, 128, 2),
    (161, 144, 2),
    (161, 160, 2),
    (161, 176, 2),
    (161, 192, 2),
    (161, 208, 2),
    (161, 80, 1),
    (161, 80, 4),
    (161, 96, 4),
    (161, 112, 4),
    (161, 128, 4),
    (161, 144, 4),
    (161, 160, 4),
    (161, 176, 4),
    (161, 192, 4),
    (161, 80, 1),
    (161, 48, 3),
    (168, 64, 3),
    (161, 80, 3),
    (161, 96, 3),
    (161, 112, 3),
    (161, 128, 3),
    (161, 144, 3),
    (161, 160, 3),
    (161, 176, 3),
    (161, 192, 3),
    (161, 208, 3),
    (161, 224, 3),
    (161, 80, 1),
    (161, 96, 1),
]


class TestAdamUpdateFP32IsWellFormed(unittest.TestCase):
    def test_check_is_clean(self):
        self.assertEqual(mcode.check(load("adam_update_fp32.mcode.gz")), [])

    def test_segment_layout(self):
        _, segs = mcode.segments(load("adam_update_fp32.mcode.gz"))
        self.assertEqual(
            [(pos, length) for pos, length, _ in segs],
            [(292, 64), (356, 32), (388, 832), (1220, 32), (1252, 576)],
        )


class TestKnownLiveTrailerReconfirmed(unittest.TestCase):
    """The README's already-documented, device-probed "Adam seg2 0b32 +
    20" live trailer, relocated in this exact fixture -- not a new
    finding, a cross-check that this file's fixture matches the one the
    README's existing device-probe work already characterized."""

    def test_0b32_20_trailer_present_before_segment2_padding(self):
        data = load("adam_update_fp32.mcode.gz")
        self.assertEqual(data[1184:1188].hex(), "0b320020")
        # Zero padding follows to (near) the end of segment 2 (388+832 =
        # 1220); the last few bytes before 1220 are the next segment's
        # own `a7` marker head, which -- per this codec's established
        # convention -- can start up to 4 bytes before its segment's
        # word boundary, not more padding.
        self.assertEqual(data[1188:1216], b"\x00" * 28)


class TestUniversalSetupQuadGeneralizesToTrainingGraph(unittest.TestCase):
    def test_quad_present_with_incrementing_id(self):
        data = load("adam_update_fp32.mcode.gz")
        found = hits(data, UNIVERSAL_QUAD_BODY)
        self.assertEqual(found, [631, 639, 647, 655])
        ids = [data[i + len(UNIVERSAL_QUAD_BODY)] for i in found]
        self.assertEqual(ids, [0x20, 0x30, 0x40, 0x50])


class TestNoInt8ScaleQuadFound(unittest.TestCase):
    """No occurrence of the established site-A/site-C framing
    (`<f32> a1 00 <id>` x4, stride 8) or the output-scale-quad framing
    (`<f32> 81 <tag2>` x4, stride 7) appears anywhere in this fixture --
    the specific byte shapes those mechanisms use, not just "any
    repeated 4 bytes" (which a verb's own recurring `bank` byte
    trivially satisfies and would otherwise false-positive on, e.g.
    the shared setup block's runs of same-bank verbs)."""

    def test_no_site_a_style_a100_framed_quad(self):
        data = load("adam_update_fp32.mcode.gz")
        n = len(data)
        # Exclude the universal setup quad's own span (631-670): its
        # `... ff a1 00 <id>` tail overlaps this test's `a1 00` probe
        # one byte into the quad and would otherwise self-match.
        quad_span = range(628, 671)
        for i in range(300, n - 6):
            if i in quad_span:
                continue
            if data[i + 4 : i + 6] != b"\xa1\x00":
                continue
            cand = data[i : i + 4]
            if cand == b"\x00\x00\x00\x00":
                # A run of `V` records sharing verb 0xa1 (=161) trivially
                # repeats an all-zero operand plus the next record's own
                # `a1 00` header at stride 8 -- a verb-structure artifact,
                # not a candidate literal value (site A/C always carry a
                # real float32, never all-zero).
                continue
            offs = [i, i + 8, i + 16, i + 24]
            if all(
                o + 6 <= n
                and data[o : o + 4] == cand
                and data[o + 4 : o + 6] == b"\xa1\x00"
                for o in offs
            ):
                self.fail(f"site-A/C-style quad found at {i}: {cand.hex()}")

    def test_no_output_scale_style_81_framed_quad(self):
        data = load("adam_update_fp32.mcode.gz")
        n = len(data)
        for i in range(300, n - 5):
            if data[i + 4] != 0x81:
                continue
            cand = data[i : i + 4]
            offs = [i, i + 7, i + 14, i + 21]
            if all(
                o + 5 <= n and data[o : o + 4] == cand and data[o + 4] == 0x81
                for o in offs
            ):
                self.fail(f"output-scale-style quad found at {i}: {cand.hex()}")


class TestSetupSequenceMatchesConvK9(unittest.TestCase):
    """The 35-record setup block this session decoded for Conv's k=9
    segment 4 (PR #1550) recurs byte-for-byte, as (verb, field, bank)
    triples, in this unrelated training-graph fixture's own segment 4 --
    direct evidence it is a universal per-op-program preamble, not
    Conv- or k=9-specific."""

    def test_shared_block_present_in_adam_segment4(self):
        data = load("adam_update_fp32.mcode.gz")
        _, segs = mcode.segments(data)
        pos, length, _ = segs[4]
        recs = mcode.decode(data, start=pos, end=pos + length, **mcode.FULL_RULE)
        vrecs = [r for r in recs if r["kind"] == "V"]
        triples = [(r["verb"], r["field"], r["bank"]) for r in vrecs]
        # Find the shared block as a contiguous subsequence.
        n = len(SHARED_SETUP_BLOCK)
        found_at = None
        for i in range(len(triples) - n + 1):
            if triples[i : i + n] == SHARED_SETUP_BLOCK:
                found_at = i
                break
        self.assertIsNotNone(found_at, "shared setup block not found in adam segment 4")

    def test_post_setup_content_is_not_a_clean_fixed_repeat(self):
        """Unlike Conv's k=9 segment 4 (4 exact repeats of one 11-verb
        block), Adam's own post-setup content is NOT byte-identical
        repeats -- a real, reproducible difference, left uncharacterized."""
        data = load("adam_update_fp32.mcode.gz")
        _, segs = mcode.segments(data)
        pos, length, _ = segs[4]
        recs = mcode.decode(data, start=pos, end=pos + length, **mcode.FULL_RULE)
        vrecs = [r for r in recs if r["kind"] == "V"]
        triples = [(r["verb"], r["field"], r["bank"]) for r in vrecs]
        n = len(SHARED_SETUP_BLOCK)
        start = next(
            i
            for i in range(len(triples) - n + 1)
            if triples[i : i + n] == SHARED_SETUP_BLOCK
        )
        after = triples[start + n :]
        # Split into blocks on the (163,0,0)-shaped terminator.
        blocks, cur = [], []
        for t in after:
            cur.append(t)
            if t == (163, 0, 0):
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        lengths = [len(b) for b in blocks]
        # At least two full blocks exist and they are not all the same
        # length -- unlike Conv's own 4 identical 11-record repeats.
        self.assertGreaterEqual(len(blocks), 2)
        self.assertGreater(len(set(lengths)), 1, f"block lengths: {lengths}")


if __name__ == "__main__":
    unittest.main()
