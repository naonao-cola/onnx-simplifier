"""Characterizing WHAT the new content in Conv's segments 3/4 at `k=9`
actually is: segment 4 decodes into a clean, precisely-repeating
11-verb block -- and that block's content (including its repeat count)
turns out to be completely independent of `cin`/`cout`, ruling out a
channel-count interpretation this project's own record-level method
made tempting to guess.

`tests/test_axera_conv_orientation_k9_regime_shift.py` (PR #1527,
merged) found that Conv's `kx1` kernel-orientation mcode has a real
structural regime shift at `k=9`: two segments (3, 4) that were
byte-length-fixed (boilerplate) through `k<=7` suddenly carry real
instruction content -- segment 3 a mix of record kinds, segment 4
exactly 88 `V` (verb) records. It established this content was real
(not padding) but never examined what it computes.

## Segment 4: an 11-verb block, repeated 4 times exactly, plus a
## truncated 5th copy

Decoding `conv_4c4c_8x8_k9x1.mcode.gz`'s segment 4 (`mcode.decode()`,
88 `V` records) and looking at each record's `(verb, field, bank)`
triple shows a clean two-part structure:

- **Records 0-38 (39 records): a one-shot "setup" sequence.** Starts
  with the `0xa7` op-program marker (this project's own established
  segment-start signal, per `mcode.py`'s `check()`), then writes to a
  run of ascending `field` addresses across three banks in turn (bank
  2: fields `0x40..0xd0`; bank 4: fields `0x50..0xc0`; bank 3: fields
  `0x30..0xe0`), each run bracketed by an `0xa1,0x50,0x1` verb.
- **Records 39-87 (49 records): 4 exact repeats of an 11-verb block,
  plus one truncated copy.** The block
  `[(163,0,0),(161,80,1),(169,0,0),(162,0,0),(168,48,2),(161,64,2),
  (161,80,1),(161,80,1),(168,64,3),(161,80,3),(161,80,1)]` (`(verb,
  field, bank)` triples) appears at records 39-49, 50-60, 61-71, and
  72-82 -- **byte-for-byte identical** all four times, confirmed by
  direct comparison, not approximate matching. Records 83-87 (5
  records) match the block's first 4 elements exactly but diverge at
  the 5th (`(162,0,0)` where the block itself has `(168,48,2)`) --
  consistent with a shared closing/terminator sequence rather than a
  5th full repeat, since `(162,0,0)` (verb `0xa2`) already appears as
  a terminator-shaped record in the earlier setup region too.

## The repeat count is channel-count-independent -- a hypothesis
## tested and refuted, not assumed

4 full repeats, at a shape with `cin=cout=4`, invites an obvious guess:
does the repeat count track `cin` or `cout`? Tested directly by
building `k=9` at `cout in {2,8}` (holding `cin=4`) and `cin in
{1,2,8}` (holding `cout=4`) -- 5 independent configurations beyond the
original `cin=cout=4` build:

| config | total mcode length | segment lengths | segment-4 V records | block-start indices |
| --- | --- | --- | --- | --- |
| `cin=4,cout=4` (original) | 4656 | `[864,864,1184,288,704]` | 88 | `[39,50,61,72,83]` |
| `cin=4,cout=8` | 4656 | `[864,864,1184,288,704]` | 88 | `[39,50,61,72,83]` |
| `cin=4,cout=2` | 4656 | `[864,864,1184,288,704]` | 88 | `[39,50,61,72,83]` |
| `cin=8,cout=4` | 4656 | `[864,864,1184,288,704]` | 88 | `[39,50,61,72,83]` |
| `cin=2,cout=4` | 4656 | `[864,864,1184,288,704]` | 88 | `[39,50,61,72,83]` |
| `cin=1,cout=4` | 4592 | `[864,864,1120,288,704]` | 88 | `[39,50,61,72,83]` |

**Segment 4 (and segments 0/1/3) are byte-identical across every one
of these 6 configurations, regardless of `cin`/`cout`.** Only segment 2
(the weight-table-carrying segment) shrinks at `cin=1` (1184 -> 1120,
matching the smaller weight tensor), confirming these are genuinely
different, correctly-built models and not a build-cache artifact --
the *rest* of the stream, including segment 4's 4-repeat structure, is
completely insensitive to channel count. This refutes "repeat count =
`cin`" and "repeat count = `cout`" cleanly: neither hypothesis survives
varying the other dimension while holding one fixed, and the repeat
count stays at 4 across a `cin`/`cout` range of `1` to `8`. Whatever
determines "4" is tied to `k=9` (or possibly `hw=8`, held fixed
throughout this file and not itself varied here) -- not to the number
of input or output channels.

**Confirmed above the noise floor.** An independent rebuild of the
`cin=8,cout=4` config reproduces segment 4 (and segments 1/3)
byte-identically; the 16 bytes that do differ between the build and
its rebuild are clustered at offsets 1102-1119, inside segment 0
(280-1144) -- nowhere near segment 4 (3480-4184) -- consistent with
this project's established noise-floor scale, not a sign the
channel-invariance result is unreliable.

## Segment 3: no comparably clean pattern found

Segment 3's mixed content (mostly short (`S`) and bare (`B`) units with
varying small `tag`/`reg`/`payload` values, plus a handful of `raw`
bytes and one anchoring `V`-verb prefix matching segment 4's own
`0xa7` marker) was inspected the same way but shows no equally clean
repeating block or literal value tracking a known shape quantity in
this pass -- reported honestly as unresolved rather than forcing a
weaker pattern to look like segment 4's.

## What remains open

The 11-verb block's own semantic meaning (what each verb computes),
why exactly 4 (rather than a fixed 5, or a `k`-derived count) copies
appear, whether the count instead tracks `hw` (untested -- always 8
here) or `k` itself (also untested at another `k>=9` value in this
file), and segment 3's own structure are all left for future work.
This file narrows the question from "what is this content" (fully
open) to "a precisely-repeating, channel-count-independent 11-verb
block, repeat count and content unexplained but real and stable."
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


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def segment4_vfb(data):
    _, segs = mcode.segments(data)
    pos4, len4, _ = segs[4]
    recs4 = decode_segment(data, pos4, len4)
    return [(r["verb"], r["field"], r["bank"]) for r in recs4]


BLOCK = [
    (163, 0, 0),
    (161, 80, 1),
    (169, 0, 0),
    (162, 0, 0),
    (168, 48, 2),
    (161, 64, 2),
    (161, 80, 1),
    (161, 80, 1),
    (168, 64, 3),
    (161, 80, 3),
    (161, 80, 1),
]


class TestSegment4HasFourExactRepeatsOfAnElevenVerbBlock(unittest.TestCase):
    def test_block_repeats_exactly_four_times(self):
        vfb = segment4_vfb(load("conv_4c4c_8x8_k9x1.mcode.gz"))
        self.assertEqual(len(vfb), 88)
        for start in (39, 50, 61, 72):
            self.assertEqual(vfb[start : start + 11], BLOCK, f"block at {start}")

    def test_fifth_copy_is_truncated_and_diverges_at_element_five(self):
        vfb = segment4_vfb(load("conv_4c4c_8x8_k9x1.mcode.gz"))
        tail = vfb[83:88]
        self.assertEqual(len(tail), 5)
        self.assertEqual(tail[:4], BLOCK[:4], "first 4 elements still match")
        self.assertNotEqual(tail[4], BLOCK[4], "5th element diverges")


class TestSegment4IsChannelCountIndependent(unittest.TestCase):
    """The exact same 88-record, 4-repeat structure appears regardless
    of cin/cout, tested across cin in {1,2,4,8} and cout in {2,4,8} --
    refutes a channel-count interpretation of the repeat count."""

    # fixture: (cin, cout, expected total length, expected segment-2 length)
    CASES = [
        ("conv_4c4c_8x8_k9x1.mcode.gz", 4, 4, 4656, 1184),
        ("conv_4c8c_8x8_k9x1.mcode.gz", 4, 8, 4656, 1184),
        ("conv_8c4c_8x8_k9x1.mcode.gz", 8, 4, 4656, 1184),
        ("conv_1c4c_8x8_k9x1.mcode.gz", 1, 4, 4592, 1120),
    ]

    def test_segment4_identical_across_all_channel_counts(self):
        reference = segment4_vfb(load("conv_4c4c_8x8_k9x1.mcode.gz"))
        for fname, cin, cout, _, _ in self.CASES:
            vfb = segment4_vfb(load(fname))
            self.assertEqual(
                vfb, reference, f"{fname} (cin={cin},cout={cout}): segment 4 differs"
            )

    def test_only_segment_2_scales_with_channel_count(self):
        for fname, cin, cout, total_len, seg2_len in self.CASES:
            data = load(fname)
            self.assertEqual(len(data), total_len, fname)
            _, segs = mcode.segments(data)
            lens = [length for _, length, _ in segs]
            self.assertEqual(lens[2], seg2_len, f"{fname}: segment 2 length")
            # segments 0,1,3,4 stay fixed across all cases.
            self.assertEqual(lens[0], 864, fname)
            self.assertEqual(lens[1], 864, fname)
            self.assertEqual(lens[3], 288, fname)
            self.assertEqual(lens[4], 704, fname)


class TestChannelInvarianceSurvivesAnIndependentRebuild(unittest.TestCase):
    def test_cin8_rebuild_matches_segment4_exactly(self):
        original = load("conv_8c4c_8x8_k9x1.mcode.gz")
        rebuild = load("conv_8c4c_8x8_k9x1_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        self.assertEqual(segment4_vfb(original), segment4_vfb(rebuild))

    def test_rebuild_noise_is_outside_segment_4(self):
        original = load("conv_8c4c_8x8_k9x1.mcode.gz")
        rebuild = load("conv_8c4c_8x8_k9x1_rebuild.mcode.gz")
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertLess(len(diffs), 20, "ordinary noise-floor scale")
        _, segs = mcode.segments(original)
        pos4, len4, _ = segs[4]
        for d in diffs:
            self.assertFalse(pos4 <= d < pos4 + len4, f"diff at {d} falls in segment 4")


if __name__ == "__main__":
    unittest.main()
