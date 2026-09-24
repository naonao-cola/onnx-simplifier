"""The 35-record "setup sequence" this session decoded for Conv's `k=9`
kernel-orientation regime (`tests/test_axera_conv_k9_segment34_content.py`,
PR #1550) and then found recurring in a training-graph fixture
(`tests/test_axera_training_graph_setup_sequence.py`, PR #1552) is not a
two-fixture coincidence: it is a universal per-op-program preamble,
confirmed present in every committed training/backward-graph fixture
this project has, with one precisely-characterized variant.

## Method

`mcode.decode()` on each fixture's `segments()`, restricted to `V`-kind
(verb) records, compared as `(verb, field, bank)` triples against the
35-record `SHARED_SETUP_BLOCK` (copied verbatim from PR #1552, not
re-derived) -- a subsequence search, not a raw-byte search, the
technique this project's own record-level work has repeatedly found
necessary once byte offsets stop being comparable across differently-
shaped streams.

## Result: present in every fixture checked, exact in three, +1-record
## variant in the two full multi-op training steps

| fixture | segment | exact 35-match | notes |
| --- | --- | --- | --- |
| `loss_head_kd.mcode.gz` | 4 | record 3 | exact |
| `resnet18_int8.mcode.gz` | 4 | record 4 | exact |
| `reshape_gather_bwd.mcode.gz` | 4 | record 3 | exact |
| `toy_training_step.mcode.gz` | 4 | -- | 36-record variant (below) |
| `w2v2fe_training_step.mcode.gz` | 4 | -- | 36-record variant (below) |

Combined with PR #1552's own `adam_update_fp32.mcode.gz` result, this
is **6 of 6** committed training/backward-graph fixtures -- every one
this project has -- carrying the block in segment 4 specifically, in
either the exact or the variant form. This is a much stronger claim
than PR #1552's own "not Conv-specific" framing supported: it is not
just cross-op-family, it is present in the *entire* training-graph
fixture corpus.

## The two full multi-op training steps carry a +1-record variant, not
## a divergence

`loss_head_kd`, `resnet18_int8`, and `reshape_gather_bwd` are each a
single op (or a short op-and-consumer pair). `toy_training_step` (24 op
programs, forward+KD-loss+backward+Adam chained in one compiled model,
per the README's "Training-graph streams" section) and
`w2v2fe_training_step` (a larger real training step) are both full,
multi-op-program training steps -- and in both, the setup block is not
simply absent or corrupted: it is the *identical* 35-record sequence
with exactly **one extra record, `(167, 0, 30)`, inserted between the
block's 22nd and 23rd elements** (i.e. right after the `(161, 48, 3)`
record that starts the block's third bank-run and before the `(168,
64, 3)` record that follows it). Confirmed record-for-record: the
first 22 records match `SHARED_SETUP_BLOCK[0:22]` exactly, the
insertion is `(167, 0, 30)` in both fixtures, and the remaining 13
records match `SHARED_SETUP_BLOCK[22:35]` exactly, shifted by one.

Verb `167` does not otherwise appear anywhere in `SHARED_SETUP_BLOCK`
itself. A plausible (not proven) reading, offered as a hypothesis: this
extra record marks something specific to being one of several chained
op-programs in a single compiled stream -- a sequencing or "next
program" marker -- present only in the two fixtures that are actually
multi-op training *steps* rather than single-op probes. Not decoded to
that meaning here; only the exact insertion point and content are
established.

## What remains open

This file locates the block precisely in every fixture and
characterizes its one real variant, but does not decode what the block
itself computes (still open per PR #1550/#1552), nor why the
insertion happens specifically at this position, nor whether it
correlates with op count, op type, or something else about multi-program
streams (only 2 data points, both multi-op streams, both showing the
identical insertion -- not enough to rule out "always exactly this" vs.
"varies with something not yet tested").
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


# Copied verbatim from tests/test_axera_training_graph_setup_sequence.py
# (PR #1552) -- not re-derived here.
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

INSERTED_RECORD = (167, 0, 30)
VARIANT_BLOCK = SHARED_SETUP_BLOCK[:22] + [INSERTED_RECORD] + SHARED_SETUP_BLOCK[22:]


def verb_triples(data, seg_index):
    _, segs = mcode.segments(data)
    pos, length, _ = segs[seg_index]
    recs = mcode.decode(data, start=pos, end=pos + length, **mcode.FULL_RULE)
    return [(r["verb"], r["field"], r["bank"]) for r in recs if r["kind"] == "V"]


def find_subsequence(haystack, needle):
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if haystack[i : i + n] == needle]


class TestExactBlockInSingleOpTrainingFixtures(unittest.TestCase):
    """loss_head_kd, resnet18_int8, and reshape_gather_bwd -- all
    single-op (or op-plus-consumer) fixtures -- carry the exact,
    unmodified 35-record block in segment 4."""

    # Indices count V records from the segment start. loss_head_kd and
    # reshape_gather_bwd have 4-byte tail padding, so the corrected
    # mcode.segments (docs/axera-mcode-segments-fix.md) starts their segment 4
    # four bytes earlier and decodes its opening a7 record too: [3] -> [4].
    CASES = {
        "loss_head_kd.mcode.gz": [4],
        "resnet18_int8.mcode.gz": [4],
        "reshape_gather_bwd.mcode.gz": [4],
    }

    def test_exact_block_present_at_expected_offset(self):
        for name, expected_hits in self.CASES.items():
            data = load(name)
            triples = verb_triples(data, 4)
            hits = find_subsequence(triples, SHARED_SETUP_BLOCK)
            self.assertEqual(hits, expected_hits, name)


class TestVariantBlockInFullMultiOpTrainingSteps(unittest.TestCase):
    """toy_training_step and w2v2fe_training_step -- both full,
    multi-op-program training steps -- carry a 36-record variant: the
    identical 35-record block with one extra record, (167, 0, 30),
    inserted between elements 22 and 23."""

    # w2v2fe has 4-byte tail padding: [4] -> [5] for the same reason as above.
    CASES = {
        "toy_training_step.mcode.gz": [5],
        "w2v2fe_training_step.mcode.gz": [5],
    }

    def test_variant_block_present_at_expected_offset(self):
        for name, expected_hits in self.CASES.items():
            data = load(name)
            triples = verb_triples(data, 4)
            hits = find_subsequence(triples, VARIANT_BLOCK)
            self.assertEqual(hits, expected_hits, name)

    def test_exact_unmodified_block_is_absent(self):
        """The plain 35-record block (without the insertion) does NOT
        appear anywhere in these two fixtures -- the insertion is not
        an optional extra copy alongside the plain form, it replaces it."""
        for name in self.CASES:
            data = load(name)
            triples = verb_triples(data, 4)
            self.assertEqual(find_subsequence(triples, SHARED_SETUP_BLOCK), [], name)

    def test_inserted_record_verb_does_not_appear_in_the_plain_block(self):
        self.assertNotIn(167, {v for v, _, _ in SHARED_SETUP_BLOCK})


class TestAllSixTrainingFixturesAccountedFor(unittest.TestCase):
    """The full committed training/backward-graph fixture corpus (6
    files, including adam_update_fp32 from PR #1552) all carry this
    block in segment 4, in one of exactly two forms."""

    ALL_SIX = [
        "adam_update_fp32.mcode.gz",
        "loss_head_kd.mcode.gz",
        "resnet18_int8.mcode.gz",
        "reshape_gather_bwd.mcode.gz",
        "toy_training_step.mcode.gz",
        "w2v2fe_training_step.mcode.gz",
    ]

    def test_every_fixture_has_either_the_exact_or_variant_block(self):
        for name in self.ALL_SIX:
            data = load(name)
            triples = verb_triples(data, 4)
            exact = find_subsequence(triples, SHARED_SETUP_BLOCK)
            variant = find_subsequence(triples, VARIANT_BLOCK)
            self.assertTrue(exact or variant, f"{name}: neither form found")


if __name__ == "__main__":
    unittest.main()
