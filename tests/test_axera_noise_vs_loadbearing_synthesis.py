"""Synthesis: how much of this codec's addressing/allocation is free
choice a future from-scratch generator could make arbitrarily, versus
how much must match a specific, currently-undecoded rule?

This is groundwork for assessing the feasibility of a genuine
from-scratch mcode code generator (as opposed to this project's current
generation capability, which always copies a real Pulsar2-compiled
reference's addressing/allocation verbatim and only substitutes values
into slots that reference already allocated -- see `tiny_emit.py`'s
`patch_output_quad`/`patch_site_a`/`patch_matmul_a_scale`/
`patch_conv_zp_x`, and `emitter.py`'s explicit "does not originate a
shape" boundary). Everything below is synthesis from this session's
already-merged PRs plus one pre-existing large-scale README finding --
no new mcode was built or decoded for this file; every claim is cited
to its source.

## The large-scale baseline: ~83% load-bearing, ~17% inert

`scripts/axera/README.md`'s "The bit-flip probe on the real `resnet18d`
mcode" section (single-byte `^= 0xFF` flips at 41 evenly-spaced offsets
across a real 49,080-byte compiled model, each checked on real AX650N
hardware) found:

- 61% (25/41) FAULT the runtime outright (`0x8030070C`, a structural/
  checksum-style rejection)
- 22% (9/41) run to completion but change the actual output (7 of 9
  change the predicted class)
- 17% (7/41) are genuinely INERT -- bit-identical output regardless of
  the flip

So **by this random sample, only ~17% of a real model's mcode bytes are
truly arbitrary in the strongest sense** (any byte value at all, not
just a small discrete set, produces identical behavior). That is the
headline number for "how much of this format is free," and it is a
much larger, unbiased sample than anything below -- but it answers a
different, narrower question than a generator actually needs answered:
it tests whether an EXISTING byte can be corrupted without consequence,
not whether the COMPILER ITSELF ever produces different-but-equally-
valid byte values for equivalent semantic content across rebuilds. The
second question is what this session's case studies below actually
probe, and it turns out to expose meaningfully more flexibility than
the 17% inert figure alone suggests -- a fair amount of the 83% "load
bearing" majority breaks down further into "must be exactly right" and
"must be one of a small valid set, but any member of that set works,"
and only rebuild-vs-rebuild comparisons (not single-byte corruption)
can tell those two apart.

## This session's case studies: not just inert-or-not, but rebuild-vs-rebuild

Seven cases, spanning three ops, each independently checked by building
the SAME semantic config multiple times and comparing:

| case | source PR(s) | what varies | verdict | evidence |
| --- | --- | --- | --- | --- |
| Gemm M=8, `transB` diff | `tests/test_axera_gemm_m8_diff_is_allocation_noise.py` | which register holds which of an already-fixed *set* of 6 short-unit payloads | noise | 1 pair (tb0 vs tb1): identical payload multiset, different reg assignment |
| MatMul `A_offset`/`B_offset` table order | `tests/test_axera_matmul_offset_table_coinflip.py`, `tests/test_axera_matmul_a_input_cascades.py` | which of 2 peer name-table entries is enumerated first | noise (unconditioned coin flip) | 8 independent rebuilds of ONE unchanged config: clean bimodal split, ~50/50, zero correlation to input |
| Gemm FC-scale "3-way rotation" | `tests/test_axera_gemm_transb_rotation_stability.py` | 3 register/tag slots' value assignment, AND the record parse boundary itself (leading raw-byte count) | noise | 6 fresh rebuilds + 2 existing fixtures: 7 distinct value-tuples across 8 samples, no two identical, record boundary itself shifts |
| Conv dilation-3/4 extra B-run | `tests/test_axera_conv_dilation_b_run_is_reliable.py` | presence/absence of a real record run (not a reassignment -- a genuine content insertion) | load-bearing, reliable | 13 builds across both states (present at d=3, absent at d=2), zero exceptions |
| Gemm `K*M-1` field regime transitions | `tests/test_axera_gemm_km1_regime2_missing_value.py` | ~110-record op-program segment content, diffusely, at every K-regime boundary | load-bearing, diffuse (not a coin flip, not a clean field) | exhaustive same-register search rules out relocation; record-level diff shows 0.78-0.79 match ratio, scattered edits, no clean single-field story |
| Conv k=9 segment-4 content | `tests/test_axera_conv_k9_segment34_content.py` | an 88-verb block (39-verb setup + 4x 11-verb repeat) | constant/boilerplate for this (k, hw) shape | 6 independent cin/cout configs (1-8, both dims): segment 4 byte-identical in all 6, channel-count-invariant |
| Whether the coin-flip mechanism can even occur | `tests/test_axera_table_order_coinflip_audit.py` | structural precondition (2+ peer live non-constant tensor entries in one name-table category) | Gemm/Conv structurally CANNOT have it -- confirmed by direct compiler rejection, not just absence of observation | `Gemm(A,B non-const)` and `Conv(bias non-const)` both fail to compile outright; 19 fixtures (11 Gemm, 8 Conv) show zero table-order variation ever |

## The taxonomy this supports

Four distinct buckets emerge, not two:

1. **Genuinely arbitrary (any value)** -- the README's ~17% inert
   bucket. Not directly probed by this session's own work, but the
   largest and most permissive category when it applies.
2. **Free choice among a small, discrete, semantically-equivalent set**
   -- register/slot reassignment among an already-fixed payload set
   (Gemm M=8), and name-table entry order (MatMul). A generator does
   NOT need Pulsar2's exact policy here -- any valid member of the set
   works, confirmed by Pulsar2 itself not committing to one canonical
   choice across rebuilds. Case 7 above gives this a real predictive
   rule, not just empirical observation: this kind of freedom appears
   specifically where an unordered internal collection has 2+ *peer*
   entries to iterate over (2+ live, non-constant, independently-
   addressed tensors of the same table category). Gemm/Conv structurally
   never have this precondition (Pulsar2 forces weight/bias to be
   compile-time constants), so this category may simply not exist for
   them at all -- it is not merely unobserved there, it is
   structurally excluded.
3. **Load-bearing but with a currently-undecoded generation rule** --
   Gemm's `K*M-1` regime transitions, Conv's dilation-3/4 trigger. Real,
   reproducible, tied to input -- but for the Gemm case specifically,
   NOT a single relocatable field; a genuine multi-record reflow with no
   known formula for what the new records should contain. A generator
   cannot fake this by picking anything valid -- it needs the actual
   rule, which is undecoded.
4. **Load-bearing AND already known to be constant/reusable** -- Conv's
   k=9 setup+repeat block, confirmed byte-identical across a real
   channel-count sweep. This is the best-case outcome for a generator:
   content that must be exactly right, but is a fixed template for a
   given (k, hw) shape family, copyable wholesale rather than needing
   per-instance derivation.

## What this means for a future generator, honestly bounded

This is seven case studies plus one 41-sample random probe across
different ops/shapes -- not a statistical sample of "all addressing
decisions in this codec." What IS supported:

- Bucket 2 (free choice, no policy to reverse-engineer) is real and has
  at least one predictive structural rule (case 7's peer-entry
  precondition), so a generator can identify SOME categories of
  allocation freedom in advance rather than needing exhaustive
  per-field probing -- a genuine, if narrow, win.
- Bucket 3 (load-bearing but diffuse/undecoded) is not rare: it is what
  this session repeatedly found at shape-regime boundaries across all
  three ops studied (Gemm, Conv, and per
  `tests/test_axera_matmul_element_count_search.py`, MatMul too), and
  it is the harder problem bucket 2's good news does not touch. Nothing
  here suggests bucket 3 is small.
- The 83%/17% split is the only large-N estimate available and should
  be read as an upper bound on "how much could plausibly be free" (an
  inert byte is unconditionally free; a bucket-2 byte is free only
  within a specific mechanism's own constraint) -- the true fraction of
  mcode that a generator could safely leave to arbitrary choice is
  therefore somewhere at or below 17%, not somewhere between 17% and
  83%, since bucket 3's diffuse-but-load-bearing content is real
  computation even though corrupting any single byte of it might
  register as merely "DIFFERENT" rather than "FAULT" in the resnet18d
  probe's own categories.

Not established here, and worth flagging for whoever continues: no
case study above tested a genuinely NEW shape's addressing from
nothing -- every case compares rebuilds of an ALREADY-CHOSEN shape.
Whether bucket 2's freedom persists, shrinks, or grows as shapes scale
up (the way bucket 3's diffuse reflows get more complex, not less, as
this session's own Gemm K-regime work found three regimes rather than
one) is untested.
"""

import gzip
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


class TestBucket2FreeChoiceCasesAreReproduced(unittest.TestCase):
    """Sanity-reconfirm, directly against already-committed fixtures,
    the two clearest "free choice among a discrete set" cases this
    synthesis relies on -- not new decoding, just re-verifying the
    source PRs' own claims hold against the fixtures as committed."""

    def test_gemm_m8_same_payload_set_different_registers(self):
        import sys

        _AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
        if _AXERA_DIR not in sys.path:
            sys.path.insert(0, _AXERA_DIR)
        import mcode  # noqa: E402

        d0 = load("gemm_8x8x8_tb0.mcode.gz")
        d1 = load("gemm_8x8x8_tb1.mcode.gz")
        recs0 = mcode.decode(d0, start=671, end=701)
        recs1 = mcode.decode(d1, start=671, end=701)
        payloads0 = sorted(r["payload"] for r in recs0)
        payloads1 = sorted(r["payload"] for r in recs1)
        self.assertEqual(payloads0, payloads1, "same payload set")
        regs0 = [r["reg"] for r in recs0]
        regs1 = [r["reg"] for r in recs1]
        self.assertNotEqual(regs0, regs1, "different register assignment")

    def test_matmul_table_order_coinflip_reconfirmed(self):
        a = load("matmul_4x8x8_rebuild_modeA.mcode.gz")
        b = load("matmul_4x8x8_rebuild_modeB.mcode.gz")
        self.assertEqual(len(a), len(b))

        def table_order(data):
            window = data[195:270]
            ia = window.find(b"A_offset")
            ib = window.find(b"B_offset")
            return ("A", "B") if ia < ib else ("B", "A")

        self.assertNotEqual(table_order(a), table_order(b))
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertGreater(len(diffs), 900)


class TestBucket4ConstantContentIsReproduced(unittest.TestCase):
    """Reconfirm Conv's k=9 setup+repeat block is genuinely
    channel-count-invariant -- the best case for a future generator
    (fixed, copyable template rather than per-instance content)."""

    def test_k9_segment4_length_is_constant_across_channel_counts(self):
        # Per test_axera_conv_k9_segment34_content.py's own table:
        # cin=cout=4, cin=4/cout=8, cin=4/cout=2, cin=8/cout=4, cin=2/cout=4
        # all give identical segment 4 (offset unchanged, still 704
        # bytes) -- only cin=1 shrinks segment 2 (the weight table), not
        # segment 4. Reconfirm directly against the already-committed
        # fixtures for two of those configs.
        import sys

        _AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
        if _AXERA_DIR not in sys.path:
            sys.path.insert(0, _AXERA_DIR)
        import mcode  # noqa: E402

        d1 = load("conv_4c4c_8x8_k9x1.mcode.gz")
        d2 = load("conv_8c4c_8x8_k9x1.mcode.gz")
        _, segs1 = mcode.segments(d1)
        _, segs2 = mcode.segments(d2)
        pos1, len1, _ = segs1[4]
        pos2, len2, _ = segs2[4]
        self.assertEqual(len1, len2, "segment 4 length constant across cin")
        self.assertEqual(
            d1[pos1 : pos1 + len1],
            d2[pos2 : pos2 + len2],
            "segment 4 content byte-identical across cin",
        )


class TestBucket3DiffuseLoadBearingIsNotACoinFlip(unittest.TestCase):
    """Reconfirm Gemm's K*M-1 regime-2 transition is a genuine content
    reflow (bucket 3), not a coin-flip-style free reassignment (bucket
    2) -- the two look superficially similar (both cause hundreds of
    bytes to differ) but only one has a known, bounded, discrete set of
    valid outcomes."""

    def test_no_relocated_same_register_record(self):
        import sys

        _AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
        if _AXERA_DIR not in sys.path:
            sys.path.insert(0, _AXERA_DIR)
        import mcode  # noqa: E402

        def full_decode(data):
            out = []
            _, segs = mcode.segments(data)
            for pos, length, _ in segs:
                end = pos + length
                while end > pos and data[end - 1] == 0:
                    end -= 1
                out += mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)
            return out

        k33 = full_decode(load("gemm_4x33x8_m4k33_r2probe.mcode.gz"))
        reg70 = [r for r in k33 if r.get("kind") == "S" and r.get("reg") == 70]
        # Exactly the one valueless field record, nothing relocated.
        self.assertEqual(len(reg70), 1)
        self.assertEqual(reg70[0]["payload"], b"\xd0\x0c")


if __name__ == "__main__":
    unittest.main()
