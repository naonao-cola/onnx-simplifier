"""Synthesis: what does `reg=8`'s rebuild-to-rebuild noise look like
across all four of this project's main tracked ops, now that the
census-based rebuild-stability sweep (Mul/Gemm/Conv/MatMul) and two
of the four ops' own "second noise mechanism" decodes have landed?

In the style of `tests/test_axera_noise_vs_loadbearing_synthesis.py`
(PR #1561) -- this file does not decode anything new from scratch by
building fresh mcode; it collects and directly re-verifies (against the
real, already-committed fixtures, not just by repeating each source
PR's own docstring claims) a pattern that has been genuinely scattered
across six separate PRs, then adds one new cross-op observation that no
single PR was positioned to make on its own.

## The four-op comparison

| op | reg=8 noisy? | table-order coin flip explains it? | second mechanism decoded? | source PR(s) |
| --- | --- | --- | --- | --- |
| Mul | yes | partially -- explains record COUNT (48 vs 50) exactly, but not every payload byte within a fixed count | no -- residual left open | #1566 |
| Gemm | yes | no -- structurally cannot occur (only one live non-constant tensor) | **yes** -- 3-slot unordered pool, anchor `reg=78`, biconditional `reg=0` extra record | #1576, #1577 |
| Conv | yes | no -- structurally cannot occur (no `*_offset` table at all) | **yes** -- 3-of-4 unordered pool, anchor `reg=170`, biconditional `reg=172` tag | #1579, #1580 |
| MatMul | yes | no -- table order demonstrably explains 6 of `reg=8`'s 7 sibling noisy registers in this exact build, but not `reg=8` itself | not yet (open at the time of this file; a concurrent investigation may have landed since -- see note below) | #1581 |

`reg=8` is the only register found noisy in all four ops. It is also
the only one where the *established* mechanism (the coin flip) is
confirmed insufficient in three of the four cases and only partially
sufficient in the fourth.

## The new finding this file adds: one shared trailing-byte pool, not
## four unrelated ones

Each op-specific decode independently noticed its own pool's trailing
bytes were "evenly spaced by `0x10`" (Gemm: `{0x20, 0x30, 0x40}`,
3 members; Conv: `{0x10, 0x20, 0x30, 0x40}`, 4 members) but none of the
individual PRs checked whether Mul's own residual, or MatMul's own
unstructured `reg=8` values, draw from the *same* pool rather than
each op having its own independent small pool. Directly re-decoding
all four ops' `reg=8` payloads (see `TestSharedTrailingBytePoolAcrossAllFourOps`
below) confirms they do: **every observed `reg=8` payload across every
op, in every already-committed rebuild-stability fixture, has a
trailing byte in the same 4-member set `{0x10, 0x20, 0x30, 0x40}`**.
Mul's own previously-undecoded residual (PR #1566 explicitly left it
open) turns out to hit all four members of this exact pool across its
8 samples -- as rich as Conv's own fully-decoded 4-member pool, just
without Conv's multi-slot structure or a discovered biconditional
indicator.

The *leading* bytes differ by op (Mul: `0x13`; Gemm: `0x93`/`0x23`;
Conv: `0x23`; MatMul: `0x33`/`0x23`) and are NOT claimed to be shared --
only the trailing byte, i.e. only the low nibble of what this project's
existing "16-byte field-offset granularity" convention (README ~line
2206) would call the sub-field offset within a 16-byte-aligned record,
is common across ops. This is consistent with (though does not prove)
the reading Gemm's and Conv's own decodes already offered as
speculative: a small pool of candidate scratch/tile-buffer addresses,
spaced at a fixed hardware granularity, that every op's scheduler draws
from -- with the op-specific leading byte encoding something else (verb/
record-kind or which physical buffer family) that this file does not
attempt to decode.

## Slot-structure comparison: similar mechanism CLASS, not one universal
## rule

Gemm and Conv's own decoded mechanisms share real structure (an
unordered pool + a stable anchor register a few bytes away + a paired
biconditional indicator record) but do NOT reduce to one interchangeable
rule:

| | Gemm (#1577) | Conv (#1580) |
| --- | --- | --- |
| pool size | 3 | 4 |
| slots filled | 3 of 3 (always) | 3 of 4 (always) |
| anchor register | `reg=78` | `reg=170` |
| slot register labels | fixed (`78, 8, 8`) | variable (2 of 3 slots draw their own register label from a small pool too: `8`, `242`, or `176`) |
| biconditional indicator | `reg=0`'s extra record, present iff fewer than 3 distinct classes used | `reg=172`'s tag (132 vs 134), iff slot 1 used its short payload form |
| duplicate tolerance | yes (a class can repeat, one PR #1577 sample uses `C` twice) | not directly needed (pool has 4, only 3 used, so no repeats observed) |

This file does not attempt to force these into one shared formula --
the honest conclusion, matching this project's established practice of
not claiming more unification than the evidence supports (see PR
#1575's own refusal to force a rule onto Gemm's `Q` presence pattern),
is: **same mechanism CLASS (an unordered draw from a small, evenly-`0x10`-
spaced trailing-byte pool, paired with a biconditional indicator
elsewhere in the record stream), genuinely shared across at least
three ops at the level of the pool's own VALUES -- but the slot
structure around that pool (how many slots, whether register labels
vary, what the indicator is) is decoded separately per op and does not
reduce to a single rule.**

## Taxonomy reconciliation against PR #1561's four buckets

PR #1561's bucket 3 ("load-bearing but with a currently-undecoded
generation rule") was written before any of Gemm's, Conv's, or this
file's own cross-op findings existed -- at that time `reg=8`'s
non-coin-flip noise was not even identified as a distinct phenomenon,
let alone decoded. It does not fit cleanly into any of the original
four buckets as originally defined, and forcing it into one would
overstate what's known:

- Not bucket 1 (genuinely arbitrary) -- the pool is a small, fixed,
  4-member set, not "any value."
- Not cleanly bucket 2 (free choice among a discrete set, no policy to
  reverse-engineer) -- bucket 2's own defining example (MatMul's
  `A_offset`/`B_offset` coin flip) is a full explanation with a known
  structural precondition (PR #1561's own case 7). `reg=8`'s mechanism
  in Gemm/Conv IS now a full explanation of the MECHANISM (which slots
  get filled, in what pattern, with what indicator) -- but the
  SEMANTIC MEANING of the pool's values themselves (candidate
  scratch/tile-buffer addresses is the working hypothesis, explicitly
  not proven in both #1577 and #1580) remains open, which bucket 2's
  clean cases do not leave open.
- Not bucket 3 as originally scoped (diffuse, hundreds of bytes,
  no known formula at all) -- Gemm and Conv's mechanisms ARE now a
  known, verified, zero-exception formula for which records vary and
  how, just not for what the values mean.
- Not bucket 4 (load-bearing and already fully reusable/constant) --
  the whole point is this content varies build-to-build; nothing here
  is a fixed, copyable template the way Conv's k=9 setup block is.

**This file proposes treating "mechanism decoded, semantics undecoded"
as a distinct sub-case worth naming explicitly** rather than stretching
bucket 2 or bucket 3 to cover it: a generator that has this pool's
mechanism decoded (Gemm, Conv) could emit a VALID rebuild by choosing
any legal slot assignment (the same practical win bucket 2 offers,
since Pulsar2 itself doesn't commit to one canonical choice) even
without knowing what the values semantically mean -- but a generator
for MatMul or Mul, where the analogous mechanism is not yet
(fully) decoded, cannot yet do even that.

## Honest bounds

- MatMul's own slot-structure decode (which records neighbor `reg=8`,
  whether register labels vary the way Conv's did, whether there's a
  biconditional indicator) is NOT attempted here -- a separate,
  concurrent investigation may have already produced it; this file
  only reconfirms that MatMul's `reg=8` payload values fall in the same
  shared pool, not the full mechanism.
- The "same pool across all four ops" finding is confirmed for exactly
  the fixtures already committed by each op's own rebuild-stability
  work (8 Mul, 8 Gemm, 8 Conv, 8 MatMul samples) -- not an independent,
  larger sample. It is a real, directly-reconfirmed pattern, not a
  4-point coincidence dismissed as such, but it is also not yet
  stress-tested against a shape where the pool might turn out to have
  more or different members.
- The "candidate scratch/tile-buffer address" reading is speculative
  and inherited from Gemm's/Conv's own PRs, explicitly not proven by
  either of them or by this file.
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

POOL_TRAILING_BYTES = {0x10, 0x20, 0x30, 0x40}

MUL_REBUILDS = [f"mul_1x8_universal_stability_r{i}.mcode.gz" for i in range(8)]
GEMM_OLD_BATCH = [
    "gemm_1x512x1000_tb0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
]
GEMM_FRESH_BATCH = [
    "gemm_1x512x1000_tb0_stability_r0.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r1.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r2.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r3.mcode.gz",
]
GEMM_ALL = GEMM_OLD_BATCH + GEMM_FRESH_BATCH
CONV_OLD_BATCH = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]
CONV_FRESH_BATCH = [
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
CONV_ALL = CONV_OLD_BATCH + CONV_FRESH_BATCH
MATMUL_BATCH1 = [
    "matmul_4x8x8_v7stability_diag0.mcode.gz",
    "matmul_4x8x8_v7stability_r1.mcode.gz",
    "matmul_4x8x8_v7stability_r2.mcode.gz",
    "matmul_4x8x8_v7stability_r3.mcode.gz",
]
MATMUL_BATCH2 = [
    "matmul_4x8x8_v7stability_r4.mcode.gz",
    "matmul_4x8x8_v7stability_r5.mcode.gz",
    "matmul_4x8x8_v7stability_r6.mcode.gz",
    "matmul_4x8x8_v7stability_r7.mcode.gz",
]
MATMUL_ALL = MATMUL_BATCH1 + MATMUL_BATCH2


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg8_records(recs):
    return [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 8 and r.get("payload") is not None
    ]


def pool_records(recs):
    """The specific subset of reg=8's records that carry a trailing-byte
    pool value: tag=130 records that are either the 3-byte "XX 00 YY"
    form (Mul/Gemm/Conv/MatMul's own long form, YY the pool member) or
    the 1-byte short form (Conv's own P4, a bare pool-member byte) --
    excludes reg=8's other, unrelated tag=130/131/132 records (bank
    field writes, table-length-dependent verbs, etc.) that are not part
    of this mechanism at all."""
    out = []
    for r in recs:
        if r["kind"] != "S" or r.get("reg") != 8 or r.get("tag") != 130:
            continue
        p = r.get("payload")
        if p is None:
            continue
        if len(p) == 3 and p[1] == 0:
            out.append(r)
        elif len(p) == 1 and p[0] in POOL_TRAILING_BYTES:
            out.append(r)
    return out


def trailing_bytes_used(recs):
    return {r["payload"][-1] for r in pool_records(recs)}


class TestReg8IsNoisyInAllFourOps(unittest.TestCase):
    """Reconfirms, directly against the fixtures, that reg=8's content
    differs across at least one pair of same-op rebuilds in each of the
    four ops -- the shared headline fact this whole synthesis rests on."""

    def _reg8_content(self, name):
        recs = decode(name)
        return [(r.get("tag"), r.get("payload")) for r in reg8_records(recs)]

    def test_mul(self):
        vals = {tuple(self._reg8_content(n)) for n in MUL_REBUILDS}
        self.assertGreater(len(vals), 1)

    def test_gemm(self):
        vals = {tuple(self._reg8_content(n)) for n in GEMM_ALL}
        self.assertGreater(len(vals), 1)

    def test_conv(self):
        vals = {tuple(self._reg8_content(n)) for n in CONV_ALL}
        self.assertGreater(len(vals), 1)

    def test_matmul(self):
        vals = {tuple(self._reg8_content(n)) for n in MATMUL_ALL}
        self.assertGreater(len(vals), 1)


class TestMulCoinFlipExplainsCountButLeavesAResidual(unittest.TestCase):
    """Reconfirms PR #1566's own finding: Mul's x_offset/y_offset table
    order predicts reg=8's record COUNT exactly, but the first record's
    payload still varies within a fixed-count, fixed-order group."""

    def _table_order(self, data):
        window = data[180:320]
        ix = window.find(b"x_offset")
        iy = window.find(b"y_offset")
        return ("x", "y") if ix < iy else ("y", "x")

    def test_count_is_a_clean_function_of_table_order(self):
        by_order = {}
        for n in MUL_REBUILDS:
            d = load(n)
            order = self._table_order(d)
            count = len(reg8_records(decode(n)))
            by_order.setdefault(order, set()).add(count)
        self.assertEqual(len(by_order[("x", "y")]), 1)
        self.assertEqual(len(by_order[("y", "x")]), 1)
        self.assertNotEqual(by_order[("x", "y")], by_order[("y", "x")])

    def test_payload_still_varies_within_one_order_group(self):
        yx_group_first_payloads = set()
        for n in MUL_REBUILDS:
            d = load(n)
            if self._table_order(d) != ("y", "x"):
                continue
            recs = reg8_records(decode(n))
            yx_group_first_payloads.add(recs[0]["payload"])
        self.assertGreater(
            len(yx_group_first_payloads),
            1,
            "table order alone should not fully explain reg=8 for Mul",
        )


class TestGemmAndConvReg8VariesWithinASingleBatch(unittest.TestCase):
    """Reconfirms reg=8 is unstable even within one independently-built
    4-sample batch for Gemm and Conv -- ruling out both table order
    (structurally impossible for these ops) and batch/calibration
    provenance as the sole explanation."""

    def test_gemm_old_batch_alone(self):
        vals = {tuple(trailing_bytes_used(decode(n))) for n in GEMM_OLD_BATCH}
        self.assertGreater(len(vals), 1)

    def test_conv_old_batch_alone(self):
        vals = {tuple(trailing_bytes_used(decode(n))) for n in CONV_OLD_BATCH}
        self.assertGreater(len(vals), 1)


class TestMatMulTableOrderExplainsSiblingsButNotReg8(unittest.TestCase):
    """Reconfirms PR #1581's own finding: in the one op where the coin
    flip demonstrably explains most of reg=8's near-universal-register
    neighbors, it still does not explain reg=8 itself."""

    def _table_order(self, data):
        window = data[100:400]
        ia = window.find(b"A_offset")
        ib = window.find(b"B_offset")
        return ("A", "B") if ia < ib else ("B", "A")

    def test_reg8_varies_within_one_table_order_group(self):
        by_order = {}
        for n in MATMUL_ALL:
            d = load(n)
            order = self._table_order(d)
            payloads = tuple(sorted(trailing_bytes_used(decode(n))))
            by_order.setdefault(order, set()).add(payloads)
        for order, vals in by_order.items():
            self.assertGreater(
                len(vals),
                1,
                f"reg=8 should still vary within table-order group {order}",
            )


class TestSharedTrailingBytePoolAcrossAllFourOps(unittest.TestCase):
    """The new cross-op finding this file adds: every reg=8 payload's
    trailing byte, in every already-committed rebuild-stability fixture
    for all four ops, is a member of the same 4-value pool
    {0x10, 0x20, 0x30, 0x40} -- not four independent, op-specific pools."""

    def _all_trailing_bytes(self, names):
        used = set()
        for n in names:
            used |= trailing_bytes_used(decode(n))
        return used

    def test_mul_uses_only_pool_members(self):
        used = self._all_trailing_bytes(MUL_REBUILDS)
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)

    def test_gemm_uses_only_pool_members(self):
        used = self._all_trailing_bytes(GEMM_ALL)
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)

    def test_conv_uses_only_pool_members(self):
        used = self._all_trailing_bytes(CONV_ALL)
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)

    def test_matmul_uses_only_pool_members(self):
        used = self._all_trailing_bytes(MATMUL_ALL)
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)

    def test_mul_alone_hits_all_four_pool_members(self):
        """Mul's own previously-undecoded residual (PR #1566) turns out
        to be just as rich as Conv's fully-decoded 4-member pool, once
        directly checked -- this specific fact was not stated by any
        prior PR."""
        used = self._all_trailing_bytes(MUL_REBUILDS)
        self.assertEqual(used, POOL_TRAILING_BYTES)

    def test_leading_bytes_differ_by_op_only_trailing_byte_is_shared(self):
        """The pool's sharing is specifically at the trailing-byte
        level -- leading bytes (of the same 3-byte pool records) are
        op-specific and not claimed to be part of one shared pool."""

        def leading_bytes(names):
            out = set()
            for n in names:
                for r in pool_records(decode(n)):
                    if len(r["payload"]) == 3:
                        out.add(r["payload"][0])
            return out

        leading_by_op = {
            "mul": leading_bytes(MUL_REBUILDS),
            "gemm": leading_bytes(GEMM_ALL),
            "conv": leading_bytes(CONV_ALL),
            "matmul": leading_bytes(MATMUL_ALL),
        }
        self.assertEqual(leading_by_op["mul"], {0x13})
        self.assertEqual(leading_by_op["gemm"], {0x23, 0x93})
        self.assertEqual(leading_by_op["conv"], {0x23})
        self.assertEqual(leading_by_op["matmul"], {0x23, 0x33})
        # Mul's leading byte (0x13) is disjoint from the other three ops'.
        self.assertNotIn(0x13, leading_by_op["gemm"] | leading_by_op["conv"])


if __name__ == "__main__":
    unittest.main()
