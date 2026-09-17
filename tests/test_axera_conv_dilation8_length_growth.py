"""Continues `tests/test_axera_conv_binary_cluster_higher_dilation.py`
(PR #1601)'s own flagged-but-not-chased wrinkle: `dilation=8`'s mcode is
232 bytes longer than `dilation=1/2/3/4`'s (all 3,528 bytes) despite an
identical 16x16 output shape (`pad == dilation` keeps the receptive-field
math constant). PR #1601 noted this "might be plausibly related to this
project's own established dilation-3/4 extra-tap-run material" without
confirming the connection.

## Important correction to PR #1601's framing: the length threshold and
## the absence of the known "extra B-run" spike at high dilation were
## ALREADY documented, just never cross-referenced

`tests/test_axera_conv_dilation_trigger_extent.py` (an earlier PR this
session, predating the binary-cluster investigation) already built
`dilation=8..14` and found: (a) `d=8,9,10,11,13,14` all serialize to
3,760 bytes (`d=12` alone to 3,792), and (b) the dilation=3/4-specific
"extra B-run" spike (`tests/test_axera_conv_dilation_b_run_is_reliable.py`,
`early_b_runs()`) is **absent** at every one of `d=8` through `d=14` --
refuting the "same B-run mechanism, just more of it" hypothesis for
those dilation values directly. Neither that file's own single
`conv_dilation8.mcode.gz`/`conv_dilation12.mcode.gz`/`conv_dilation14.mcode.gz`
fixtures nor PR #1601's later `conv_dilation8_r{0..7}.mcode.gz` batch
were ever cross-referenced against each other, so PR #1601's own framing
("a genuine surprise... not chased") slightly overstated how open this
question was -- the *length threshold* and the *B-run-absence* were both
already established. What neither prior file decoded is **what actually
occupies the extra bytes** -- that is this file's own contribution.

Re-confirmed directly below (`TestEarlySpikeAbsentAtD8`) against all 8
of PR #1601's own fresh rebuilds plus the original pre-existing fixture
(9 samples, 0 exceptions) -- not just repeated from the earlier file's
own prose, the same "cite and reconfirm" discipline this project's
synthesis work has established.

## What fills the extra bytes: a new, content-stable 38-record block,
## including the exact same bank=0xE1 field=32/48 records this session's
## own Gemm sparse-tier work (PR #1570) already found and left partly
## unexplained

Diffing `dilation=4` and `dilation=8`'s decoded record streams (using
`mcode.decode()`, not raw offsets, since the streams reflow) finds a
**38-record structured block, entirely absent from every `dilation=1/2/3/4`
fixture, but present and byte-for-byte identical across all 9 available
`dilation=8` samples** (8 independent `pulsar2:7.0-lite` rebuilds from
PR #1601, plus the original `conv_dilation8.mcode.gz` from an earlier,
separate build session/toolchain -- confirmed via `git log` predates this
session's binary-cluster work entirely). The block spans roughly the
byte range PR #1601's own length delta implies and accounts for most of
the 232-byte growth (the remainder is ordinary small reflow of
surrounding content, not investigated further here).

**The most notable two records in this block**: a `bank=0xE1` (225),
`field=32` V-record with operand `33 03 00`, and a `bank=0xE1`,
`field=48` V-record with operand `35 03 00 a1` -- **byte-identical** to
the two "unexplained" `0xE1` field=32/48 records
`tests/test_axera_gemm_bank_81_e1_decode.py` (PR #1570) found at large-N
Gemm shapes and explicitly left uncharacterized ("two additional
`0xe1` records at large `N`... never characterized further"). This is
the first time this exact constant pair has been seen outside Gemm.
Conv's `dilation=1/2/3/4` fixtures carry **zero** `bank=0xE1` records of
any kind (confirmed directly, not assumed) -- this pair appears only
once Conv's own mcode crosses into the `dilation=8`-and-longer length
class, the same way it appears in Gemm only once `N` crosses its own
33-shape threshold. Both `0x81` and `0xE1`'s *other* already-decoded
Gemm fields (`0x81` field=192's `1024//K-1`, `0xE1` field=112's `35 81 1a`
constant) are absent here -- only this one specific pair generalizes to
Conv, not the whole bank.

This does not decode why either op's compiler reaches for this specific
constant pair at its own "large" threshold, but it is a real, verified,
cross-op structural connection between two previously-separate findings
(Gemm's own large-N sparse tier and Conv's own large-dilation length
growth) that neither op's own investigation was positioned to notice in
isolation -- exactly the kind of link this project's periodic synthesis
work has looked for elsewhere (see `tests/test_axera_reg8_cross_op_synthesis.py`,
PR #1582).

## What remains open

- The block's remaining ~36 records (registers 7-30, various tags) are
  pinned exactly (see `EXPECTED_BLOCK` below) but not semantically
  decoded -- this file establishes WHERE the growth lives and confirms
  it is stable, real content, not noise, but does not explain what any
  individual record computes.
- Whether the block (or the `0xE1` field=32/48 pair specifically) first
  appears exactly at `dilation=8`, or already at some intermediate value
  between `5` and `8` (this project's own `dilation=5` sweep predates
  this file and used a different, incompatible 3,560-byte length class
  per `scripts/axera/README.md`'s own dilation-sweep section -- not
  re-tested here) is not pinned down.
- Whether the same `0xE1` field=32/48 pair also appears in MatMul at its
  own analogous "large" threshold is not checked here.
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

D4_NAMES = [f"conv_dilation4_r{i}.mcode.gz" for i in range(8)] + [
    "conv_dilation4.mcode.gz"
]
D8_NAMES = [f"conv_dilation8_r{i}.mcode.gz" for i in range(8)] + [
    "conv_dilation8.mcode.gz"
]
LOW_DILATION_NAMES = (
    [f"conv_dilation1_r{i}.mcode.gz" for i in range(8)]
    + [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)]
    + ["conv_dilation3.mcode.gz"]
    + D4_NAMES
)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    data = load(name)
    return mcode.decode(data, start=0, end=len(data), **mcode.FULL_RULE)


def early_b_runs(data, min_len=3, cutoff=1600):
    """Same detection method as tests/test_axera_conv_dilation_b_run_is_reliable.py
    (PR #1508) / tests/test_axera_conv_dilation_trigger_extent.py."""
    recs = mcode.decode(data, start=0, end=len(data), **mcode.FULL_RULE)
    runs = []
    i = 0
    while i < len(recs):
        if recs[i].get("kind") == "B":
            j = i
            while j < len(recs) and recs[j].get("kind") == "B":
                j += 1
            if j - i >= min_len and recs[i]["at"] < cutoff:
                runs.append((recs[i]["at"], j - i))
            i = j
        else:
            i += 1
    return runs


def sig(r):
    return (
        r["kind"],
        r.get("reg"),
        r.get("tag"),
        r.get("bank"),
        r.get("field"),
        r.get("verb"),
        r.get("payload"),
        r.get("operand"),
    )


def new_block(name):
    """The 38-record structured block found only at dilation=8, located
    by its own byte range (2651-2879 in the r0 sample; other samples'
    absolute offsets may differ slightly due to ordinary small reflow
    elsewhere in the stream, so this searches a slightly wider window
    and filters to the exact expected length/anchor)."""
    recs = decode(name)
    window = [r for r in recs if r.get("at") is not None and 2600 <= r["at"] <= 2950]
    # Anchor on the bank=0xE1 field=32 record (the most distinctive,
    # rarest signature in this window) and take the 30 structured
    # records immediately before it plus the 7 immediately after,
    # matching the exact 38-record span identified by direct diffing.
    e1_32 = [
        i
        for i, r in enumerate(window)
        if r["kind"] == "V" and r.get("bank") == 225 and r.get("field") == 32
    ]
    assert len(e1_32) == 1, (name, e1_32)
    idx = e1_32[0]
    structured = [r for r in window if r["kind"] != "raw"]
    pos = None
    seen = 0
    for i, r in enumerate(window):
        if r["kind"] != "raw":
            if i == idx:
                pos = seen
                break
            seen += 1
    assert pos is not None, name
    return tuple(sig(r) for r in structured[pos - 31 : pos + 7])


EXPECTED_BLOCK = (
    ("B", 176, 151, None, None, None, None, None),
    ("V", None, None, 0, 0, 162, None, b'"\x00 \x00'),
    ("A", None, None, None, None, None, None, None),
    ("V", None, None, 10, 0, 167, None, b"\x00\x00\x00\x00"),
    ("V", None, None, 4, 208, 161, None, b"\x04\x00\x00\x00"),
    ("V", None, None, 1, 80, 161, None, b"\x00\x00\x00\x10"),
    ("V", None, None, 1, 96, 161, None, b"\xff\t\x00\x00"),
    ("V", None, None, 1, 112, 168, None, b"\x05\x82\x08\x01"),
    ("S", 8, 129, None, None, None, b"\xa1\x00\x90", None),
    ("S", 10, 129, None, None, None, b"\x00\xa1\x00\xa0", None),
    ("S", 12, 129, None, None, None, b"\x00\xa1\x00\xb0", None),
    ("S", 14, 129, None, None, None, b"\x00\xa1\x00\xc0", None),
    ("B", 18, 129, None, None, None, None, None),
    ("B", 8, 132, None, None, None, None, None),
    ("S", 8, 132, None, None, None, b"\xe0", None),
    ("S", 8, 132, None, None, None, b"\xf0", None),
    ("S", 8, 131, None, None, None, b"\x00\x02", None),
    ("S", 14, 130, None, None, None, b"\x10\x02\x00\x1e", None),
    ("S", 16, 132, None, None, None, b"\x06", None),
    ("S", 18, 130, None, None, None, b"\x04\x00\x01", None),
    ("S", 10, 161, None, None, None, b"\x04", None),
    ("S", 30, 131, None, None, None, b"\xa1\x00\x90\x03", None),
    ("S", 8, 132, None, None, None, b"\xa0\x03", None),
    ("S", 30, 131, None, None, None, b"\x03", None),
    ("S", 24, 132, None, None, None, b"\xc0\x03", None),
    ("S", 24, 132, None, None, None, b"\x03", None),
    ("B", 8, 132, None, None, None, None, None),
    ("S", 10, 132, None, None, None, b"\xf0", None),
    ("B", 47, 225, None, None, None, None, None),
    ("B", 24, 129, None, None, None, None, None),
    ("B", 49, 225, None, None, None, None, None),
    ("V", None, None, 225, 32, 161, None, b"3\x03\x00"),
    ("V", None, None, 225, 48, 161, None, b"5\x03\x00\xa1"),
    ("S", 8, 132, None, None, None, b"@", None),
    ("S", 20, 129, None, None, None, b"\xb0\x04\x00\x00\x10", None),
    ("S", 7, 161, None, None, None, b"\x01\x01\x10\x00\xa3", None),
    ("B", 34, 133, None, None, None, None, None),
    ("S", 9, 161, None, None, None, b"\x00\x00\x01\xa9", None),
)


class TestAllFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        for name in D4_NAMES + D8_NAMES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestLengthThresholdMatchesPriorWork(unittest.TestCase):
    """dilation=1/2/3/4 are all 3,528 bytes; dilation=8 is 3,760 --
    232 bytes longer. Matches tests/test_axera_conv_dilation_trigger_extent.py's
    own already-established finding (d=8..14 -> 3,760 bytes, except
    d=12 -> 3,792), re-confirmed here against PR #1601's own fresh
    rebuild batch, not just the single earlier fixture."""

    def test_d4_is_3528_everywhere(self):
        for name in D4_NAMES:
            self.assertEqual(len(load(name)), 3528, name)

    def test_d8_is_3760_everywhere(self):
        for name in D8_NAMES:
            self.assertEqual(len(load(name)), 3760, name)


class TestEarlySpikeAbsentAtD8(unittest.TestCase):
    """The dilation=3/4-specific "extra B-run" spike
    (tests/test_axera_conv_dilation_b_run_is_reliable.py) is present at
    every dilation=4 sample and absent at every dilation=8 sample --
    reconfirms tests/test_axera_conv_dilation_trigger_extent.py's own
    finding directly against PR #1601's fresh 8-sample batch, not just
    the single earlier fixture that file used."""

    def test_d4_shows_the_known_spike_everywhere(self):
        for name in D4_NAMES:
            self.assertEqual(early_b_runs(load(name)), [(552, 8), (1160, 8)], name)

    def test_d8_shows_no_early_spike_anywhere(self):
        for name in D8_NAMES:
            self.assertEqual(early_b_runs(load(name)), [], name)


class TestNewThirtyEightRecordBlockIsStableAcrossAllD8Samples(unittest.TestCase):
    """The block accounting for most of dilation=8's length growth is
    byte-for-byte identical across all 8 independent PR #1601 rebuilds
    plus the original, separately-built conv_dilation8.mcode.gz fixture
    -- 9 samples, 0 exceptions."""

    def test_block_matches_expected_everywhere(self):
        for name in D8_NAMES:
            self.assertEqual(new_block(name), EXPECTED_BLOCK, name)


class TestBlockIsAbsentFromEveryLowerDilation(unittest.TestCase):
    """No dilation=1/2/3/4 fixture carries any bank=0xE1 (225) record
    at all -- the block (or any part of it keyed on that bank) is
    entirely specific to the dilation=8 length class, not a
    relabelling of something already present at lower dilation."""

    def test_no_bank_0xe1_records_at_lower_dilation(self):
        for name in LOW_DILATION_NAMES:
            recs = decode(name)
            e1 = [r for r in recs if r["kind"] == "V" and r.get("bank") == 225]
            self.assertEqual(e1, [], name)

    def test_no_bank_0x81_records_at_dilation8_either(self):
        # Matches Gemm's own "some large-N shapes carry neither 0x81
        # nor 0xE1" observation (tests/test_axera_gemm_sparse_bank_k_boundary.py,
        # PR #1571) -- Conv's dilation=8 carries 0xE1 alone, same as
        # Gemm's K=256 pre-threshold state, not both banks together.
        for name in D8_NAMES:
            recs = decode(name)
            b81 = [r for r in recs if r["kind"] == "V" and r.get("bank") == 0x81]
            self.assertEqual(b81, [], name)


class TestE1Field32And48MatchGemmsOwnConstantsExactly(unittest.TestCase):
    """The core cross-op finding: dilation=8's bank=0xE1 field=32/48
    records carry the EXACT same operand bytes
    tests/test_axera_gemm_bank_81_e1_decode.py (PR #1570) found at
    large-N Gemm shapes and left unexplained -- a real structural
    connection between two previously-separate investigations."""

    def test_field32_matches_gemms_constant(self):
        for name in D8_NAMES:
            recs = decode(name)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("bank") == 225 and r.get("field") == 32
            ]
            self.assertEqual(len(hits), 1, name)
            self.assertEqual(hits[0]["operand"], b"3\x03\x00", name)

    def test_field48_matches_gemms_constant(self):
        for name in D8_NAMES:
            recs = decode(name)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("bank") == 225 and r.get("field") == 48
            ]
            self.assertEqual(len(hits), 1, name)
            self.assertEqual(hits[0]["operand"], b"5\x03\x00\xa1", name)

    def test_exactly_two_bank_0xe1_records_total(self):
        # Only the field=32/48 pair -- none of Gemm's OTHER decoded
        # 0xE1 content (e.g. field=112's own constant) appears here.
        for name in D8_NAMES:
            recs = decode(name)
            e1 = [r for r in recs if r["kind"] == "V" and r.get("bank") == 225]
            self.assertEqual(len(e1), 2, name)
            fields = sorted(r["field"] for r in e1)
            self.assertEqual(fields, [32, 48], name)


if __name__ == "__main__":
    unittest.main()
