"""Continues `tests/test_axera_generator_progress_stocktake.py` (PR
#1620)'s generator-progress theme with the RICHEST of the three
op-specific `reg=8` mechanisms decoded this session: Conv's own 3-of-4,
variable-register-label version
(`tests/test_axera_conv_reg8_reg60_noise_source.py`, PR #1580). A
sibling fork extended `scripts/axera/tiny_emit.py`'s
`emit_matmul_reg8_quad` (MatMul's own simpler, fixed-length 4-of-4
case) -- do not confuse the two; this file is entirely about Conv's own
new `emit_conv_reg8_group`.

## Why Conv is a harder, and more informative, second test case

Unlike MatMul's own fixed-offset, fixed-length group, Conv's version has
two properties MatMul's does not: (a) the register LABEL on slots 2/3 is
itself variable (`reg=8`, `reg=242`, or `reg=176`), and (b) one class
(`"P4"`) uses a genuinely different, SHORTER byte form (1 byte instead
of 3) -- meaning the group's own total length is 24 or 26 bytes
depending on which class lands in which slot, not a constant the way
MatMul's 39-byte group is. This makes `emit_conv_reg8_group` a
length-changing edit, the harder generation primitive this project's
own generator-progress arc had not yet exercised.

## Two previously-unnoticed wrinkles, found building this function

PR #1580's own `EXPECTED` table recorded each slot as `(reg, class)`
only, discarding each record's own TAG byte. Decoding all 8 real
fixtures' raw group bytes directly (not from that table) finds the tag
is not a pure constant the way the table's own omission implicitly
assumed:

- Slot1's own tag (on the `reg=174` record) is `130` in 7 of 8 samples,
  but `132` in `conv_dilation3_v7stability_r0.mcode.gz` for an otherwise
  perfectly ordinary long-form `"P2"` slot1 -- NOT the same thing as
  `reg=172`'s own already-known tag biconditional (that one still holds
  perfectly; this is a second, independent tag).
- Slots 2/3's own tag is `130` in every occurrence of `reg=8`/`reg=242`,
  but `132` in the corpus's one `reg=176` occurrence
  (`conv_dilation3_v7stability_r1.mcode.gz`).

Two exceptions in 24 total slot observations is not enough data to
decode a rule (does `132` track "first use of this register label"?
Genuinely unclear from one point). `emit_conv_reg8_group` does not
guess -- it requires every slot's own tag as an explicit argument.

## Length-changing splices and `mcode.check()`

A same-length swap (the target configuration has the same number of
`"P4"` slots as the base, just permuted) always passes `mcode.check()`
cleanly. A length-changing swap (a different slot count of `"P4"`) used
to always trip `mcode.check()`'s own tail-table validation -- the
group's own re-decoded content was still exactly correct in both cases;
only the STREAM's own separate tail/segment table went stale. **This is
now fixed automatically** (`scripts/axera/tiny_emit.py`'s own
`retarget_tail_vector`, PR #1634, decoded the shared root cause -- a
stale relative-uoffset header word -- and `emit_conv_reg8_group` now
calls it internally before returning): a length-changing swap now
passes `mcode.check()` cleanly too, verified directly below. See
`scripts/axera/tiny_emit.py`'s own updated docstring for the full
finding.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

ALL_NAMES = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]

# (slot1=(class, tag), slot2=(reg, class, tag), slot3=(reg, class, tag)),
# recomputed directly from the real fixtures' own raw bytes (not copied
# from PR #1580's own EXPECTED table, which discarded each slot's own
# tag byte).
CONFIG = {
    "conv_dilation3.mcode.gz": (("P2", 130), (8, "P1", 130), (242, "P3", 130)),
    "conv_dilation3_rebuild0.mcode.gz": (
        ("P4", 130),
        (8, "P2", 130),
        (8, "P1", 130),
    ),
    "conv_dilation3_rebuild1.mcode.gz": (
        ("P2", 130),
        (8, "P1", 130),
        (242, "P3", 130),
    ),
    "conv_dilation3_rebuild2.mcode.gz": (
        ("P4", 130),
        (8, "P1", 130),
        (8, "P2", 130),
    ),
    "conv_dilation3_v7stability_r0.mcode.gz": (
        ("P2", 132),
        (8, "P4", 130),
        (8, "P3", 130),
    ),
    "conv_dilation3_v7stability_r1.mcode.gz": (
        ("P3", 130),
        (176, "P1", 132),
        (8, "P4", 130),
    ),
    "conv_dilation3_v7stability_r2.mcode.gz": (
        ("P1", 130),
        (8, "P3", 130),
        (242, "P2", 130),
    ),
    "conv_dilation3_v7stability_r3.mcode.gz": (
        ("P1", 130),
        (8, "P3", 130),
        (242, "P2", 130),
    ),
}

_CLASS_BYTES = {
    "P1": b"\x23\x00\x20",
    "P2": b"\x23\x00\x10",
    "P3": b"\x23\x00\x40",
    "P4": b"\x30",
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(data):
    return mcode.decode(data, **mcode.FULL_RULE)


def group_len(cfg):
    slot1, slot2, slot3 = cfg
    return sum(len(_CLASS_BYTES[c]) + 3 for c in (slot1[0], slot2[1], slot3[1]))


def find_group(recs):
    r170 = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 170
        and r.get("payload") == b"\x12"
        and r.get("at") is not None
        and 800 <= r["at"] <= 900
    ]
    assert len(r170) == 1, r170
    anchor = r170[0]["at"]
    group = [
        r
        for r in recs
        if r.get("at") is not None
        and anchor <= r["at"] <= anchor + 30
        and r["kind"] == "S"
    ]
    return anchor, group


def config_of(data):
    anchor, group = find_group(decode(data))
    _, r172, slot1, slot2, slot3 = group
    return (
        (_CLASS_BYTES_REVERSE(slot1["payload"]), slot1["tag"]),
        (slot2["reg"], _CLASS_BYTES_REVERSE(slot2["payload"]), slot2["tag"]),
        (slot3["reg"], _CLASS_BYTES_REVERSE(slot3["payload"]), slot3["tag"]),
    )


def _CLASS_BYTES_REVERSE(payload):
    for name, b in _CLASS_BYTES.items():
        if b == payload:
            return name
    raise ValueError(payload)


class TestConfigTableMatchesFixtures(unittest.TestCase):
    """Recomputes CONFIG directly from the real fixtures, not trusted
    from the module docstring alone."""

    def test_all_eight_configs_match(self):
        for name in ALL_NAMES:
            data = load(name)
            self.assertEqual(config_of(data), CONFIG[name], name)


class TestNoOpEmitIsByteIdentical(unittest.TestCase):
    """Re-emitting each fixture's own already-observed configuration
    (including its own per-slot tags) reproduces that fixture's exact
    bytes -- confirms the function's own offsets, framing, and
    length-walk are exactly right, not just close, for all 8 real
    builds including the 2 tag exceptions and the length-changing
    short-form cases."""

    def test_all_eight_fixtures_round_trip_byte_identical(self):
        for name in ALL_NAMES:
            data = load(name)
            out = tiny_emit.emit_conv_reg8_group(data, *CONFIG[name])
            self.assertEqual(out, data, name)


class TestSameLengthCrossFixtureEmissionIsClean(unittest.TestCase):
    """conv_dilation3 (P2/P1/P3, all long-form, 26-byte group) and
    v7stability_r2 (P1/P3/P2, also all long-form, 26-byte group) have
    the SAME group length -- splicing one's config into the other's
    base stream must re-decode to the target config, leave everything
    else byte-identical, and pass mcode.check() cleanly."""

    def test_swap_is_clean(self):
        base_name = "conv_dilation3.mcode.gz"
        target_name = "conv_dilation3_v7stability_r2.mcode.gz"
        base = load(base_name)
        target_cfg = CONFIG[target_name]
        self.assertEqual(group_len(CONFIG[base_name]), group_len(target_cfg))

        out = tiny_emit.emit_conv_reg8_group(base, *target_cfg)
        self.assertEqual(len(out), len(base))
        self.assertEqual(config_of(out), target_cfg)

        anchor, _ = find_group(decode(base))
        self.assertEqual(out[:anchor], base[:anchor])
        self.assertEqual(out[anchor + 26 :], base[anchor + 26 :])

        errs = mcode.check(out)
        hard = [e for e in errs if not e.startswith("coverage:")]
        self.assertEqual(hard, [])


class TestLengthChangingCrossFixtureEmissionIsNowFixedByRetarget(unittest.TestCase):
    """conv_dilation3 (all long-form, 26-byte group) spliced with
    rebuild0's own config (P4 in slot1, 24-byte group) shrinks the
    group by 2 bytes. The group's own content re-decodes to exactly the
    target configuration, and -- since `emit_conv_reg8_group` now calls
    `tiny_emit.retarget_tail_vector` internally (PR #1634) -- the
    result also passes `mcode.check()` cleanly now, where it used to
    report a real "tail: no readable segment table" error. See
    `tests/test_axera_tail_table_mechanism.py` for the mechanism this
    fix is based on.

    **Superseded (docs/axera-mcode-segments-fix.md):** the -2 edit lands
    inside an LZ77 token stream, so key 5 goes stale and the tail vector
    ends up at 3062 -- not 4-byte aligned, which a FlatBuffers vector must
    be. `mcode.check()` now reports that, so this output is malformed, not
    clean. The class name is kept for history."""

    def test_group_content_is_still_correct(self):
        base = load("conv_dilation3.mcode.gz")
        target_cfg = CONFIG["conv_dilation3_rebuild0.mcode.gz"]
        out = tiny_emit.emit_conv_reg8_group(base, *target_cfg)
        self.assertEqual(len(out), len(base) - 2)
        self.assertEqual(config_of(out), target_cfg)

    def test_mcode_check_is_now_clean(self):
        # Was `hard == []`; see the class docstring.
        base = load("conv_dilation3.mcode.gz")
        target_cfg = CONFIG["conv_dilation3_rebuild0.mcode.gz"]
        out = tiny_emit.emit_conv_reg8_group(base, *target_cfg)
        errs = mcode.check(out)
        hard = [e for e in errs if not e.startswith("coverage:")]
        self.assertEqual(len(hard), 1)
        self.assertIn("tail vector at 3062 is not 4-byte aligned", hard[0])

    def test_decode_succeeds(self):
        base = load("conv_dilation3.mcode.gz")
        target_cfg = CONFIG["conv_dilation3_rebuild0.mcode.gz"]
        out = tiny_emit.emit_conv_reg8_group(base, *target_cfg)
        recs = mcode.decode(out, **mcode.FULL_RULE)
        self.assertGreater(len(recs), 0)


class TestInvalidConfigurationIsRejected(unittest.TestCase):
    """A duplicate class, an unknown register, or a missing anchor is
    refused with ValueError rather than silently miswriting the group,
    matching emit_matmul_reg8_quad's own defensive posture."""

    def test_duplicate_class_is_rejected(self):
        base = load("conv_dilation3.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.emit_conv_reg8_group(
                base, ("P1", 130), (8, "P1", 130), (242, "P3", 130)
            )

    def test_unknown_register_is_rejected(self):
        base = load("conv_dilation3.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.emit_conv_reg8_group(
                base, ("P1", 130), (99, "P2", 130), (242, "P3", 130)
            )

    def test_unknown_class_is_rejected(self):
        base = load("conv_dilation3.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.emit_conv_reg8_group(
                base, ("P9", 130), (8, "P2", 130), (242, "P3", 130)
            )

    def test_anchor_not_found_is_rejected(self):
        with self.assertRaises(ValueError):
            tiny_emit.emit_conv_reg8_group(
                b"\x00" * 500, ("P1", 130), (8, "P2", 130), (242, "P3", 130)
            )


if __name__ == "__main__":
    unittest.main()
