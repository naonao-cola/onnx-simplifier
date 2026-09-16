"""Batched MatMul's short-form `A` quad "var" tag byte: shape-dependent,
not a coin flip and not calibration-dependent -- but not reduced to a
single arithmetic formula either.

`tests/test_axera_matmul_batched_site_a.py` (PR #1502, merged) found
that rank-3 `A` gets a real quad in the short 3-byte form: `<3 bytes,
low(1/A_scale)> 82 <var> 02` x4 at stride 6, and noted the middle byte
("var": `0x62` in its batched build, `0x5e` in its broadcast build)
"varies build to build ... not part of the fixed frame" without
investigating further.

## Ruled out first: not the table-order coin flip

`tests/test_axera_matmul_offset_table_coinflip.py` (PR #1506, merged)
proved a superficially similar "varies build to build" MatMul byte
sequence (the `A_offset`/`B_offset` name-table order) was actually an
unconditioned per-compile coin flip. Checked directly against the two
already-committed PR #1502 fixtures first: both `matmul_2x4x8x8_batched`
(`var=0x62`) and `matmul_2x4x8x8_broadcast` (`var=0x5e`) have the
**same** table order (`B_offset` before `A_offset`) despite different
`var` values -- already inconsistent with a pure table-order effect
before any new builds were needed.

## Confirmed: stable across independent rebuilds, not calibration-dependent

**5 independent rebuilds** of one unchanged `A[2,4,8] @ B[2,8,8]`
config (matching PR #1502's own "batched" shape) all give `var=0x62`,
including rebuilds landing on *both* sides of the `A_offset`/`B_offset`
coin flip (3 in `A,B` order, 1 in `B,A` order among the two committed
here) -- ruling out both a coin flip and any dependence on table order.

Holding that exact shape fixed and varying only `A`'s calibration range
(narrow `±0.5`, baseline `±1.0`, wide `±3.0` -- three different
`A_scale` values, `0.00392`/`0.00784`/`0.02352`) leaves `var` at `0x62`
for the baseline and wide builds. The narrow build shows something
distinct and worth recording on its own (see below) rather than a
simple "var changed" result.

## Real, but shape-dependent: not a single-dimension formula

Varying shape while holding `A`'s calibration fixed changes `var`:

| shape (batch, M, K, N) | var |
| --- | --- |
| `(2, 4, 8, 8)` (baseline) | `0x62` |
| `(3, 4, 8, 8)` (batch 2->3) | `0x62` (unchanged) |
| `(4, 4, 8, 8)` (batch 2->4) | `0x30` |
| `(2, 8, 8, 8)` (M 4->8) | `0x2c` |
| `(2, 4, 16, 8)` (K 8->16) | `0x30` |
| `(2, 4, 8, 16)` (N 8->16) | `0x62` (unchanged) |

`N` does not affect `var` at all (matches `B`'s own site-A treatment
being N-agnostic in the same way `B`'s reciprocal scale never encodes
`N`). `batch`, `M`, and `K` each affect it, but not through any single
tested arithmetic combination: `batch=3` leaves `var` unchanged from
`batch=2` while `batch=4` changes it; `(4,4,8,8)` and `(2,4,16,8)` --
different shapes with no obvious shared dimension -- coincidentally
land on the same value (`0x30`) while `(2,8,8,8)` (same total `A`
element count, `128`, as both of those) lands on a third value
(`0x2c`). Consistent with `var` encoding some compiler-internal
tiling/allocation quantity derived from shape, the way this project has
repeatedly found for other still-opaque framing bytes, not a value with
an obvious closed-form relationship to `batch`/`M`/`K`/`N` individually.
Not decoded further here -- reported as a real, narrowed, honest
partial result.

## A second, distinct finding: the quad's own framing can change with `A`'s scale

At the narrow-calibration build (`A` range `±0.5`, `A_scale =
0.003920023795217276`, so `1/A_scale = 255.1004923...`, notably close
to `255`), the pattern search for the 3-byte short form still matches
(any 4-byte match trivially contains its own first-3-byte substring),
but the **actual bytes present are the full 4-byte float32** of
`1/A_scale` (`ba 19 7f 43`), immediately followed by `81 62` --
**not** the short form's `82 <var> 02` tag, and at **stride 7**, not
6. The trailing byte is still `0x62` -- the same value the baseline and
wide-calibration builds at this identical shape show -- so this is
consistent with the same underlying field, just written in a different
on-disk form (full float vs. truncated-plus-tag) depending on `A`'s
own scale. *Why* this specific build crosses into the full-form
encoding (whether it's triggered by `1/A_scale` approaching `255`,
by some byte-collision-avoidance rule, or by something else) is not
investigated further here -- flagged as a real, reproducible,
motivated lead for whoever looks at it next, the same way this
project's other still-open leads are recorded.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def offset_table_order(data):
    window = data[195:270]
    ia = window.find(b"A_offset")
    ib = window.find(b"B_offset")
    return ("A", "B") if ia < ib else ("B", "A")


def find_short_form_var(data, a_scale):
    """Locate the short-form A quad (`<3 bytes> 82 <var> 02` x4, stride
    6) and return its 4 var bytes, or [] if absent."""
    short = struct.pack("<f", 1.0 / a_scale)[:3]
    found = [i for i in range(len(data) - 5) if data[i : i + 3] == short]
    for start in found:
        run = [start]
        i = start + 6
        while i in found:
            run.append(i)
            i += 6
        if len(run) == 4:
            return [data[i + 4] for i in run]
    return []


class TestVarByteIsStableAcrossIndependentRebuilds(unittest.TestCase):
    """Two independent rebuilds of the identical baseline shape, landing
    on opposite sides of the A_offset/B_offset coin flip (PR #1506),
    both give var=0x62 -- not a coin flip, not order-dependent."""

    A_SCALE = 0.007840047590434551

    def test_rebuild0_and_rebuild2_have_opposite_table_order(self):
        r0 = load("matmul_var_byte_rebuild0.mcode.gz")
        r2 = load("matmul_var_byte_rebuild2.mcode.gz")
        self.assertNotEqual(offset_table_order(r0), offset_table_order(r2))

    def test_both_give_the_same_var_byte(self):
        r0 = load("matmul_var_byte_rebuild0.mcode.gz")
        r2 = load("matmul_var_byte_rebuild2.mcode.gz")
        v0 = find_short_form_var(r0, self.A_SCALE)
        v2 = find_short_form_var(r2, self.A_SCALE)
        self.assertEqual(v0, [0x62, 0x62, 0x62, 0x62])
        self.assertEqual(v2, [0x62, 0x62, 0x62, 0x62])


class TestVarByteIsNotCalibrationDependentAtFixedShape(unittest.TestCase):
    """Wide-calibration rebuild of the identical shape still gives
    var=0x62 (the narrow-calibration case is handled separately below,
    since it changes the quad's framing itself, not just the value)."""

    def test_wide_calibration_gives_the_same_var_byte(self):
        data = load("matmul_var_byte_acal_wide.mcode.gz")
        a_scale = 0.02352014183998108
        self.assertEqual(find_short_form_var(data, a_scale), [0x62, 0x62, 0x62, 0x62])


class TestVarByteIsShapeDependentNotSingleFormula(unittest.TestCase):
    # fixture: (a_scale, expected var, what changed from baseline)
    CASES = [
        ("matmul_var_byte_batch3.mcode.gz", 0.007840047590434551, 0x62, "batch 2->3"),
        ("matmul_var_byte_batch4.mcode.gz", 0.007840047590434551, 0x30, "batch 2->4"),
        ("matmul_var_byte_mkn_8x8x8.mcode.gz", 0.007840047590434551, 0x2C, "M 4->8"),
        ("matmul_var_byte_mkn_4x16x8.mcode.gz", 0.007840047590434551, 0x30, "K 8->16"),
        ("matmul_var_byte_mkn_4x8x16.mcode.gz", 0.007840047590434551, 0x62, "N 8->16"),
    ]

    def test_shape_changes_produce_the_documented_var_values(self):
        for fname, a_scale, expected, label in self.CASES:
            data = load(fname)
            var = find_short_form_var(data, a_scale)
            self.assertEqual(var, [expected] * 4, f"{label} ({fname})")

    def test_n_alone_does_not_change_var_from_baseline(self):
        # mkn_4x8x16 changes only N (8->16) from the (2,4,8,8) baseline
        # shape that gives var=0x62 -- confirms N has no effect, unlike
        # batch/M/K which all changed var in at least one tested case.
        data = load("matmul_var_byte_mkn_4x8x16.mcode.gz")
        self.assertEqual(
            find_short_form_var(data, 0.007840047590434551), [0x62, 0x62, 0x62, 0x62]
        )


class TestNarrowCalibrationSwitchesQuadFraming(unittest.TestCase):
    """At A_scale=0.00392 (1/A_scale close to 255), the same shape's A
    quad appears as a full 4-byte float + `81 <var>` at stride 7,
    instead of the usual truncated 3-byte + `82 <var> 02` at stride 6
    -- a distinct, real framing change, not just a value change. The
    trailing byte is still 0x62, matching this shape's other builds."""

    def test_full_four_byte_form_present_at_stride_7(self):
        data = load("matmul_var_byte_acal_narrow.mcode.gz")
        a_scale = 0.003920023795217276
        full = struct.pack("<f", 1.0 / a_scale)
        found = [i for i in range(len(data) - 3) if data[i : i + 4] == full]
        self.assertEqual(len(found), 4)
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {7})
        for i in found:
            self.assertEqual(data[i + 4], 0x81, f"@{i}: tag byte")
            self.assertEqual(data[i + 5], 0x62, f"@{i}: var byte")

    def test_short_form_frame_is_absent_here(self):
        # The usual `82 <anything> 02` tag right after the truncated
        # 3-byte prefix does NOT appear -- byte 3 after each hit is the
        # float's own 4th byte (0x43), not the short-form's 0x82 tag.
        data = load("matmul_var_byte_acal_narrow.mcode.gz")
        a_scale = 0.003920023795217276
        short = struct.pack("<f", 1.0 / a_scale)[:3]
        found = [i for i in range(len(data) - 2) if data[i : i + 3] == short]
        for i in found:
            self.assertNotEqual(
                data[i + 3], 0x82, f"@{i}: should not be short-form tag"
            )


if __name__ == "__main__":
    unittest.main()
