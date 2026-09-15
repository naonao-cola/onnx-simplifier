"""Input-side reciprocal slots in single-Mul mcode streams.

Discovery (factorial sweep over Mul[1,8] input ranges, 2026-09-15): the NPU
quantizes float inputs with an integer multiplier derived from the
*reciprocal* of each input's MinMax scale, stored as float32 in the stream:

- Site A (~offset 1122, 4 copies at stride 8): float32(1 / x_scale).
- Site B (~offset 1246, 4 copies at stride 7): float32(1 / y_scale), except
  when the S-unit program takes its short form (base build), where only the
  low 3 bytes of the float appear followed by an 0x82 tag byte and the unit
  shrinks to stride 6.  The S-unit program shape adapts to operand magnitude.

Verified values (quant JSON scales -> stream bytes):

- base (x=0.00779978, y=0.00768933): site A = 128.2088 = 1/x (x4, stride 8);
  site B carries de0c02 = low 3 bytes of float32(1/y) = float32(130.05).
- x10y01 (x=0.07799776, y=0.00076893, z identical to base): site A = 12.8209
  = 1/x (x4, stride 8); site B = 1300.50 = 1/y (x4, stride 7).
- x01y10 (x=0.00077252, y=0.07746022): site A = 1294.47 = 1/x (x4, stride 8);
  site B = 12.91 = 1/y (x4, stride 7).

Open (correlated but undecoded): the 3-byte groups preceding each 42a100
marker (4 copies): base 7c7cee, x10y01 7b7cee (byte0 - 1 for 10x x with z
fixed), mul_sep (x=0.78) 7a7cee (-2 for 100x), x01y10 9b53c3 (all bytes
change when z also moves).  They track the input scales but are not a pure
function of any single scale; pinned here as observations.

Update (2026-09-16): SOLVED -- those "3-byte groups" were a misaligned
read. The real structure is a third reciprocal-family slot:

- Site C (~offset 1421, 4 copies at stride 8, framed `0f <f32> a1 00 <id>`
  like site A): float32(z_scale / (x_scale * y_scale)), the integer
  requant multiplier, bit-exact in all five scratch builds (base, x10y01,
  x01y10, sep, w2 -- including w2's apparent "8 markers", which were just
  its site-A slots matching the old `42 a1 00` motif by float-byte
  coincidence when 1/x fell in [32, 64)). The old motif conflated two
  different sites sharing one frame: site A carries the input reciprocal
  (high byte varies: 43/42/41/44/3f), site C the requant (high byte 42
  in every build so far). `TestRequantMultiplierSlot` pins site C.

RESOLVED (2026-09-16): the short-vs-full selector for site B's S-unit
program is *not* the x/y scale ratio alone -- it is the conjunction of that
ratio being close to 1 **and** x's own zero point being nonzero. y's zero
point plays no part. See ``TestSiteBFormSelectorIsZpX`` below for the
controlled 2x2 build matrix (positive-only calibration for zp=0, a
constant-span negative shift to flip zp without moving the scale ratio)
that pins this down across all eight builds gathered so far, including the
three above.

RESOLVED (2026-09-16, later), corrected twice same day: x's own zero
point (zp_x) is *sometimes* written to the stream as a literal byte. See
``TestZpXLiteralByteWhenPresent`` -- when the unit
``02 10 1b <zp_x> 83 36`` is present, its fourth byte carries zp_x
verbatim, no arithmetic transform needed, with zero exceptions in twelve
independent values so far. Presence is *not* gated by zp_x's magnitude in
any way found yet: it is not a clean "zp_x >= 33" threshold (a first
version of this note claimed that; a follow-up sweep over zp_x in [1, 32]
falsified it the same day), and it is not even true that every zp_x >= 33
uses it (a second correction, caught reusing ``mul_1x8`` -- zp_x=128 uses
the *opaque* form, while ``mul_1x8_recip_x01``'s zp_x=127 uses the
literal one). ``TestZpXImmediateRegion`` pins the raw bytes of the two
opaque forms this literal unit is not used for, still undecoded.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")

CASES = {
    # fixture: (x_scale, y_scale, siteB_full_float, group_hex)
    "mul_1x8.mcode.gz": (0.007799775805324316, 0.0076893349178135395, False, "7c7cee"),
    "mul_1x8_recip_x10.mcode.gz": (
        0.07799775898456573,
        0.0007689335034228861,
        True,
        "7b7cee",
    ),
    "mul_1x8_recip_x01.mcode.gz": (
        0.0007725197938270867,
        0.07746022194623947,
        True,
        "9b53c3",
    ),
}

# Output scales for the same fixtures (quant JSON, pinned alongside the
# inputs so site C is computable without a rebuild).
_Z_SCALES = {
    "mul_1x8.mcode.gz": 0.007151617668569088,
    "mul_1x8_recip_x10.mcode.gz": 0.007151617202907801,
    "mul_1x8_recip_x01.mcode.gz": 0.005844127852469683,
}

# Unary op (Neg): same site-A mechanism for its single input.  All 13 builds
# of the Neg input-range sweep carry float32(1/x) x4 at stride 8 (offsets
# 1067/1068 family); pinned here for the neg_1x8 fixture (x=0.00768933).
NEG_CASES = {
    "neg_1x8.mcode.gz": 0.0076893349178135395,
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestInputReciprocalSlots(unittest.TestCase):
    def test_site_a_carries_inverse_x_scale(self):
        for name, (xs, _ys, _full, _grp) in CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: site A hits for 1/x={1.0 / xs}")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site A stride")
        for name, xs in NEG_CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: unary site A hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: unary site A stride")

    def test_site_b_carries_inverse_y_scale(self):
        for name, (_xs, ys, full, _grp) in CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / ys)
            found = hits(data, pat)
            if full:
                self.assertEqual(
                    len(found), 4, f"{name}: site B hits for 1/y={1.0 / ys}"
                )
                strides = {b - a for a, b in zip(found, found[1:])}
                self.assertEqual(strides, {7}, f"{name}: site B stride")
            else:
                # Short S-unit form: low 3 bytes of the float, 0x82 tag.
                self.assertGreaterEqual(
                    len(hits(data, pat[:3])), 3, f"{name}: site B short form"
                )

    def test_group_values_pinned(self):
        marker = bytes.fromhex("42a100")
        for name, (_xs, _ys, _full, grp) in CASES.items():
            data = load(name)
            idx = hits(data, marker)
            self.assertEqual(len(idx), 4, f"{name}: marker count")
            for i in idx:
                self.assertEqual(data[i - 3 : i].hex(), grp, f"{name}: group")


class TestRequantMultiplierSlot(unittest.TestCase):
    """Site C: float32(z_scale / (x_scale * y_scale)) x4 at stride 8.

    The integer requant multiplier, bit-exact -- the value the NPU's
    fixed-point core multiplies the (x_q - zpx)(y_q - zpy) product by
    before shifting down to the output scale. Same `0f <f32> a1 00 <id>`
    framing as site A (verified context bytes), different payload.
    """

    def test_site_c_carries_output_over_input_product_scale(self):
        for name, (xs, ys, _full, _grp) in CASES.items():
            data = load(name)
            req = struct.pack("<f", _Z_SCALES[name] / (xs * ys))
            found = hits(data, req)
            self.assertEqual(len(found), 4, f"{name}: site C hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site C stride")


if __name__ == "__main__":
    unittest.main()


# NOTE: this class lives after the __main__ guard on purpose -- the
# stacked site-C diff edits the docstring and appends its own class
# above, and keeping this addition textually disjoint avoids a merge
# collision between the two PRs.
class TestOutputScaleQuads(unittest.TestCase):
    """Output scale quads: `03 <float32(z_scale)> 81 82` x4 at stride 7.

    The output-side counterpart to the input slot families: four copies
    of the exact float32 output scale, each framed by a 0x03 lead byte
    and constant 0x81/0x82 tag bytes (verified on all three fixtures).
    Two things this settles:

    - The `81` tag is NOT the output zero point: x01's z zero point is
      110 (0x6e) yet its quads still read `... 81 82`. Raw zp immediates
      (128/127/126/129/110 census over the program region) and zp*scale
      combined floats were also ruled out across five builds -- zero
      points live elsewhere (still open).
    - Emitter caution: base and x10 carry the same real-valued z scale
      to ~1e-9 yet their quad floats differ by 1 ULP (1e vs 1d) -- the
      two builds' float64 scales straddle a float32 boundary. Patch by
      exact float32 word, never by recomputing from float64.
    """

    _Z = {
        "mul_1x8.mcode.gz": 0.007151617668569088,
        "mul_1x8_recip_x10.mcode.gz": 0.007151617202907801,
        "mul_1x8_recip_x01.mcode.gz": 0.005844127852469683,
    }

    def test_quads_carry_exact_float32_output_scale(self):
        for name, zs in self._Z.items():
            data = load(name)
            pat = struct.pack("<f", zs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: quad count")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {7}, f"{name}: quad stride")
            for i in found:
                self.assertEqual(data[i - 1], 0x03, f"{name}@{i}: lead")
                self.assertEqual(data[i + 4 : i + 6].hex(), "8182", f"{name}@{i}: tags")


class TestSiteBFormSelectorIsZpX(unittest.TestCase):
    """Site B's short-vs-full form tracks x's zero point, not the x/y ratio.

    Four new Mul[1,8] builds complete a 2x2 grid at x/y scale ratio close to
    1 (where the pre-existing ratio-only hypothesis predicted short form
    throughout), crossed with zp_x and zp_y independently zero or not:

    - ``mul_1x8_zp_both0`` (positive-only calibration for both inputs: any
      MinMax range with min >= 0 clips the computed zero point to 0):
      zp=(0, 0), ratio 1.002 -> **full**.
    - ``mul_1x8_zp_yonly`` (x unchanged/positive, y shifted negative): zp=(0,
      16), ratio 1.043 -> **full**.
    - ``mul_1x8_zp_xonly`` (x shifted negative, y unchanged/positive): zp=(19,
      0), ratio 0.974 -> **short**.
    - ``mul_1x8_zp_both`` (both shifted negative by the same constant as
      xonly/yonly -- same span, hence unchanged scale, as their unshifted
      counterparts): zp=(19, 16), ratio 1.014 -> **short**.

    "Shifted negative by a constant" is the controlled part: shifting a
    calibration array by a constant changes min and max by the same amount,
    so the span (and thus the MinMax scale) is unchanged while the zero
    point moves off 0 -- isolating zp as the only variable, unconfounded by
    scale or ratio drift, unlike comparing builds from independent random
    calibration draws.

    Combined with the three base-file builds above (base: zp=(128, 126),
    ratio 1.014, short; x10y01/x01y10: zp with both components nonzero,
    ratio far from 1, full), all eight builds agree: short form iff ratio is
    close to 1 **and** zp_x != 0. zp_y never matters. This resolves the
    handoff's open "short-vs-full site-B selection rule" question -- it was
    never a pure ratio rule.

    (A same-x_scale, different-zp_x pair -- ``mul_1x8_zp_xonly`` shifted by
    -0.2 giving zp_x=19 vs. an unlanded -0.1 shift giving zp_x=6 -- produced
    byte-identical site-A-adjacent group/tag bytes despite the different zp
    magnitude, so whatever selects short-vs-full reads zp_x as a boolean,
    not a literal value; the group bytes there are not zp's own encoding,
    since they also depend on which MinMax scale formula fired -- still
    open, not pinned here.)
    """

    # fixture: (x_scale, y_scale, zp_x, zp_y, full) -- zp values are quant
    # JSON ground truth (not independently recoverable from the stream
    # itself, see the class docstring); `full` is the observed site B form.
    CASES = {
        "mul_1x8_zp_both0.mcode.gz": (
            0.00782180204987526,
            0.007805639877915382,
            0,
            0,
            True,
        ),
        "mul_1x8_zp_yonly.mcode.gz": (
            0.00782180204987526,
            0.007497101556509733,
            0,
            16,
            True,
        ),
        "mul_1x8_zp_xonly.mcode.gz": (
            0.007604781538248062,
            0.007805639877915382,
            19,
            0,
            False,
        ),
        "mul_1x8_zp_both.mcode.gz": (
            0.007604781538248062,
            0.007497101556509733,
            19,
            16,
            False,
        ),
    }

    def test_site_a_present_full_form_regardless(self):
        # Site A (x's own reciprocal scale) is unaffected by any of this --
        # only site B's form changes.
        for name, (xs, _ys, _zpx, _zpy, _full) in self.CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: site A hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site A stride")

    def test_site_b_form_tracks_zp_x_not_ratio(self):
        for name, (_xs, ys, zpx, _zpy, full) in self.CASES.items():
            self.assertEqual(
                full, zpx == 0, f"{name}: full iff zp_x == 0 (this dataset)"
            )
            data = load(name)
            pat = struct.pack("<f", 1.0 / ys)
            found = hits(data, pat)
            if full:
                self.assertEqual(len(found), 4, f"{name}: expected full-form site B")
                strides = {b - a for a, b in zip(found, found[1:])}
                self.assertEqual(strides, {7}, f"{name}: site B stride")
            else:
                self.assertEqual(len(found), 0, f"{name}: expected no full-form site B")
                short_hits = hits(data, pat[:3])
                self.assertGreaterEqual(
                    len(short_hits), 3, f"{name}: expected short-form site B"
                )


class TestZpXImmediateRegion(unittest.TestCase):
    """Localizes zp_x's own magnitude to a small pre-site-A region.

    An 8-build shift sweep (x shifted down by 0.04..0.36 in steps, y and the
    op fixed) holds x_scale constant (0.007604781538248062, confirmed
    identical across every build with zp_x != 0 -- span is shift-invariant)
    while sweeping zp_x = round(-min_x / x_scale) through 3, 9, 14, 24, 30,
    35, 40. Diffing the region 28-32 bytes *before* site A (offsets
    ~1092-1100, i.e. immediately preceding the frame ``84 22 .. 84 24 ..``
    that leads into site A's own preamble) shows a variable-width field that
    changes with the sweep and separates into three forms -- see
    ``TestZpXLiteralByteWhenPresent`` below for which zp_x values land in
    which form; it is not purely magnitude-gated:

    - zp_x=0: a fixed 3-byte tail ``00 10 84`` (no extra immediate).
    - The ``02 10 1b <zp_x> 83 36`` literal form (some zp_x, see below).
    - Two other forms (tag ``0x83`` or ``0xa1``) this class still pins
      as raw bytes without a decode.

    The two non-literal forms' own varying byte is a *deterministic*
    function of zp_x alone -- confirmed by rebuilding zp_x=9 with three
    wildly different y calibration ranges (y_scale 0.000193 to 0.019528,
    a 100x span) and getting the identical byte (``0x2c``) every time, so
    it is not register-allocation noise tied to incidental build context
    (an earlier, reasonable-sounding guess this rules out) -- but no
    formula tried (linear, affine, XOR, byte-reversal, mod-256 arithmetic
    on zp_x) reproduces it from zp_x. Most likely explanation: a lookup
    table or fixed-point reciprocal-style computation elsewhere in the
    S-unit ISA that this project hasn't reverse-engineered yet, consistent
    with the other "residual, unmodeled" bytes already documented for this
    stream (module docstring, and ``scripts/axera/README.md``'s "Files"
    section).
    """

    # fixture: (x_scale, region bytes at [1090:1090+len], zp_x)
    CASES = {
        "mul_1x8_zp0sweep.mcode.gz": (
            0.007664938922971487,
            bytes.fromhex("8422001084240020842600"),
            0,
        ),
        "mul_1x8_zp3sweep.mcode.gz": (
            0.007604781538248062,
            bytes.fromhex("842201101b8386002084"),
            3,
        ),
        "mul_1x8_zp9sweep.mcode.gz": (
            0.007604781538248062,
            bytes.fromhex("842201101ba12c02a10020"),
            9,
        ),
        "mul_1x8_zp32sweep.mcode.gz": (
            0.007604781538248062,
            bytes.fromhex("842201101ba1b602a100208426"),
            32,
        ),
    }

    def test_region_bytes_pinned(self):
        # Searched, not sliced at a fixed offset: this region's start drifts
        # by a byte or two build to build (bytes further upstream can shift
        # it), which is itself part of why a fixed-offset byte correlation
        # search across many builds produces misleading noise -- see the
        # module-level lesson recorded in TestZpXLiteralByteWhenPresent.
        for name, (_xs, region, _zpx) in self.CASES.items():
            data = load(name)
            idx = data.find(region, 1085, 1130)
            self.assertNotEqual(
                idx,
                -1,
                f"{name}: pre-site-A region changed -- re-examine before trusting"
                " the docstring's regime split",
            )

    def test_site_a_unaffected(self):
        # Confirms the sweep only ever touches the pre-site-A region --
        # site A itself (x's own reciprocal scale) is stable throughout.
        for name, (xs, _region, _zpx) in self.CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: site A hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site A stride")

    def test_zp9_opaque_byte_is_y_independent(self):
        # mul_1x8_zp9_narrowy: same zp_x=9 (same x shift) as
        # mul_1x8_zp9sweep, but y's calibration range is 0.9-1.1 instead of
        # 0-2 (y_scale 0.000784 vs 0.007806, a 10x difference). The opaque
        # byte at this slot is identical either way -- the CSE/incidental
        # register-allocation guess in the docstring above would predict
        # otherwise, since the two builds share nothing about y.
        zp9_region = self.CASES["mul_1x8_zp9sweep.mcode.gz"][1]
        data = load("mul_1x8_zp9_narrowy.mcode.gz")
        self.assertIn(
            zp9_region,
            data,
            "mul_1x8_zp9_narrowy: the zp_x=9 opaque region changed when only"
            " y's calibration range changed -- the determinism claim in the"
            " docstring above needs re-examining",
        )


class TestZpXLiteralByteWhenPresent(unittest.TestCase):
    """When present, this unit's fourth byte is zp_x, verbatim. RESOLVED
    (partially -- see the correction below, which replaces an overclaim
    from when this class was first landed).

    The class above localized zp_x's effect but could not decode the
    immediate; the reason turned out to be that the earlier analysis
    compared bytes at a *fixed absolute offset* across builds whose
    surrounding units have different lengths, silently comparing unrelated
    fields (this is also, with hindsight, why a whole-stream byte/word
    correlation sweep against zp_x topped out around |r|=0.8 instead of
    hitting 1.0 anywhere: it was the same misalignment, just averaged over
    more offsets). Re-deriving each build's variable-width unit with
    ``mcode.decode()`` instead of a fixed slice removes the misalignment
    and turns up a fixed 6-byte form:

        02 10 1b <zp_x> 83 36

    where every byte except the fourth is constant (``02``: p=2, i.e.
    3-byte payload; ``10 1b``: constant payload prefix; ``83``: tag;
    ``36``: register) and the fourth byte *is* zp_x, verbatim, confirmed
    for twelve independent values (6, 8, 15, 18, 22, 25, 33, 35, 36, 51,
    80, and x01's 127 -- see the second correction below) with zero
    exceptions and zero arithmetic transform needed.

    CORRECTION (2026-09-16, same day): this class originally claimed the
    threshold "sits strictly between 32 and 33" -- that a dense follow-up
    sweep over zp_x in [1, 32] immediately falsified. This form also fires
    for several zp_x < 33 (6, 8, 15, 18, 22, 25 above), while *other*
    zp_x in the very same range (1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14,
    16, 20, 24, 27, 29, 30, 31, 32) use one of two different, still-opaque
    forms (``TestZpXImmediateRegion`` above) instead -- non-monotonically:
    e.g. 8 gets this form but 9 through 14 do not, then 15 does again.

    SECOND CORRECTION (2026-09-16, later still): the surviving claim above
    -- "zp_x >= 33 uses this form 100% of the time, 5/5" -- is *also*
    false, caught while reusing the well-established ``mul_1x8`` (base,
    zp_x=128) fixture as an emitter reference: it uses the opaque
    ``0x83``-tag form, not this one, and so do ``mul_1x8_recip_x10``
    (zp_x=128) and ``mul_1x8_w2`` (zp_x=128) -- while ``mul_1x8_recip_x01``
    (zp_x=127, one less) *does* use this literal form. So the "5/5 for
    zp_x >= 33" regularity was an artifact of this sweep family's specific
    construction (a fixed x array shifted down, y and the op otherwise
    untouched, zp_y always 0) rather than a property of zp_x's magnitude
    at all -- consistent with the already-documented non-monotonicity
    below 33, just not previously shown to extend above it too.

    Practical upshot for an emitter, restated without the magnitude
    claim: this pattern can be used *opportunistically* -- search a
    reference build for ``02 10 1b <byte> 83 36`` and trust the byte if
    found, since it has never once been wrong across any build gathered
    (twelve for twelve) -- but its *absence* says nothing about zp_x's
    value at any magnitude, small or large; an emitter must handle "not
    found" as a real, common case, not an edge case.

    Still open: the two zp_x <= 32 forms' actual encoding
    (``TestZpXImmediateRegion`` above), what selects between all three
    forms, and whether y ever gets an analogous literal-byte encoding
    (this sweep held y's zero point at 0 throughout, by construction --
    see ``TestSiteBFormSelectorIsZpX``, which already showed zp_y plays no
    part in site B's form selector; whether it gets *its own* literal slot
    elsewhere is untested).
    """

    # fixture: (zp_x,) -- x_scale is one of two builds that agree to
    # ~2e-14 (0.007604781538248062 or 0.0076047820039093494, a float64
    # rounding-mode artifact of which MinMax reduction order a given build
    # happened to use) across every case in this sweep family; irrelevant
    # to this class, which never reads x_scale.
    CASES = {
        "mul_1x8_zp6sweep.mcode.gz": 6,
        "mul_1x8_zp25sweep.mcode.gz": 25,
        "mul_1x8_zp33sweep.mcode.gz": 33,
        "mul_1x8_zp35sweep.mcode.gz": 35,
        "mul_1x8_zp80sweep.mcode.gz": 80,
        # Already-committed fixture, not a new build -- zp_x=127 here (one
        # less than mul_1x8/x10/w2's 128, which use the *opaque* form; see
        # the second correction in the class docstring).
        "mul_1x8_recip_x01.mcode.gz": 127,
    }

    _CONST_PREFIX = bytes.fromhex("02101b")
    _CONST_SUFFIX = bytes.fromhex("8336")

    def test_zp_x_is_the_literal_fourth_byte(self):
        for name, zpx in self.CASES.items():
            data = load(name)
            unit = self._CONST_PREFIX + bytes([zpx]) + self._CONST_SUFFIX
            found = hits(data, unit)
            self.assertEqual(
                len(found),
                1,
                f"{name}: expected exactly one `02 10 1b {zpx:02x} 83 36` unit",
            )
            # And confirm no *other* byte value at that position would also
            # match the constant framing -- i.e. the byte really is load
            # bearing, not incidentally equal to zp_x.
            wrong = self._CONST_PREFIX + bytes([(zpx + 1) % 256]) + self._CONST_SUFFIX
            self.assertEqual(
                hits(data, wrong), [], f"{name}: framing is exact, not fuzzy"
            )
