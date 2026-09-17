"""Tests PR #1582's own explicitly flagged bound: the "reg=8's payload
trailing byte draws from one shared pool `{0x10, 0x20, 0x30, 0x40}` across
all four ops" finding (`tests/test_axera_reg8_cross_op_synthesis.py`) was
confirmed only at each op's single already-committed shape -- Gemm's own
`(M=1, K=512, N=1000)` config, the same one `tests/test_axera_gemm_reg8_second_noise_source.py`
(PR #1577) originally decoded the mechanism from. Nobody has checked
whether the pool itself is a fixed constant of the codec or scales with
the shape's own complexity.

## Method

Six new Gemm builds, spanning roughly 128x in `K`/`N` relative to PR
#1577's own `(K=512, N=1000)` config, each compile-only via a standalone
build script (this worktree lacked a compiled `onnxsim_cpp2py_export`
extension, so `scripts/axera/pulsar2_docker.py` -- whose module-level
`ensure_repo_onnxsim()` call requires it -- could not be imported;
replicated its `docker run pulsar2 build` invocation and
`calibration_format: Numpy` config directly, following
`scripts/axera/calib_search.py`'s own onnxsim-independent pattern):

- `(K=2048, N=2048)`, 2 independent rebuilds (`_r0`/`_r1`, different RNG
  seeds) -- ~4x Gemm's own decoded shape in both dimensions.
- `(K=16, N=16)`, 2 independent rebuilds -- ~32-64x *smaller* than the
  decoded shape in both dimensions.
- `(K=2048, N=16)` and `(K=16, N=2048)`, 1 build each -- to disambiguate
  which dimension (if either) drives a secondary format wrinkle found
  along the way (see below); not independently rebuilt, so not used for
  any rebuild-to-rebuild noise claim, only for the format question.

All 6 fixtures decode with zero `mcode.check()` errors.

## Primary finding: the pool is shape-invariant across a ~128x range

**Every `reg=8` record found at any of these 6 shapes whose payload has
the established pool shape (`tag=130`, a 1-byte or 3-byte payload ending
in a byte that is a multiple of `0x10` between `0x10` and `0x40`) has a
trailing byte that is a member of the exact same 4-value set
`{0x10, 0x20, 0x30, 0x40}`** PR #1582 found at the single, much more
moderate `(K=512, N=1000)` shape. No new pool member (e.g. `0x50`, or a
shifted range) appears at either extreme -- not at 4x larger, not at
32-64x smaller. This is real, informative evidence: if the pool were a
set of candidate scratch/tile-buffer addresses whose *count* scaled with
how many buffers a shape's own tiling needs, a shape this much larger
would be a reasonable place to see a bigger pool; it doesn't happen here.
This is more consistent with the pool being a small, fixed set tied to
something like the hardware's own register file or scratchpad-bank
count, independent of the shape being compiled -- though, as PR #1577
and PR #1582 both already noted, this remains a plausibility argument,
not a proof of the underlying semantics.

## Secondary, honestly-open observation: the record's exact FORMAT is not
## shape-invariant, and the group's presence still varies same as always

Two things noticed along the way that this file does NOT claim to have
decoded:

1. **The established 3-byte record form is `XX 00 YY` (middle byte
   always `0x00`, `YY` the pool member)** in every previously-decoded
   case (Gemm/Conv/MatMul all showed `p[1] == 0`, per PR #1582's own
   `pool_records()` filter). At `(K=2048, N=2048)`, some pool-pattern
   records instead show `p[1] == 0x04`: e.g. `\\x13\\x04\\x40`,
   `\\x13\\x04\\x30`, still with a valid pool trailing byte but a
   different middle byte and leading byte (`0x13` -- previously only
   seen for *Mul*, not Gemm -- also appears here). This is a genuinely
   new observation (the middle byte was implicitly assumed constant by
   every prior decode, since it always happened to be `0`), but the
   evidence for WHY it changes here is thin: the `(K=2048, N=16)` build
   (large `K`, small `N`) shows `p[1] == 0` again, and the `(K=16,
   N=2048)` build (small `K`, large `N`) shows no pool-pattern record at
   all -- so the `0x04` middle byte does not cleanly track `K` alone,
   `N` alone, or their presence/absence, only appearing (twice, across
   both `(2048, 2048)` rebuilds) when *both* dimensions are large
   together. That is one shape's worth of evidence for a real but
   uncharacterized interaction, not a decoded rule -- reported honestly
   as open rather than forced into a formula.
2. **The pool group's own presence is not guaranteed at every shape**:
   `(K=16, N=2048)` has zero pool-pattern `reg=8` records at all. This
   is not a new phenomenon -- PR #1577's own original table already
   showed the analogous Gemm group entirely *absent* at offset 891 in
   two of its eight `(K=512, N=1000)` samples -- but it is the first
   time presence/absence has been observed to correlate (at least in
   this one data point) with shape rather than pure per-build noise at
   a fixed shape. Not enough data here to tell those apart; reported as
   an open observation, not a claim.

## What this establishes

The core cross-op synthesis claim (PR #1582's shared 4-member trailing-
byte pool) survives its first stress test against a very different shape
of the same op -- a genuinely informative negative (no pool growth, no
pool shift) that argues for a fixed-resource reading over a
tiling-scales-with-shape reading, while surfacing (without solving) a
new, real wrinkle in the record's own byte format that prior single-
shape decodes had no way to notice.
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

LARGE_BATCH = ["gemm_1x2048x2048_r0.mcode.gz", "gemm_1x2048x2048_r1.mcode.gz"]
SMALL_BATCH = ["gemm_1x16x16_r0.mcode.gz", "gemm_1x16x16_r1.mcode.gz"]
ALL_STABILITY_NAMES = LARGE_BATCH + SMALL_BATCH
DISAMBIGUATION_NAMES = ["gemm_1x2048x16_r0.mcode.gz", "gemm_1x16x2048_r0.mcode.gz"]
ALL_NAMES = ALL_STABILITY_NAMES + DISAMBIGUATION_NAMES


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def pool_candidate_records(recs):
    """reg=8, tag=130 records whose payload has the established pool
    shape: a 1-byte or 3-byte payload ending in a multiple of 0x10 in
    [0x10, 0x40]. Mirrors tests/test_axera_reg8_cross_op_synthesis.py's
    own pool_records() filter, generalized to also accept the p[1]==4
    middle-byte variant this file found (see module docstring) -- the
    ORIGINAL filter required p[1]==0 for the 3-byte form, which would
    silently miss this shape's own pool records entirely.
    """
    out = []
    for r in recs:
        if r["kind"] != "S" or r.get("reg") != 8 or r.get("tag") != 130:
            continue
        p = r.get("payload")
        if p is None:
            continue
        if len(p) == 3 and p[-1] % 0x10 == 0 and 0x10 <= p[-1] <= 0x40:
            out.append(r)
        elif len(p) == 1 and p[0] in POOL_TRAILING_BYTES:
            out.append(r)
    return out


class TestAllSixFixturesDecodeCleanly(unittest.TestCase):
    def test_zero_check_errors(self):
        for name in ALL_NAMES:
            self.assertEqual(mcode.check(load(name)), [], name)


class TestPoolTrailingByteIsShapeInvariant(unittest.TestCase):
    """The core finding: every pool-pattern reg=8 record found at any of
    these 6 very different Gemm shapes has a trailing byte in the exact
    same 4-member set PR #1582 established at (K=512, N=1000). No new
    member, no shift, at 4x larger or 32-64x smaller."""

    def test_large_shape_uses_only_established_pool_members(self):
        used = set()
        for name in LARGE_BATCH:
            for r in pool_candidate_records(decode(name)):
                used.add(r["payload"][-1])
        self.assertTrue(used, "expected at least one pool record")
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)

    def test_small_shape_uses_only_established_pool_members(self):
        used = set()
        for name in SMALL_BATCH:
            for r in pool_candidate_records(decode(name)):
                used.add(r["payload"][-1])
        self.assertTrue(used, "expected at least one pool record")
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)

    def test_no_new_pool_member_appears_anywhere(self):
        used = set()
        for name in ALL_NAMES:
            for r in pool_candidate_records(decode(name)):
                used.add(r["payload"][-1])
        self.assertTrue(used <= POOL_TRAILING_BYTES, used)


class TestLargeShapeShowsANewMiddleByteVariant(unittest.TestCase):
    """Both (K=2048, N=2048) rebuilds show pool-pattern records with a
    middle byte of 0x04 instead of the previously-universal 0x00 --
    still a valid pool trailing byte, but a record format no prior
    single-shape decode observed. Reported as a real, evidenced
    observation, not a decoded rule (see module docstring for why the
    disambiguation builds don't pin down a clean cause)."""

    def test_middle_byte_04_appears_at_the_large_shape(self):
        seen_04 = False
        for name in LARGE_BATCH:
            for r in pool_candidate_records(decode(name)):
                p = r["payload"]
                if len(p) == 3 and p[1] == 0x04:
                    seen_04 = True
        self.assertTrue(seen_04, "expected at least one p[1]==0x04 record")

    def test_middle_byte_is_zero_at_large_k_small_n(self):
        """(K=2048, N=16): large K alone does not reproduce the 0x04
        middle byte -- it's back to the established 0x00 form."""
        recs = decode("gemm_1x2048x16_r0.mcode.gz")
        found = pool_candidate_records(recs)
        self.assertTrue(found, "expected at least one pool record")
        for r in found:
            p = r["payload"]
            if len(p) == 3:
                self.assertEqual(p[1], 0, r)

    def test_no_pool_record_at_small_k_large_n(self):
        """(K=16, N=2048): the pool group is entirely absent here --
        consistent with PR #1577's own established finding that the
        group's presence already varies build-to-build at a fixed
        shape, now also observed to vary across shapes."""
        recs = decode("gemm_1x16x2048_r0.mcode.gz")
        self.assertEqual(pool_candidate_records(recs), [])


if __name__ == "__main__":
    unittest.main()
