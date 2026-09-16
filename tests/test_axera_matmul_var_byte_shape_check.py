"""Checking an untested assumption in the whole `var`-byte thread:
does batched MatMul's short-form `A` quad ever change RECORD SHAPE
(not just its `var` value) at extreme `batch`/`M`/`K` -- the way
Gemm's own `K*M-1` field turned out to (`tests/test_axera_gemm_km1_k_regime_check.py`,
PR #1545, going from a 3-byte value form to a valueless 2-byte form to
a widened 4-byte form as `K` grows)?

Every prior investigation of MatMul's `var` byte (`tests/test_axera_matmul_batched_var_byte.py`
and its many follow-ups mapping `var`'s period-32/mod-4 structure)
only ever tracked the byte's *value*, always implicitly assuming the
surrounding frame -- `<3 bytes, low(1/A_scale)> 82 <var> <02 or 83>`
at stride 6 -- stays fixed. That assumption was never tested at shape
values outside the roughly 1-48 range this thread's `batch`/`M`/`K`
sweeps stayed within.

## Result: the frame is genuinely fixed, at least through batch/M/K=64-128

Built `batch=64`, `M=64`, `K=64`, and `K=128` (each varied
independently against the established `batch=2,M=4,K=8,N=8` baseline,
`pulsar2_docker.build()`, same weight/calibration convention as
`tests/test_axera_matmul_batched_site_a.py`) -- shape values 2-16x
larger than anything tested in this thread's `var`-byte work so far.
At every one of these four points, the record's exact byte frame is
identical to the baseline's own: `<3 bytes> 82 <var> 02` for the first
three of the four stride-6 copies, `<3 bytes> 82 <var> 83` for the
fourth (the "last copy differs" pattern `tests/test_axera_matmul_batched_site_a.py`,
PR #1502, already established) -- only the `var` byte's own value
changes (`0x62` at baseline, `0x30` at every one of the four extreme
points tested here), never the surrounding frame's tag bytes, stride,
or byte width.

**Confirmed above the noise floor.** An independent rebuild of `K=64`
(identical shape/calibration, a genuinely separate compile) reproduces
the identical frame and `var` value (`0x30`) at the identical offset
(2138); the two builds' whole streams differ at only 15 bytes, offsets
791-807 -- this project's ordinary noise-floor scale, not the
`A_offset`/`B_offset` table-order coin flip's own signature (900+
bytes, starting at offset ~204, per `tests/test_axera_matmul_offset_table_coinflip.py`).

## What this settles, and what it does not

This is a real, useful negative result: MatMul's `var`-byte frame does
NOT undergo a Gemm-style shape-triggered structural change within the
range tested here, unlike Gemm's own analogous field. It does not
prove the frame is fixed at every possible shape -- only that it holds
through `batch`/`M`/`K` = 64 (and `K` = 128), well past this thread's
previous ceiling. Whether even larger shapes (hundreds/thousands, or
values the real device's tiling limits would actually reject) ever
trigger a shape change remains untested; also untested here is whether
`var`'s own *value* at these new extreme points fits any of the
already-decoded period-32 (`K`)/mod-4 (`M`) patterns, since that was
not this file's question -- all four new points happen to land on the
already-seen value `0x30`, which this file does not attempt to explain.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def find_short_form_var_run(data, a_scale):
    """Locate the short-form A quad (`<3 bytes> 82 <var> <02|83>` x4,
    stride 6) and return the 4 start offsets, or [] if not found in
    this exact shape."""
    short = struct.pack("<f", 1.0 / a_scale)[:3]
    found = [i for i in range(len(data) - 5) if data[i : i + 3] == short]
    for start in found:
        run = [start]
        i = start + 6
        while i in found:
            run.append(i)
            i += 6
        if len(run) == 4:
            return run
    return []


class TestFrameStaysFixedAtExtremeShapes(unittest.TestCase):
    # fixture: (a_scale, expected var)
    CASES = {
        "matmul_var_byte_rebuild0.mcode.gz": (0.007840047590434551, 0x62),
        "matmul_var_shape_check_batch64.mcode.gz": (0.007842687889933586, 0x30),
        "matmul_var_shape_check_m64.mcode.gz": (0.007842687889933586, 0x30),
        "matmul_var_shape_check_k64.mcode.gz": (0.007842687889933586, 0x30),
        "matmul_var_shape_check_k128.mcode.gz": (0.007842687889933586, 0x30),
    }

    def test_frame_and_stride_are_identical_at_every_shape(self):
        for fname, (a_scale, expected_var) in self.CASES.items():
            data = load(fname)
            run = find_short_form_var_run(data, a_scale)
            self.assertEqual(len(run), 4, f"{fname}: expected 4-copy stride-6 run")
            strides = {b - a for a, b in zip(run, run[1:])}
            self.assertEqual(strides, {6}, f"{fname}: stride")
            for j in run[:-1]:
                self.assertEqual(data[j + 3], 0x82, f"{fname}@{j}: tag0")
                self.assertEqual(data[j + 5], 0x02, f"{fname}@{j}: tag2")
            last = run[-1]
            self.assertEqual(data[last + 3], 0x82, f"{fname}@{last}: tag0 (last)")
            self.assertEqual(data[last + 5], 0x83, f"{fname}@{last}: tag2 (last)")
            for j in run:
                self.assertEqual(data[j + 4], expected_var, f"{fname}@{j}: var")


class TestK64FrameSurvivesAnIndependentRebuild(unittest.TestCase):
    def test_rebuild_reproduces_identical_frame_at_identical_offset(self):
        a_scale = 0.007842687889933586
        orig = load("matmul_var_shape_check_k64.mcode.gz")
        reb = load("matmul_var_shape_check_k64_rebuild.mcode.gz")
        run_orig = find_short_form_var_run(orig, a_scale)
        run_reb = find_short_form_var_run(reb, a_scale)
        self.assertEqual(run_orig, run_reb, "identical offsets across rebuild")
        self.assertEqual(
            orig[run_orig[0] : run_orig[0] + 30],
            reb[run_reb[0] : run_reb[0] + 30],
            "identical frame content across rebuild",
        )

    def test_rebuild_noise_is_ordinary_scale_not_the_table_order_coinflip(self):
        orig = load("matmul_var_shape_check_k64.mcode.gz")
        reb = load("matmul_var_shape_check_k64_rebuild.mcode.gz")
        self.assertEqual(len(orig), len(reb))
        diffs = [i for i in range(len(orig)) if orig[i] != reb[i]]
        self.assertLess(
            len(diffs), 50, "expected ordinary noise-floor scale, not the coin flip"
        )
        self.assertGreater(min(diffs), 300, "not the coin flip's own offset-204 start")


if __name__ == "__main__":
    unittest.main()
