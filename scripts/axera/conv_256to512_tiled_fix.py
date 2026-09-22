"""Fixes the two real ResNet18 Conv shapes `docs/axera-conv-weight-learn-256to512.md`
(PR #1777) found bit-permutation-learning gets wrong, using `conv_scaffold_arithmetic.py`'s
(PR #1784) arithmetic scaffold formula -- extended here with two real layout facts that
formula's own validated case (`Conv(128,128,3,3)`) did not need:

* **K=1 has its own tile geometry**, not the K=3 byte-budget formula. `Conv(x[16,256,14,14],
  w[512,256,1,1])`'s real tile width is 128 output channels (an apparent hard cap, matching
  the value already independently found as `Conv(64,64,128,1,1)`'s single-tile width in
  `docs/axera-conv-weight-learn-downsample.md`), and the scaffold for each 128-wide tile
  starts at `weight_bytes_per_tile * 9 // 8` from that tile's own weight-code start --
  confirmed exactly (down to the byte) at both the already-working `Cin=64` shape
  (`9216 == 8192*9//8`) and this `Cin=256` shape (`36864 == 32768*9//8`), across 3
  independent builds each.
* **Cin > 128 duplicates the scaffold.** `Conv(x[16,256,14,14], w[512,256,3,3])`'s
  `Cin=256` splits into two 128-wide input-channel sub-tiles (matching the same 128 cap the
  K=1 case shows), and EACH sub-tile carries its own copy of the *same* per-output-channel
  `(bias, M)` block -- the real compiled table holds it twice, 37,120 bytes apart (two
  `128*9*32 + 256`-byte half-tiles per real 32-output-channel tile), not once. Missing this
  is exactly why the earlier device fault happened: `origin`/`emit_table` correctly never
  touches this region (both copies are `CONST` in the bit-permutation map, since the first-
  pass learner works on whichever ONE copy the reference happened to also carry the
  reference's own value in -- see the module's own diagnostic search), so a scaffold patch
  that only writes one copy silently leaves the other holding stale reference-build values.

Verified (see `docs/axera-conv-256to512-tiled-fix.md`):

* Both shapes' scaffold placement/values, directly against real compiled tables (3 builds
  each for 1x1, 2 for 3x3): `M` byte-exact, `bias` within the ~1.5e-4-absolute tolerance
  this project's requantisation formula already documents everywhere else.
* A held-out weight set emitted via this module, diffed against a real native rebuild of
  the same weights: 0 `npu_params` bytes differ outside the scaffold region for either
  shape (weight-code region is untouched and was already known byte-exact).
* On the AX8850 (`axcl-vm`): `Conv(512,256,3,3)`'s previous 17-bad-channel, up-to-11.8-error
  failure is gone -- 0 channels over 1.0 error, max error 0.15 (vs the broken build's
  11.79), after also patching `scripts/axera/patch_scales.py`'s mcode scale literals.
  `Conv(512,256,1,1)`'s scaffold is equally confirmed correct (0 `npu_params` bytes wrong
  outside expected rounding), but `patch_scales.py`'s mcode patch introduces roughly 1,500
  bytes of genuine scheduling divergence against a real rebuild at this shape (not just
  scale literals) and the resulting artifact faults the device runtime -- a real, separately
  diagnosed limitation in a DIFFERENT, pre-existing tool, not in this module's scaffold fix
  (confirmed by running the scaffold-only, scale-UNPATCHED emission, which runs cleanly and
  gives a bounded, expected numeric error from the stale reference y_scale).
"""

from __future__ import annotations

import conv_scaffold_arithmetic as csa
import emitter
import numpy as np

TILE_PARAMS = {
    # tw: output channels per tile. first_bias: byte offset of tile 0's bias block,
    # relative to the table start. stride: byte offset from one tile's bias block to
    # the next. dup_offset: for shapes whose Cin > 128, the byte offset from a tile's
    # primary (bias, M) copy to its duplicate (see module docstring); None otherwise.
    "c1x1": dict(tw=128, first_bias=36864, stride=37888, dup_offset=None),
    "c3x3": dict(tw=32, first_bias=36864, stride=74240, dup_offset=37120),
}
COUT = 512


def emit_holdout_table(prefix: str, reference_table, origin, w, b, x_s, x_z, y_s, y_z):
    """A whole `npu_params` table for a held-out `(w, b)` at `prefix` in `TILE_PARAMS`.

    Weight codes go through the already-learned bit-permutation `origin` (unaffected by
    this module -- already validated byte-exact everywhere in this series). The scaffold
    is written directly from `conv_scaffold_arithmetic.scaffold_bytes` at this module's
    measured tile offsets, writing every duplicate copy a shape needs.
    """
    p = TILE_PARAMS[prefix]
    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes).copy()
    w_scale = emitter.weight_scales(w)
    bias_all, m_all = csa.scaffold_bytes(codes, x_s, x_z, y_s, y_z, w_scale, b)

    tw, first_bias, stride, dup_offset = (
        p["tw"],
        p["first_bias"],
        p["stride"],
        p["dup_offset"],
    )
    copies_offsets = [0] if dup_offset is None else [0, dup_offset]
    for t in range(COUT // tw):
        channel = t * tw
        bias_start = first_bias + t * stride
        for off in copies_offsets:
            copy_bias = bias_start + off
            copy_m = copy_bias + 4 * tw
            table[copy_bias : copy_bias + 4 * tw] = np.frombuffer(
                bias_all[channel : channel + tw].tobytes(), dtype=np.uint8
            )
            table[copy_m : copy_m + 4 * tw] = np.frombuffer(
                m_all[channel : channel + tw].tobytes(), dtype=np.uint8
            )
    return table
