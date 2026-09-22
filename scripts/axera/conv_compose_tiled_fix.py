"""Follow-through on `docs/axera-conv-compose-real.md` (PR #1783): does the
tiled scaffold layout from `conv_scaffold_arithmetic.py` (PR #1784) explain
and fix the wrong device output that PR found when patching new weights into
the terminal `Conv(64,64,3,3)` position of a real composed chain?

**Answer: no to both halves of that question.**

* The shape does not tile. `conv_scaffold_arithmetic.tile_width(cin=64, k=3,
  cout=64) == 64 == cout`, i.e. one tile covering the whole channel range --
  by that module's own docstring, this is exactly the case that "stays one
  contiguous span exactly as the already-merged emitters found". Applying
  the tiled formula to this shape is mathematically identical to the
  contiguous formula PR #1783 already used and found wrong. Ruled out
  without a new build.
* A real, independent bug WAS found and fixed along the way: PR #1783's
  patch computed the scaffold's per-channel `M` using the *original,
  already-baked-in* weight's scale (read from the composed build's own
  `quant_axmodel.json`, which describes whatever weight Pulsar2 originally
  compiled at that position) instead of `emitter.weight_scales()` applied to
  the *new* weight being patched in. `wrong_m()` below reproduces PR #1783's
  exact (buggy) values bit-for-bit from the composed build's json alone, no
  device access needed -- see `test_conv_compose_tiled_fix.py`.
* **Fixing that bug does not fix the device output.** With the weight-code
  region confirmed byte-exact (0/36864 mismatches, same method as PR
  #1783's Finding 2) and the scaffold's `M` now confirmed to match
  `x_scale * emitter.weight_scales(w_new) / y_scale` exactly (not the buggy
  value), a fresh device run gave `max_err=5.3250 mean_err=0.788294` --
  statistically indistinguishable from PR #1783's own buggy-scaffold
  numbers (`max_err 5.29-5.46, mean_err 0.787-0.788`). The w_scale-source
  bug is real and independently worth fixing, but it was not the cause of
  the wrong output. **The actual cause remains open.**

What this leaves as the most plausible remaining explanation, unconfirmed:
PR #1783's own original theory -- that `requant_block_biased`'s formula,
derived and validated entirely against a standalone graph-*boundary* input's
scale, may not apply as-is once the Conv's real input is an internal, fused,
never-materialized tensor (here, the `Relu` output `r1`). Checking whether
`r1`'s json-reported scale is an *alias* of some other tensor's config (the
"OVERLAPPED"/"ACTIVATED" `dominator` mechanism `quant_axmodel.json` uses)
came up clean -- `r1` and the `a1` name used inside `Conv`'s own tensor_config
both resolve to the identical hash and value everywhere they appear, so a
config-aliasing bug is ruled out too. What specifically differs about an
internal tensor's *representation* (beyond having the "right" scale/zero
value) that the compute engine's epilogue actually uses is not decoded here.

Reproduction, no Docker/device required:

    conv_compose_tiled_fix.py check    # tile_width + wrong-m + fixed-m + code-exactness

Device numbers above are recorded as text, the same way PR #1783 recorded
its own (not automated -- outside CI's reach).
"""

from __future__ import annotations

import gzip
import io
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import conv_weight_learn as cwl  # noqa: E402
import emitter  # noqa: E402
from conv_compose_real_check import (  # noqa: E402
    TERMINAL_OFFSET,
    code_byte_mask,
    standalone_reference,
)
from conv_scaffold_arithmetic import tile_width  # noqa: E402

_FIXTURES = os.path.join(_HERE, "fixtures", "conv_compose_tiled_fix")

# The real terminal Conv's shape in PR #1783's composed chain.
CIN, COUT, K = 64, 64, 3

# Real scale/zero values read from the composed build's own
# out/quant/quant_axmodel.json (docs/axera-conv-compose-tiled-fix.md), for
# the terminal Conv's real (internal) input `r1` and its output `y`.
X_SCALE, X_ZERO = 0.018685635179281235, 0.0
Y_SCALE, Y_ZERO = 0.03341430798172951, 128.0

# The terminal Conv's *original* per-channel weight scale, as
# quant_axmodel.json reported it for the weight already baked into the
# composed reference build -- what PR #1783's patch used by mistake.
ORIGINAL_W_SCALE = np.array(
    [
        0.0016210776520892978,
        0.0015113104600459337,
        0.0014921077527105808,
        0.0013980214716866612,
        0.0015354302013292909,
        0.0016084356466308236,
        0.0014586434699594975,
        0.0013617901131510735,
    ],
    dtype=np.float32,
)


def _load_gz_npy(name: str) -> np.ndarray:
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return np.load(io.BytesIO(f.read()))


def held_out_weights() -> tuple[np.ndarray, np.ndarray]:
    """The genuinely-distinct-seed weight/bias set PR #1783 settled on for its
    device test (its own first attempt accidentally collided with the
    published `holdout` fixture; this is the corrected second one)."""
    return _load_gz_npy("w_new.npy.gz"), _load_gz_npy("b_new.npy.gz")


def is_tiled() -> bool:
    """Whether the terminal Conv's shape is wide enough for PR #1784's tiled
    scaffold formula to differ from the contiguous formula PR #1783 used."""
    return tile_width(CIN, K, COUT) < COUT


def wrong_m(w_scale: np.ndarray = ORIGINAL_W_SCALE) -> np.ndarray:
    """Reproduces PR #1783's actual (buggy) scaffold M values: the formula
    applied to the *original* baked-in weight's scale, not the new weight's."""
    return (X_SCALE * w_scale / Y_SCALE).astype(np.float32)


def fixed_table(w_new: np.ndarray, b_new: np.ndarray) -> np.ndarray:
    """The corrected table for the terminal position: weight codes via the
    learned bit-permutation map, scaffold via `emitter.weight_scales(w_new)`
    -- not any scale read from a build's own json."""
    standalone_table, origin = standalone_reference()
    return cwl.emit_biased(
        standalone_table,
        origin,
        w_new,
        b_new,
        X_SCALE,
        X_ZERO,
        Y_SCALE,
        Y_ZERO,
        block_at=CIN * COUT * K * K,
    )


def code_mismatches(table: np.ndarray, w: np.ndarray) -> int:
    """Weight-code byte mismatches between `table` and the prediction for `w`,
    same method as PR #1783's Finding 2 (`find_weight_code_offset`)."""
    standalone_table, origin = standalone_reference()
    predicted = emitter.emit_table(standalone_table, origin, emitter.codes_of(w))
    mask = code_byte_mask(len(standalone_table), origin)
    return int(np.count_nonzero((table != predicted) & mask))


def check() -> int:
    tiled = is_tiled()
    print(f"tile_width({CIN},{K},{COUT}) tiled: {tiled} (expected False)")

    w_new, b_new = held_out_weights()
    correct_w_scale = emitter.weight_scales(w_new)
    m_wrong = wrong_m()
    m_correct = (X_SCALE * correct_w_scale[: len(m_wrong)] / Y_SCALE).astype(np.float32)
    same = np.allclose(m_wrong, m_correct)
    print(f"buggy M == corrected M (should differ): {same}")

    table = fixed_table(w_new, b_new)
    m_from_table = table[
        CIN * COUT * K * K + 4 * COUT : CIN * COUT * K * K + 8 * COUT
    ].view(np.float32)
    expected_m = (X_SCALE * correct_w_scale / Y_SCALE).astype(np.float32)
    m_ok = np.allclose(m_from_table, expected_m, rtol=1e-5)
    print(f"fixed table's M matches x_scale*weight_scales(w_new)/y_scale: {m_ok}")

    mism = code_mismatches(table[TERMINAL_OFFSET - TERMINAL_OFFSET :], w_new)
    print(f"weight-code mismatches: {mism} / 36864 (expected 0)")

    return 0 if (not tiled and not same and m_ok and mism == 0) else 1


if __name__ == "__main__":
    sys.exit(check())
