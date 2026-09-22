"""The ResNet18 stem `Conv`: `emitter.py`'s bit-permutation weight learner,
plus `conv_bias_requant.py`'s bias-aware requantisation block, applied to
the one real ResNet18 training-step `Conv` shape neither had reached yet.

`docs/axera-conv-weight-learn-stem.md` (PR #1769) validated the technique on
`Conv(64,64,3x3,stride=1)`; `docs/axera-conv-weight-learn-downsample.md` (PR
#1771) validated it on the two 64->128 downsample paths. Neither reached the
stem: `Conv(cin=3, cout=64, k=7x7, stride=2, pad=3, batch=16, spatial=224)` --
the only convolution in the step that touches the raw image, with the
largest kernel and the largest spatial size of any real shape in the graph.
See `docs/axera-conv-weight-learn-stem-full.md` for the full campaign and
device evidence.

Everything shape-specific this module needs (the requantisation block's
byte offset/length, and the mcode's scale/zero-point byte offsets) was
measured directly against 36+ real builds of this exact shape and is only
valid for it -- the same scoping every other module in this family uses.
"""

from __future__ import annotations

import struct

import numpy as np

BLOCK_AT = 18432
"""Byte offset of the per-channel bias/scale requantisation block in this
shape's ``npu_params`` table -- one contiguous 512-byte span (``Cout * 2
floats * 4 bytes``), the same layout family PR #1769's 64x64 shape and PR
#1771's 1x1 downsample used (unlike PR #1770's wide-channel shapes, whose
block is scattered across (at least) four per-32-channel regions)."""

BLOCK_LEN = 512

SCALE_OFFSETS = (8375, 8383, 8391, 8399)
"""Four little-endian float32 offsets in the mcode, each equal to ``y_scale``
-- found by `emitter.learn_mcode` at ``min_agreement=0.7`` across 36 builds,
confirmed by direct comparison against a clean (non-outlier) native rebuild."""

ZERO_OFFSET = 8357
"""One byte in the mcode equal to ``round(y_zero) & 0xff``."""

UNPATCHABLE_ZERO_POINTS = (127, 128)
"""``round(y_zero)`` values that produce a genuinely shifted mcode stream --
not a false refusal like PR #1771's downsample case. Two of the first three
held-out builds tried here landed on 127 (the third, used for validation,
landed on 126 and patched cleanly). `learn_mcode` reports these the same way
it reported them for the other shapes in this family; see the module
docstring and ``docs/axera-conv-weight-learn-stem-full.md`` for the direct
confirmation that 127/128 are real outliers here, not misclassifications."""


def is_patchable(y_zero: float) -> bool:
    return int(round(y_zero)) not in UNPATCHABLE_ZERO_POINTS


def emit_stem_conv(
    reference_table: np.ndarray,
    reference_mcode: bytes,
    origin,
    w: np.ndarray,
    b: np.ndarray,
    x_scale: float,
    x_zero: float,
    y_scale: float,
    y_zero: float,
) -> tuple[np.ndarray, bytes]:
    """Emit ``(new_table, new_mcode)`` for the stem shape's held-out weights.

    ``reference_table``/``reference_mcode`` come from any one of the 36
    ``k``-build reference compiles (``emitter.table_of`` / the ``*_neu``
    initializer of a reference ``.axmodel``). Raises ``ValueError`` if
    ``y_zero`` lands on a known-unpatchable value (see
    ``UNPATCHABLE_ZERO_POINTS``) -- these are real, not a coarse
    over-refusal like PR #1771's downsample case, so this module does not
    attempt to work around them.
    """
    if not is_patchable(y_zero):
        raise ValueError(
            f"y_zero={y_zero!r} rounds to an unpatchable output zero point "
            f"({UNPATCHABLE_ZERO_POINTS}); this shape's mcode shifts stream "
            "layout entirely at these values -- no byte-offset patch reaches it"
        )
    from conv_bias_requant import emit_conv_table

    new_table = emit_conv_table(
        reference_table,
        origin,
        w,
        b,
        x_scale,
        x_zero,
        y_scale,
        y_zero,
        block_at=BLOCK_AT,
        block_len=BLOCK_LEN,
    )
    new_mcode = bytearray(reference_mcode)
    new_scale_bytes = struct.pack("<f", np.float32(y_scale))
    for off in SCALE_OFFSETS:
        new_mcode[off : off + 4] = new_scale_bytes
    new_mcode[ZERO_OFFSET] = int(round(y_zero)) & 0xFF
    return new_table, bytes(new_mcode)
