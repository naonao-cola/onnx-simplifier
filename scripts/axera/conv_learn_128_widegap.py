"""Two follow-ups to `docs/axera-conv-weight-learn-{stem,downsample,wide}.md`.

**Part A**: the untested `Conv(128,128,3,3,stride=1)` shape -- ResNet18's
stage-2 block, the channel count between the fully-working 64/64 shape
(`conv_weight_learn.py`) and the struggling 256/256 shape
(`conv_learn_wide.py`). Its scaffold/requantisation region turns out to be
**scattered** (128 runs of 3 or 131 bytes), not the single contiguous block
64/64 has -- the same layout family as 256/256 and 512/512, not 64/64. A
second-pass `emitter.learn()` (`requant_block_biased`'s output as the "code",
the scattered bytes as the "table") resolves **80.4%** of it, noticeably
better than `conv_learn_wide.py`'s 51% at 256/256 with the same `k`-48-ish
build count.

**Part B**: re-runs that same second-pass learn on `Conv(256,256,3,3)` with a
**fresh, larger sample** (`k=65`, up from the original 47-48) to test whether
the unresolved 37.5% was a sample-size limit or a structurally different
encoding. It resolves to **79.9%** -- almost exactly Part A's number -- so
**more builds help substantially, it is not a plateau.**

Neither shape's scaffold hits 100% here. `emit_conv_128()` below assembles
what Part A's evidence supports: the weight-code region (byte-exact), the
80.4%-resolved scaffold, and the scale literals via `patch_scales.patch_model`
-- but leaves the **zero-point unpatched**, a known, precisely diagnosed
(not merely suspected) gap: see "The zero-point gap, proven not guessed"
below and `docs/axera-conv-weight-learn-128-and-widegap.md`.
"""

from __future__ import annotations

import numpy as np


def emit_conv_128(
    reference_table,
    origin,
    scaffold_origin,
    outer_ambiguous,
    w,
    x_scale,
    x_zero,
    y_scale,
    y_zero,
    bias,
):
    """Patch a `Conv(128,128,3,3)` reference table for new weights.

    `origin`/`scaffold_origin`/`outer_ambiguous` are `conv128_map.npz` and
    `conv128_scaffold_map.npz`'s contents (see the module docstring and
    `docs/axera-conv-weight-learn-128-and-widegap.md`). Does NOT patch the
    output zero-point in the mcode -- callers must still account for the
    known, quantified residual (`(true_y_zero - reference_y_zero) * y_scale`,
    additive on the output) documented there, or supply matching zero points.
    """
    import conv_weight_learn as cwl
    import emitter

    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes)
    w_scale = emitter.weight_scales(w)
    scaffold_code_bytes = cwl.requant_block_biased(
        codes, x_scale, x_zero, y_scale, y_zero, w_scale, bias
    )

    ref_bits = np.unpackbits(reference_table.reshape(1, -1), axis=1, bitorder="little")[
        0
    ]
    ref_scaffold_bytes = np.packbits(ref_bits[outer_ambiguous], bitorder="little")
    new_scaffold_bytes = emitter.emit_table(
        ref_scaffold_bytes, scaffold_origin, [scaffold_code_bytes]
    )
    new_scaffold_bits = np.unpackbits(new_scaffold_bytes, bitorder="little")[
        : len(outer_ambiguous)
    ]

    final_bits = np.unpackbits(table.reshape(1, -1), axis=1, bitorder="little")[0]
    final_bits[outer_ambiguous] = new_scaffold_bits
    return np.packbits(final_bits.reshape(1, -1), axis=1, bitorder="little")[0]
