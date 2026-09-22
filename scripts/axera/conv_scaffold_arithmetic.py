"""The wide-Cout Conv requantisation scaffold is a per-tile arithmetic layout,
not a bit-permutation-learned region.

`docs/axera-conv-weight-learn-wide.md`, `docs/axera-conv-weight-learn-128-and-widegap.md`,
and `docs/axera-conv-weight-learn-256to512.md` all treated the per-output-channel
`(bias, M)` requantisation block as opaque once it stopped being one contiguous span
(true for `Cout <= tile_width`, see below) -- applying `emitter.learn()` a *second* time
to guess a bit permutation from `requant_block()`'s own computed bytes into the table's
scattered "ambiguous" positions, recovering 51-80% depending on shape and, for one
256->512 shape, converging to values that were confirmed WRONG on device.

This module finds the actual layout instead of statistically inferring it. See
`docs/axera-conv-scaffold-arithmetic.md` for the full evidence; summary:

* The table is organised as repeating **per-output-channel-tile** chunks:
  `[that tile's weight codes][tile's bias, float32 x tile_width][tile's M, float32 x
  tile_width]`, tile 0 starting at byte 0 (no header, confirmed for every `K=3` shape
  checked -- `K=1` shapes may have a leading header instead of tiling at all; not
  resolved here, see the module docstring's caveats).
* **Tile width is set by a fixed per-tile weight-code byte budget, not a fixed channel
  count**: `tile_width = 36864 // (Cin * K * K)` (clamped so a tile is never wider than
  `Cout`). `36864` reproduces every checked case: `Cout<=tile_width` (64/64, 1x1
  downsample) stays one contiguous span exactly as the already-merged emitters found;
  128/128 splits into 4 tiles of 32; the formula's predicted tile/run count for 256/256
  (16 tiles of 16 channels, 2 runs/channel = 512 runs) matches
  `docs/axera-conv-weight-learn-wide.md`'s independently-reported 512 scattered runs
  exactly, without needing a new build to check.
* Blind-verified twice on fresh `Conv(128,128,3,3)` Pulsar2 builds with independent
  random weights (not the weights used to derive the formula): reading `bias`/`M`
  directly at the formula's predicted offsets, with no search, matched the real compiled
  table's per-channel float32 values -- `M` byte-exact, `bias` within the
  ~1.5e-4-absolute rounding tolerance `conv_bias_requant.py`'s own formula already
  documents.

**Not done here**: an end-to-end emission (weight-code bit-permutation + this scaffold
placement) verified against a native Pulsar2 build or on real hardware. The weight-code
half is already validated elsewhere (`conv_weight_learn.py`, `conv_learn_wide.py`); what
this module adds is purely the scaffold's placement and value formula, which callers
combine with that existing machinery via `emit_conv_table_tiled` below. Also not
resolved: `K=1` shapes' apparent header offset (the 1x1 downsample's real `block_at=9216`
does not match this module's headerless `K=3` formula's 8192 prediction for the same
`Cin*Cout`; `Cout=512, Cin=256or512` (the two shapes where prior forks found the
bit-permutation approach actually wrong, not just incomplete) were not directly rebuilt
and checked against this formula for lack of time -- only structurally corroborated via
the 256/256 run-count match above.
"""

from __future__ import annotations

import numpy as np

TILE_BYTE_BUDGET = 36864
"""Bytes of weight code per output-channel tile; see the module docstring."""


def tile_width(cin: int, k: int, cout: int, budget: int = TILE_BYTE_BUDGET) -> int:
    per_channel_bytes = cin * k * k
    return max(1, min(cout, budget // per_channel_bytes))


def tile_offsets(cout: int, cin: int, k: int, budget: int = TILE_BYTE_BUDGET):
    """``[(channel0, width, weight_start, bias_start, m_start), ...]`` for every tile,
    tile 0 at byte 0 (no header -- confirmed for K=3 shapes only, see caveats above)."""
    per_channel_bytes = cin * k * k
    tw = tile_width(cin, k, cout, budget)
    offsets = []
    byte_pos = 0
    channel = 0
    while channel < cout:
        width = min(tw, cout - channel)
        weight_start = byte_pos
        bias_start = weight_start + per_channel_bytes * width
        m_start = bias_start + 4 * width
        offsets.append((channel, width, weight_start, bias_start, m_start))
        byte_pos = m_start + 4 * width
        channel += width
    return offsets, byte_pos


def scaffold_bytes(codes, x_scale, x_zero, y_scale, y_zero, w_scale, bias):
    """Per-channel ``(bias_term, M)`` values, same formula as
    ``conv_bias_requant.requant_block_with_bias``, split by channel rather than
    concatenated as one ``[all bias][all M]`` block."""
    codes = np.asarray(codes, dtype=np.int64)
    q = codes.reshape(codes.shape[0], -1) - 2 ** (8 - 1)
    m = (np.asarray(x_scale) * np.asarray(w_scale) / np.asarray(y_scale)).astype(
        np.float32
    )
    bias_term = (
        y_zero
        - x_zero * q.sum(axis=1) * m
        + np.asarray(bias, dtype=np.float64) / np.asarray(y_scale, dtype=np.float64)
    ).astype(np.float32)
    return bias_term, m


def emit_conv_table_tiled(
    reference_table,
    origin,
    w,
    b,
    x_scale,
    x_zero,
    y_scale,
    y_zero,
    cin,
    k,
    cout,
    budget: int = TILE_BYTE_BUDGET,
):
    """A whole ``npu_params`` table for new weights, at a shape whose scaffold is
    tiled (``Cout*Cin*K*K`` large enough that ``tile_width < Cout``).

    Weight codes go through the already-learned bit-permutation ``origin`` map
    (``emitter.emit_table``), same as every other shape in this project. The
    scaffold is written directly from the formula above at each tile's computed
    offset -- no learning, no reference-table dependency for those bytes at all.
    """
    import emitter

    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes).copy()
    w_scale = emitter.weight_scales(w)
    bias_all, m_all = scaffold_bytes(
        codes, x_scale, x_zero, y_scale, y_zero, w_scale, b
    )

    offsets, _ = tile_offsets(cout, cin, k, budget)
    for channel, width, _weight_start, bias_start, m_start in offsets:
        table[bias_start : bias_start + 4 * width] = bias_all[
            channel : channel + width
        ].tobytes()
        table[m_start : m_start + 4 * width] = m_all[
            channel : channel + width
        ].tobytes()
    return table
