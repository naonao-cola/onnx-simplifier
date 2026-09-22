"""`emitter.py`'s bit-permutation Conv weight emitter, validated at the
ResNet18 stage-2-to-3 downsample pair -- `Conv(x[16,128,28,28], w[256,128,1,1],
stride 2)` and `Conv(x[16,128,28,28], w[256,128,3,3], stride 2)`, both with a
real trained bias. See `docs/axera-conv-weight-learn-128to256.md` for the full
evidence.

Both shapes' **weight-code region** is byte-exact via `emitter.py`'s standard
`learn`/`emit_table` bit-permutation (k=46 for the 1x1 shape's 262,144 code
bits, k=54 for the 3x3 shape's 2,359,296 -- both zero collisions).

The two shapes differ in their **scaffold** (the per-output-channel
requantisation `(bias, M)` block) placement, matching the pattern
`docs/axera-conv-scaffold-arithmetic.md` and `docs/axera-conv-weight-learn-
128-and-widegap.md` already found for other channel counts:

* **1x1** (`Cin*K*K = 128`): one contiguous block for all 256 channels,
  `[bias(256 x f32)][M(256 x f32)]`, at a fixed offset (`36864`) found
  empirically from the first-pass `learn()`'s own ambiguous-bit range -- not
  derived from `conv_scaffold_arithmetic.py`'s `K=1` formula, which is
  explicitly unresolved for this shape (see below).
* **3x3** (`Cin*K*K = 1152`): tiled, `tile_width = 36864 // 1152 = 32`
  (8 tiles of 32 output channels), each tile's own weight codes immediately
  followed by its own `[bias(32 x f32)][M(32 x f32)]` -- confirmed both by
  direct byte-search for the computed `M` values (found exactly at
  `tile_i * 37120 + 36992` for every tile) and by the first-pass `learn()`'s
  ambiguous bits landing exactly inside those spans and nowhere else. The
  table has a **480-byte trailing region** past the last tile
  (`8*37120=296960` vs. the real `297440`-byte table) that carries zero
  ambiguous bits across all 54+ builds -- constant, safe to leave at the
  reference's own value, not part of any tile.

## A real bug in `conv_scaffold_arithmetic.emit_conv_table_tiled` (master, unfixed)

That function (merged to `origin/master`) assigns a raw Python `bytes` object
directly to a `numpy.uint8` array slice:
`table[a:b] = some_float32_array[...].tobytes()`. This does **not** perform a
byte copy -- NumPy treats the assigned `bytes` as a single scalar value to
cast into the destination dtype and raises
`ValueError: invalid literal for int() with base 10: b'...'` for any
non-trivial length. Confirmed by a minimal repro (`np.zeros(N, dtype=np.uint8)
[a:b] = np.arange(w, dtype=np.float32).tobytes()` fails identically). The fix
is `np.frombuffer(...tobytes(), dtype=np.uint8)` in place of the bare
`.tobytes()`. This was never previously exercised against a real build or
device -- `docs/axera-conv-scaffold-arithmetic.md` itself says so
("No end-to-end emission was device-checked") -- which is why the bug
survived being merged. `emit_p3_table` below reimplements the (corrected)
tiling logic directly rather than depending on the broken function, matching
this project's established pattern of not editing another module in place;
whoever owns `conv_scaffold_arithmetic.py` should apply the one-line fix.
"""

from __future__ import annotations

import numpy as np

P1_BLOCK_AT = 36864
P1_BLOCK_LEN = 2 * 4 * 256  # [256 x bias f32][256 x M f32]

P3_TILE_WEIGHT_BYTES = 36864  # Cin(128)*K*K(9)*tile_width(32)
P3_TILE_WIDTH = 32
P3_TILE_TOTAL_BYTES = 37120  # weight(36864) + bias(128) + M(128)
P3_TABLE_LEN = 297440  # 8 tiles * 37120 + 480-byte constant trailer


def _requant(codes, x_scale, x_zero, y_scale, y_zero, w_scale, bias):
    """Per-channel `(bias_term, M)`, the standard quantised-Conv requantisation
    formula (`conv_bias_requant.requant_block_with_bias`'s math, split by
    channel instead of concatenated)."""
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


def emit_p1_table(reference_table, origin, w, b, x_scale, x_zero, y_scale, y_zero):
    """`Conv(x[16,128,28,28], w[256,128,1,1], stride 2)`: weight-code
    bit-permutation plus the one contiguous scaffold block."""
    import emitter

    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes).copy()
    w_scale = emitter.weight_scales(w)
    bias_term, m = _requant(codes, x_scale, x_zero, y_scale, y_zero, w_scale, b)
    block = np.concatenate([bias_term, m]).view(np.uint8)
    table[P1_BLOCK_AT : P1_BLOCK_AT + P1_BLOCK_LEN] = block[:P1_BLOCK_LEN]
    return table


def emit_p3_table(reference_table, origin, w, b, x_scale, x_zero, y_scale, y_zero):
    """`Conv(x[16,128,28,28], w[256,128,3,3], stride 2)`: weight-code
    bit-permutation plus the 8-tile scaffold, with the `np.frombuffer` fix
    `conv_scaffold_arithmetic.emit_conv_table_tiled` needs (see module
    docstring)."""
    import emitter

    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes).copy()
    w_scale = emitter.weight_scales(w)
    bias_all, m_all = _requant(codes, x_scale, x_zero, y_scale, y_zero, w_scale, b)

    for tile in range(8):
        channel = tile * P3_TILE_WIDTH
        tile_start = tile * P3_TILE_TOTAL_BYTES
        bias_start = tile_start + P3_TILE_WEIGHT_BYTES
        m_start = bias_start + 4 * P3_TILE_WIDTH
        table[bias_start : bias_start + 4 * P3_TILE_WIDTH] = np.frombuffer(
            bias_all[channel : channel + P3_TILE_WIDTH].tobytes(), dtype=np.uint8
        )
        table[m_start : m_start + 4 * P3_TILE_WIDTH] = np.frombuffer(
            m_all[channel : channel + P3_TILE_WIDTH].tobytes(), dtype=np.uint8
        )
    return table
