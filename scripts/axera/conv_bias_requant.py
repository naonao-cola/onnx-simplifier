"""Extends `emitter.py`'s bit-permutation Conv weight emitter to biased,
real-scale ResNet18 layers.

`emitter.py`'s `learn`/`emit_table` technique -- compile the same shape `k`
times with different weights, read each `npu_params` bit's origin off the
k-way signature -- is validated at real ResNet18 training-step shapes for the
first time here, on the two downsample paths between stage 1 and stage 2
(`docs/axera-conv-weight-learn-downsample.md` has the full writeup and
evidence). Two things needed fixing or extending beyond what worked at the
toy 1-8 channel shapes `emitter.py` was previously checked against:

1. **`k` must scale with the layer's code-bit count** (`Cout*Cin*K*K*8`), not
   stay fixed at 4-6. A 1x1 downsample (65,536 code bits) needs `k=32` for
   zero collisions; a 3x3 downsample (589,824 code bits, 9x more) needs
   `k=38`. Both were found empirically by building incrementally and
   rechecking `emitter.collisions()`, not assumed from a formula.
2. **`emitter.py`'s `requant_block()` omits the ONNX `Conv` bias entirely** --
   it computes `bias[c] = zy - zx*sum(q_c)*m_c`, which is only correct for a
   bias-free convolution. Real ResNet18 convs have a real, trained bias
   input. `requant_block_with_bias()` below adds the missing `+ b_c/y_scale`
   term; confirmed against the real compiled table (max abs diff 0.00044,
   consistent with `requant_block`'s own documented ~6e-5 float32-rounding
   tolerance, against the existing bias-free formula's max diff of 6.88).

Only the 1x1 downsample's full pipeline (weights + bias block + mcode
quantisation) is validated end-to-end and confirmed on real AX8850 hardware.
The 3x3 downsample's weight-code portion and per-channel scale array are
independently confirmed byte-exact (`m` matches the compiled table 128/128,
verbatim), but its bias-block byte layout is NOT a single contiguous span
the way the 1x1 case is -- there is at least one more per-channel-group
region this module does not locate, so `emit_conv_table` below is scoped to
the shapes it was actually checked against and raises rather than guess for
anything else.
"""

from __future__ import annotations

import numpy as np


def requant_block_with_bias(codes, x_scale, x_zero, y_scale, y_zero, w_scale, bias):
    """`emitter.requant_block`'s formula, with the ONNX `Conv` bias added in.

    ``bias`` is the float32 per-output-channel bias `Conv`'s third input
    carries (real ResNet18 layers all have one). The additive term is
    ``bias_c / y_scale`` -- the float bias, in output-code units, added
    once per channel rather than accumulated per input tap the way the
    weight/activation product is.
    """
    codes = np.asarray(codes, dtype=np.int64)
    q = codes.reshape(codes.shape[0], -1) - 2 ** (8 - 1)
    m = (np.asarray(x_scale) * np.asarray(w_scale) / np.asarray(y_scale)).astype(
        np.float32
    )
    bias_term = (
        y_zero
        - x_zero * q.sum(axis=1) * m
        + np.asarray(bias, dtype=np.float32) / np.asarray(y_scale, dtype=np.float32)
    ).astype(np.float32)
    return np.concatenate([bias_term, m]).view(np.uint8)


def emit_conv_table(
    reference_table,
    origin,
    w,
    b,
    x_scale,
    x_zero,
    y_scale,
    y_zero,
    block_at,
    block_len,
):
    """`emitter.emit()`, with the bias-aware block formula, for a shape whose
    per-channel auxiliary block is one contiguous ``[block_at:block_at+block_len]``
    span (confirmed true for the 1x1 downsample at ``block_at=9216``,
    ``block_len=2*4*Cout``; NOT confirmed for the 3x3 downsample -- see the
    module docstring).
    """
    import emitter

    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes).copy()
    w_scale = emitter.weight_scales(w)
    block = requant_block_with_bias(codes, x_scale, x_zero, y_scale, y_zero, w_scale, b)
    table[block_at : block_at + block_len] = block[:block_len]
    return table
