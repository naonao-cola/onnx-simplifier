"""Predict the ``npu_params`` table for a tiled AX650 two-input elementwise op's
DMA load, e.g. a standalone float32 ``Add(x[N,C,H,W], z[N,C,H,W])``.

This extends ``dma_tile_predict.py``'s single-input tile table to the
two-input case. ``docs/axera-dma-queue.md`` found Add's table "structurally
different" and left it out of scope; this module decodes the part of it that
is a clean, data-dependent formula.

Two things distinguish Add's table from Relu's:

* **A quantization header, not a data-independent constant.** The first 2 or
  4 bytes hold ``round(x_scale/y_scale * 32768)`` and
  ``round(z_scale/y_scale * 32768)`` as little-endian ``uint16`` Q15
  fixed-point values (``x_scale``/``z_scale``/``y_scale`` are the compiled
  model's per-tensor calibration scales for the two inputs and the output).
  If the two rounded values are equal, only one is written (2-byte header);
  otherwise both are written, x first (4-byte header). Confirmed exact
  (``round(ratio * 32768)`` to the bit) against 6 builds spanning symmetric
  scales (the two inputs calibrated over the same range, giving a dedup) and
  one with deliberately asymmetric, non-overlapping ranges (giving two
  distinct words) -- see ``scripts/axera/add_tile_sweep.py``.
* **Fifteen repetitions of the tile cycle, not ten.** Below a batch*channel
  threshold (``N*C < 2048`` in every build checked), the offset table after
  the header is exactly the same ``[N,C,H,W]``-tile cycle
  ``dma_tile_predict.predict_words`` computes for the single-input case,
  repeated 15 times instead of Relu's 10 -- read and write apparently share
  one tiling in this regime. At ``N*C = 2048`` (two shapes checked: a
  ``[16,128,28,28]`` and a differently-shaped ``[1,2048,28,28]`` of the same
  total size, both deterministic across a rebuild) the table becomes a
  three-way mix of the plain (write) cycle and a finer, two-groups read
  cycle in an irregular repeat pattern that was not decoded -- this module
  raises ``ValueError`` for ``N*C >= 2048`` rather than guess at it.

This predicts only ``npu_params``, not the MCode: as with ``Relu`` and
``Transpose``, the compute-engine segment does not reduce to a per-shape
template in the same size range (see ``docs/axera-teng2-add-two-input.md``),
so this is a characterization tool, not a full emitter.
"""

from __future__ import annotations

import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dma_tile_predict import predict_words as _relu_predict_words  # noqa: E402

_SPLIT_THRESHOLD = 2048
"""N*C at or above which the table splits into an undecoded read/write mix."""

_REPS = 15
"""Repeat count of the tile cycle in Add's table (Relu's own table uses 10)."""


def _header_words(x_scale: float, z_scale: float, y_scale: float) -> list[int]:
    qx = round(x_scale / y_scale * 32768) & 0xFFFF
    qz = round(z_scale / y_scale * 32768) & 0xFFFF
    return [qx] if qx == qz else [qx, qz]


def predict_params(
    shape: list[int],
    x_scale: float,
    z_scale: float,
    y_scale: float,
    elem_bytes: int = 4,
) -> bytes:
    """The ``npu_params`` bytes for ``Add(x, z)`` at ``shape = [N, C, H, W]``.

    ``x_scale``, ``z_scale``, and ``y_scale`` are the compiled model's per-tensor
    calibration scales (from ``quant_axmodel.json``'s ``tensor_configs``/``values``,
    or any source of the same numbers -- this is a data-dependent prediction, unlike
    the single-input tile table). Raises ``ValueError`` outside the validated domain.
    """
    if len(shape) != 4:
        raise ValueError("only decoded for rank-4 NCHW tensors")
    n_, c_ = shape[0], shape[1]
    if n_ * c_ >= _SPLIT_THRESHOLD:
        raise ValueError(
            f"N*C={n_ * c_} >= {_SPLIT_THRESHOLD}: the table splits into an "
            "undecoded read/write mix at this size"
        )
    header = _header_words(x_scale, z_scale, y_scale)
    relu_words = _relu_predict_words(shape, elem_bytes)
    entries = len(relu_words) // 10
    cycle = relu_words[:entries]
    body = cycle * _REPS
    return struct.pack(f"<{len(header)}H", *header) + struct.pack(
        f"<{len(body)}I", *body
    )
