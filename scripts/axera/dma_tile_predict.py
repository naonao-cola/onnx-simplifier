"""Predict the ``npu_params`` tile table for a tiled AX650 single-input
elementwise op's DMA load, e.g. a standalone float32 ``Relu`` or ``Sqrt``
on a 4-D NCHW-shaped tensor.

This generalizes ``transpose_tiled_params.py``'s tiling discovery
(threshold, per-tile byte budget, CPython ``set``-order table) to a
structurally different op family: an elementwise op's compiled model has
only one tile-table axis (there is no separate transposed-output offset
group), and its tiling folds the batch and channel dimensions together
rather than following ``Transpose``'s column/row-tile rule.

The rule, from ~50 Pulsar2 7.0-lite (AX650, MinMax calibration) builds of a
standalone float32 ``Relu(x[N,C,H,W])`` (``scripts/axera/dma_tile_sweep.py``,
data under ``fixtures/dma_tiles/``):

* Untiled (``npu_params`` is ten zero words) for every measured shape with
  ``N*C*H*W*4 <= 262144`` bytes. Tiled for every measured shape with
  ``N*C*H*W*4 >= 401408``; the exact boundary between the two is not
  bracketed.
* Once tiled, let ``L = N*C`` and ``row = H*W``. Find the smallest
  ``n`` in ``4, 8, 16, 32, ...`` such that a tile of ``ceil(L/n)`` rows
  fits in a 524,288-byte (512 KiB) budget: ``ceil(L/n) * row * 4 <=
  524288``. That ``n`` is the table's entry count, and its offsets are
  ``k * ceil(L/n) * row * 4`` for ``k`` in ``range(n)``, written in the
  iteration order of a Python ``set`` built by inserting them in that
  order (the same rule ``transpose_tiled_params.py`` found).
* Validated only when ``L`` divides evenly by that ``n``: 37 of 41 tiled
  builds in the sweep, covering every batch/channel combination that is a
  power of two (1, 2, 4, 8, 16 batch; any channel count a multiple of 4).
  Not decoded: batches that are not a power of two (3, 6, 12 all picked a
  different, non-doubling entry count -- 6, 12, 48 -- so batch tiling is
  not simply folded into channel tiling the way this rule assumes), a
  channel count not a multiple of 4, and a tensor whose flattened middle
  dimensions don't match its literal shape (``[16,1,64,3136]``: it tiles as
  if ``L=1024, row=3136``, not this rule's ``L=16, row=200704``).
* A two-input op (``Add``) has a structurally different table -- every one
  of 10 tested shapes mismatched this rule -- and is out of scope.

This predicts only ``npu_params``, not the MCode: the compute-engine
program (``teng2``) does not reduce to a per-shape template in the same
size range (see ``docs/axera-dma-queue.md``), so this is a characterization
tool, not an emitter. Everything outside the validated domain raises
``ValueError``.
"""

from __future__ import annotations

import struct

_GROUP_BUDGET = 524288
"""Bytes; the largest single-chunk tile group observed to stay untiled at that entry count."""

_UNTILED_MAX = 262144
"""Bytes; the largest measured shape whose npu_params was all zero (untiled)."""


def predict_words(shape: list[int], elem_bytes: int = 4) -> list[int]:
    """The 10 uint32 words of one table entry, repeated ``entry_count`` times.

    ``shape`` is the input tensor's ``[N, C, H, W]``. Raises ``ValueError`` outside the
    validated domain (see the module docstring).
    """
    if len(shape) != 4:
        raise ValueError("only decoded for rank-4 NCHW tensors")
    n_, c_, h_, w_ = shape
    if c_ == 1:
        # The one measured C=1 shape ([16,1,64,3136]) tiles as if L=1024, row=3136 --
        # folding H into the leading axis instead of this rule's L=N*C, row=H*W. Not decoded.
        raise ValueError("C=1 tensors fold a different axis into the tile; not decoded")
    if n_ > 1 and n_ & (n_ - 1):
        # Every measured batch that is not a power of two (3, 6, 12) picked a different,
        # non-doubling entry count than this rule's 4/8/16/32 sequence ever produces.
        raise ValueError(f"batch={n_} is not a power of two; entry count not decoded")
    total_bytes = n_ * c_ * h_ * w_ * elem_bytes
    if total_bytes <= _UNTILED_MAX:
        return [0] * 10
    length = n_ * c_
    row = h_ * w_
    entries = 4
    while True:
        chunk = -(-length // entries)
        if chunk * row * elem_bytes <= _GROUP_BUDGET or entries >= length:
            break
        entries *= 2
    if length % entries:
        raise ValueError(
            f"L=N*C={length} does not divide evenly by the required entry count {entries}"
        )
    stride = chunk * row * elem_bytes
    offsets = list(set(k * stride for k in range(entries)))
    return offsets * 10


def predict_params(shape: list[int], elem_bytes: int = 4) -> bytes:
    words = predict_words(shape, elem_bytes)
    return struct.pack(f"<{len(words)}I", *words)
