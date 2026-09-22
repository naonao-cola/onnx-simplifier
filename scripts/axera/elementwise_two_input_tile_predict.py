"""Predict the ``npu_params`` table for the remaining two-input elementwise ops
(``Sub``, ``Div``, ``Mul``), extending ``add_tile_predict.py``'s decode of
``Add`` to its siblings. The three ops turn out to split two ways, not one:

* **``Sub`` reuses ``Add``'s formula unchanged.** Same quantization header
  (``round(x_scale/y_scale*32768)``/``round(z_scale/y_scale*32768)`` as
  little-endian Q15 ``uint16``s, deduped to one word when equal) followed by
  the same single-input tile cycle repeated 15 times. Confirmed byte-exact
  against 6 builds (untiled symmetric/asymmetric, tiled at 4/8/32 entries,
  and a held-out shape/calibration-range pair not used to derive anything
  here), reusing ``add_tile_predict.py``'s own validated body logic directly
  rather than re-deriving it.
* **``Mul`` and ``Div`` have no header at all.** Their ``npu_params`` is
  exactly the tile cycle repeated 15 times, with none of ``Add``/``Sub``'s
  leading scale-ratio bytes -- confirmed byte-exact the same way, across the
  same 6 shapes each, including a held-out case for each op. This was found
  by noticing the real table length never includes the 2 or 4 header bytes
  ``Add``'s formula would predict, not assumed from symmetry with ``Add``.
  Plausible reading: unlike ``Add``/``Sub``, neither ``Mul`` nor ``Div`` can
  be computed by first rescaling one input to the other's units and then a
  single elementwise op on raw quantized codes -- so there may be no shared
  "ratio" a header would need to carry. This is offered as one guess about
  *why*, not a confirmed mechanism.

Both findings share ``Add``'s own boundary: only checked below the
``N*C = 2048`` split threshold ``add_tile_predict.py`` found (where the table
becomes an undecoded three-way read/write mix for ``Add``); untested for
``Sub``/``Mul``/``Div`` at or above it.

As with every other elementwise decode in this project, this predicts only
``npu_params``, not the MCode -- the compute-engine segment (``teng2``) does
not reduce to a per-shape template (``docs/axera-dma-queue.md``,
``docs/axera-teng2-calibration-isolation.md``), so this is a characterization
tool, not a full emitter.
"""

from __future__ import annotations

import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from add_tile_predict import predict_params as _add_predict_params  # noqa: E402
from dma_tile_predict import predict_words as _relu_predict_words  # noqa: E402

_SPLIT_THRESHOLD = 2048
"""N*C at or above which Add's table splits into an undecoded read/write mix;
inherited as the boundary of the validated domain here too, though it was not
independently reconfirmed for Sub/Mul/Div."""

_REPS = 15
"""Repeat count of the tile cycle (matches Add/Sub; Relu's own table uses 10)."""

_HAS_HEADER = {"Sub": True, "Mul": False, "Div": False}


def _body_bytes(shape: list[int], elem_bytes: int) -> bytes:
    relu_words = _relu_predict_words(shape, elem_bytes)
    entries = len(relu_words) // 10
    cycle = relu_words[:entries]
    body = cycle * _REPS
    return struct.pack(f"<{len(body)}I", *body)


def predict_params(
    op: str,
    shape: list[int],
    x_scale: float,
    z_scale: float,
    y_scale: float,
    elem_bytes: int = 4,
) -> bytes:
    """The ``npu_params`` bytes for ``op(x, z)`` at ``shape = [N, C, H, W]``.

    ``op`` is one of ``"Sub"``, ``"Mul"``, ``"Div"``. ``x_scale``/``z_scale``/``y_scale``
    are the compiled model's per-tensor calibration scales (from
    ``quant_axmodel.json``'s ``tensor_configs``/``values``, matching
    ``add_tile_predict.predict_params``'s convention) -- required for ``Sub``'s header,
    accepted but unused for ``Mul``/``Div`` since they carry no header. Raises
    ``ValueError`` outside the validated domain or for an unknown ``op``.
    """
    if op not in _HAS_HEADER:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_HAS_HEADER)}")
    if len(shape) != 4:
        raise ValueError("only decoded for rank-4 NCHW tensors")
    n_, c_ = shape[0], shape[1]
    if n_ * c_ >= _SPLIT_THRESHOLD:
        raise ValueError(
            f"N*C={n_ * c_} >= {_SPLIT_THRESHOLD}: the table splits into an "
            "undecoded read/write mix at this size (inherited from Add; not "
            "independently confirmed for this op)"
        )
    if op == "Sub":
        # Identical to Add's own formula; delegate rather than duplicate it.
        return _add_predict_params(shape, x_scale, z_scale, y_scale, elem_bytes)
    return _body_bytes(shape, elem_bytes)


__all__ = ["predict_params"]
