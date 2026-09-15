#!/usr/bin/env python3
"""Emit NPU mcode for tinygrad-traced graphs -- first op: unary negation.

Pipeline: a tinygrad ``Tensor`` graph is traced to its UOp pattern (unary
negation lowers to ``MUL(x, -1.0)``), matched, and emitted as an mcode
stream assembled from a reference Pulsar2 build with caller-chosen
output scales (found by value as float32 words -- Neg carries its
output scale as four stride-7 copies).

Status is honestly v0: scale words are mapped and verified (the MinMax
formula reproduces Pulsar2's scales to 1e-10; emitted streams
round-trip exactly and pass ``mcode.check``), but the emitted stream
runs ~0.19-vs-0.006 against ORT -- the input-side and zero-point
encodings are still open (see the README's "Emitting mcode" section
for the full field map and the gap analysis). tinygrad itself is an
optional, lazily-imported dependency: everything else here needs only
``numpy``.
"""

from __future__ import annotations

import struct


def trace_neg(tensor):
    """Match a tinygrad tensor holding unary negation.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``MUL(x, const(-1.0))`` (what ``-t`` lowers to), else raises
    ``ValueError``. tinygrad is imported lazily so this module stays
    importable without it.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.MUL or len(uop.src) != 2:
        raise ValueError(f"not a multiply: {uop.op}")
    data, const = uop.src
    if const.op is not Ops.CONST or float(const.arg) != -1.0:
        raise ValueError(f"not a multiply-by-minus-one: {const}")
    shape = list(data.shape)
    return {"shape": shape, "dtype": str(data.dtype)}


def find_scale_words(mcode: bytes, scale: float) -> list:
    """Offsets of every float32 occurrence of ``scale`` in the stream.

    Neg carries its output scale as four stride-7 copies; this returns
    wherever the value literally occurs so the caller can decide which
    copies are output-side. No layout assumptions beyond the byte match.
    """
    if isinstance(mcode, bytearray):
        mcode = bytes(mcode)
    pat = struct.pack("<f", float(scale))
    return [i for i in range(len(mcode) - 3) if mcode[i : i + 4] == pat]


def minmax_scale(samples) -> float:
    """Pulsar2's output-scale formula, verified to ~1e-10 against six
    real Neg builds: ``(max - min) / 255`` over the calibration samples
    (computed in float64). The zero point formula is still open (floor
    fits 4/9 builds), so this returns the scale only."""
    import numpy as np

    flat = np.concatenate([np.asarray(s).reshape(-1) for s in samples]).astype(
        np.float64
    )
    return float((flat.max() - flat.min()) / 255.0)


def emit_neg(reference_mcode: bytes, old_scale: float, new_scale: float) -> bytes:
    """Replace every float32 occurrence of ``old_scale`` with ``new_scale``.

    Both ends must be exactly representable (pass ``float(np.float32(x))``
    -- a float64 with extra digits never matches). Returns bytes that
    decode and round-trip exactly like the reference; whether they
    *compute* identically is a device question (see module docstring).
    """
    import struct as _struct

    new = _struct.pack("<f", float(new_scale))
    offsets = find_scale_words(reference_mcode, float(old_scale))
    if not offsets:
        raise ValueError(f"scale {old_scale!r} occurs nowhere: wrong reference?")
    out = bytearray(reference_mcode)
    for off in offsets:
        out[off : off + 4] = new
    return bytes(out)
