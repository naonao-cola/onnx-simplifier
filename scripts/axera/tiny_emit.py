#!/usr/bin/env python3
"""Emit NPU mcode for tinygrad-traced graphs -- first op: unary negation.

Pipeline: a tinygrad ``Tensor`` graph is traced to its UOp pattern (unary
negation lowers to ``MUL(x, -1.0)``), matched, and emitted as an mcode
stream assembled from a reference Pulsar2 build with caller-chosen
output scales (found by value as float32 words -- Neg carries its
output scale as four stride-7 copies).

``patch_mul_scales`` extends the same reference-patch idea to two-input
Mul streams' input side (sites A/C/B -- see
tests/test_axera_mcode_reciprocal.py for the field map): given the
reference build's recorded scales and the target scales, it rewrites the
input reciprocal and requant slots by value, verifying each family forms
its exact stride run. What it deliberately does not touch: the output
scale quads (``emitter.py``'s ``learn_mcode`` domain), the S-unit
programs (magnitude-adaptive shape, unmodeled ISA), the manifest string
table (tensor-name order varies build to build), and zero points.

Status is honestly v0+v1: scale words are mapped and verified (the MinMax
formula reproduces Pulsar2's scales to 1e-10; emitted streams
round-trip exactly and pass ``mcode.check``), but the emitted stream
runs ~0.19-vs-0.006 against ORT -- the zero-point
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


def _strided_run(mcode: bytes, pattern: bytes, stride: int, count: int = 4) -> list:
    """Offsets where ``pattern`` occurs as exactly ``count`` stride-run copies.

    Raises ``ValueError`` unless the occurrences are exactly ``count`` and
    land on a perfect stride grid -- an incidental byte collision anywhere
    else in the stream fails loudly instead of patching half a slot family.
    """
    hits = [
        i
        for i in range(len(mcode) - len(pattern) + 1)
        if mcode[i : i + len(pattern)] == pattern
    ]
    if len(hits) != count or any(b - a != stride for a, b in zip(hits, hits[1:])):
        raise ValueError(
            f"pattern {pattern.hex()} hits {hits}: not a stride-{stride} x{count} run"
        )
    return hits


def patch_mul_scales(reference_mcode: bytes, old_scales, new_scales) -> bytes:
    """Rewrite a two-input Mul stream's input-side scale slots by value.

    ``old_scales``/``new_scales`` are ``(x_scale, y_scale, z_scale)``
    triples -- the reference build's recorded MinMax scales and the
    target's. Three slot families move (see
    tests/test_axera_mcode_reciprocal.py):

    - site A (stride-8 x4): float32(1/x_scale), the x quant multiplier;
    - site C (stride-8 x4): float32(z_scale/(x_scale*y_scale)), the
      integer requant multiplier;
    - site B: float32(1/y_scale) x4 at stride 7 when the reference uses
      the full S-unit form, else the short form's low 3 bytes x4 at
      stride 6 (tag byte preserved). The reference's form is kept: this
      patches values, it does not recompile programs.

    Every family is stride-verified; a missing or ambiguous family
    raises. Output quads, S-unit programs, the string table and zero
    points are untouched (see module docstring).
    """
    old_x, old_y, old_z = (float(s) for s in old_scales)
    new_x, new_y, new_z = (float(s) for s in new_scales)
    # Locate every family on the pristine reference first: patching one
    # family must never disturb another family's search (overlapping
    # values across families would otherwise corrupt the later lookup).
    edits = []

    def locate(old_value: float, new_value: float, stride: int, width: int = 4):
        old_pat = struct.pack("<f", old_value)[:width]
        new_pat = struct.pack("<f", new_value)[:width]
        for off in _strided_run(reference_mcode, old_pat, stride):
            edits.append((off, new_pat))

    locate(1.0 / old_x, 1.0 / new_x, 8)
    locate(old_z / (old_x * old_y), new_z / (new_x * new_y), 8)
    try:
        locate(1.0 / old_y, 1.0 / new_y, 7)
    except ValueError:
        locate(1.0 / old_y, 1.0 / new_y, 6, width=3)
    out = bytearray(reference_mcode)
    for off, new_pat in edits:
        out[off : off + len(new_pat)] = new_pat
    return bytes(out)
