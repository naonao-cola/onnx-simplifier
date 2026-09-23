"""Retarget a compiled standalone AX650 elementwise op to a new quantization
scale without Pulsar2.

``docs/axera-teng2-register-decode.md`` (PR #1831) located the calibration
floats in a standalone ``Relu``'s ``teng2`` (TENG-engine) segment. A controlled
calibration sweep (``elementwise_scale_sweep.py``: every sample spans exactly
``[lo, hi]``, so Pulsar2's MinMax scale and zero point are known per build)
shows that, **at a fixed zero point**, the whole compiled model depends on the
scale through exactly eight float32 copies in segment 2 and nothing else:

* the **quant multiplier** ``f32(1 / s)`` -- the operand of the ``W`` record
  that writes register ``0x0f50`` (lane 0) and of the three ``V`` records that
  first write ``0x0f60 / 0x0f70 / 0x0f80`` (lanes 1-3);
* the **dequant multiplier** ``f32(s)`` -- four literal bytes immediately
  before the second (dequant) write of ``0x0f60`` (lane 0), and the operands of
  the second writes of ``0x0f60 / 0x0f70 / 0x0f80``.

``s`` is Pulsar2's own float32 scale (``quant_axmodel.json``); both formulas
reproduce every one of 35 builds bit for bit, including scales that are not a
power-of-two multiple of each other (which change mantissa bytes, not just
exponents). ``npu_params``, the node attributes and every other MCode byte are
calibration-invariant at a fixed zero point.

The **zero point is not retargetable here**. It is carried by the
variable-length units of the stream: changing it inserts or removes bytes and
can move the segment across a 32-byte boundary (``zp == 0`` and ``zp != 0``
give different segment lengths). So a template serves exactly one zero point.
Two templates per shape cover two practical classes completely: ``zp == 128``
(any symmetric calibration range, ``lo == -hi``) and ``zp == 0`` (any
non-negative range, ``lo == 0``).

Everything outside the measured scope raises ``ValueError``: an op or shape
without a validated template, a zero point other than the template's, or a
template whose registers do not match its recorded calibration.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import sys

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

QUANT_LANE0_REG = 0x0F50
SCALE_REGS = (0x0F60, 0x0F70, 0x0F80)
TEMPLATE_DIR = os.path.join(_HERE, "fixtures", "elementwise_scale_emit")


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(np.float32(x))))[0]


def scale_floats(scale: float) -> tuple[bytes, bytes]:
    """``(quant, dequant)`` float32 little-endian bytes for Pulsar2 scale ``scale``.

    ``scale`` is first rounded to float32 (Pulsar2 stores it that way);
    quant is ``f32(1 / s)`` and dequant ``f32(s)``.
    """
    s = float(np.float32(scale))
    if not (s > 0 and np.isfinite(s)):
        raise ValueError(f"scale must be a positive finite float, got {scale!r}")
    return struct.pack("<I", _f32_bits(1.0 / s)), struct.pack("<I", _f32_bits(s))


def minmax_params(lo: float, hi: float) -> tuple[float, int]:
    """Pulsar2's MinMax U8 asymmetric ``(scale, zero_point)`` for a calibration
    range whose realised float32 min/max are ``lo``/``hi``."""
    lo64 = float(np.float32(min(lo, 0.0)))
    hi64 = float(np.float32(max(hi, 0.0)))
    span = hi64 - lo64
    if not span > 0:
        raise ValueError(f"empty calibration range [{lo}, {hi}]")
    scale = float(np.float32(span / 255.0))
    # zero point from the unrounded range, round-half-to-even (127.5 -> 128)
    zp = int(np.clip(np.round(-lo64 * 255.0 / span), 0, 255))
    return scale, zp


def _mcode_initializer(model: onnx.ModelProto):
    matches = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(f"expected one *_neu MCode initializer, found {len(matches)}")
    return matches[0]


def _groups_of_four(offsets: list[int]) -> bool:
    """Whether ``offsets`` splits into consecutive groups of four lane copies,
    each spaced one register-write record apart (7 or 8 bytes: the stream's
    compact and full record forms)."""
    if not offsets or len(offsets) % 4:
        return False
    for g in range(0, len(offsets), 4):
        grp = offsets[g : g + 4]
        if any(b - a not in (7, 8) for a, b in zip(grp, grp[1:])):
            return False
    return True


def scale_slots(mc: bytes, scale: float) -> dict[str, list[int]]:
    """Byte offsets (into the whole MCode blob) of every quant and dequant
    float32 copy for ``scale`` in segment 2.

    Copies are located by value, because the stream's variable-length forms
    tokenize the same four-lane group differently at different shapes (a
    batch-1 Relu writes the dequant group as full ``V`` records, a batch-16
    one as compact 7-byte records, and a tiled program repeats the quant
    group per tile). The structure is still checked: both floats must occur
    in whole groups of four lane copies one record apart, entirely inside
    segment 2, without overlapping. Raises ``ValueError`` otherwise.
    """
    try:
        _, segs = mcode.segments(mc)
    except (AssertionError, IndexError, ValueError, struct.error) as exc:
        raise ValueError(f"not a decodable MCode stream: {exc}") from exc
    pos, length, _ = segs[2]
    quant, dequant = scale_floats(scale)
    if quant == dequant:
        raise ValueError("quant and dequant floats coincide; cannot tell copies apart")
    found = {}
    for key, pat in (("quant", quant), ("dequant", dequant)):
        offs = [o for o in range(pos, pos + length - 3) if mc[o : o + 4] == pat]
        if not _groups_of_four(offs):
            raise ValueError(
                f"segment 2 lacks the measured {key}-float structure: copies at "
                f"{[o - pos for o in offs]} (segment-relative) are not whole groups "
                "of four lane records"
            )
        found[key] = offs
    spans = sorted((o, o + 4) for offs in found.values() for o in offs)
    if any(a_end > b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:])):
        raise ValueError("quant and dequant float copies overlap")
    return found


def retarget_scale(mc: bytes, old_scale: float, new_scale: float) -> bytes:
    """``mc`` with every quant/dequant float copy rewritten from
    ``old_scale`` to ``new_scale``."""
    slots = scale_slots(mc, old_scale)
    new_q, new_d = scale_floats(new_scale)
    out = bytearray(mc)
    for key, new in (("quant", new_q), ("dequant", new_d)):
        for off in slots[key]:
            out[off : off + 4] = new
    return bytes(out)


def load_template(op: str, shape, zero_point: int, template_dir: str = TEMPLATE_DIR):
    """``(model, meta)`` for a committed template, or ``ValueError`` if no
    validated template covers ``(op, shape, zero_point)``."""
    index_path = os.path.join(template_dir, "index.json")
    with open(index_path) as f:
        index = json.load(f)
    key = f"{op}:{'x'.join(str(d) for d in shape)}:zp{int(zero_point)}"
    if key not in index:
        known = sorted(index)
        raise ValueError(f"no validated template for {key}; have {known}")
    meta = index[key]
    with gzip.open(os.path.join(template_dir, meta["file"]), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return model, meta


def emit(
    op: str,
    shape,
    scale: float,
    zero_point: int,
    out_path: str,
    template_dir: str = TEMPLATE_DIR,
) -> str:
    """Write a compiled ``op(x[shape])`` model for Pulsar2 quantization
    ``(scale, zero_point)`` to ``out_path`` from a committed template."""
    zero_point = int(zero_point)
    if not 0 <= zero_point <= 255:
        raise ValueError(f"zero_point must be in [0, 255], got {zero_point}")
    model, meta = load_template(op, shape, zero_point, template_dir)
    init = _mcode_initializer(model)
    init.raw_data = retarget_scale(bytes(init.raw_data), meta["scale"], scale)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    return out_path


def emit_from_reference(
    reference_path: str,
    reference_scale: float,
    reference_zero_point: int,
    scale: float,
    zero_point: int,
    out_path: str,
) -> str:
    """Like ``emit`` but from any compiled single-op reference whose own
    calibration ``(reference_scale, reference_zero_point)`` is known (e.g. from
    its ``quant_axmodel.json``). The zero point must not change."""
    if int(zero_point) != int(reference_zero_point):
        raise ValueError(
            f"zero point change {reference_zero_point} -> {zero_point} is not "
            "supported: it reflows the variable-length stream"
        )
    model = onnx.load(reference_path, load_external_data=False)
    init = _mcode_initializer(model)
    init.raw_data = retarget_scale(bytes(init.raw_data), reference_scale, scale)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    return out_path
