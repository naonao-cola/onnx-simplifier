"""Decode the register-write structure of a compiled ``teng2`` (elementwise
compute) segment, and read off the quantization scale registers.

Background: ``docs/axera-pulsar-v1-mining.md`` (PR #1818) showed ``mcode.py``'s
``V`` records are 8-byte register writes ``[verb][unit][reg_lo][reg_hi]
[value32]`` (``field``+``bank`` = a 16-bit register address). This module
applies that to a *standalone elementwise op*'s segment 2 and, via a
device fault-injection study (``docs/axera-teng2-register-decode.md``),
identifies the few registers that carry the op's quantization scales:

* **Quant multiplier** ``1 / input_scale`` and **dequant multiplier**
  ``output_scale`` are each written as float32 to registers ``0x0f60 /
  0x0f70 / 0x0f80`` -- three copies, one per output *lane* (element index
  ``mod 4``; lane 0 is carried by a companion ``W`` record just before
  them). Each register is written twice in the segment: occurrence 0 is the
  quant multiplier, occurrence 1 the dequant multiplier. Confirmed
  byte-exact on device: doubling the dequant copy of ``0x0f60`` doubled
  exactly the ``mod 4 == 1`` output elements.
* Registers ``0x1c20 / 0x1c30 / 0x1c40`` hold the per-lane activation clamp
  lower bound (the ReLU floor), again lane-mapped 1/2/3.

Everything else in segment 2 that this module surfaces is
calibration-invariant register programming (addresses, tile counts) whose
individual meanings are not all decoded. This is a *reader*, not an
emitter: it does not synthesize a segment, and it does not touch the
variable-length short units (which carry the zero-point byte and whose
width shifts the whole stream -- see the doc).
"""

from __future__ import annotations

import gzip
import os
import struct
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

SCALE_REGS = (0x0F60, 0x0F70, 0x0F80)
"""The three per-lane registers carrying float32 quant/dequant multipliers."""

CLAMP_REGS = (0x1C20, 0x1C30, 0x1C40)
"""The three per-lane registers carrying the activation clamp lower bound."""


def load_mcode(path: str) -> bytes:
    """Raw ``*_neu`` mcode bytes from an ``.axmodel`` (optionally gzipped)."""
    data = (
        gzip.open(path, "rb").read()
        if path.endswith(".gz")
        else open(path, "rb").read()
    )
    model = onnx.load_model_from_string(data)
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


def segment_register_writes(mc: bytes, seg: int = 2) -> list[dict]:
    """Every ``V`` (and companion ``W``) register write in segment ``seg``, as
    ``{at, reg, value, width, kind, occurrence}`` in stream order.

    ``occurrence`` counts how many times this register has been written so far
    in the segment (0 for the first), which is what distinguishes the quant
    copy (0) from the dequant copy (1) of a scale register.
    """
    _, segs = mcode.segments(mc)
    pos, length, _ = segs[seg]
    records = mcode.decode(mc, start=pos, end=pos + length, **mcode.FULL_RULE)
    seen: dict[int, int] = {}
    out = []
    for r in records:
        if r["kind"] not in ("V", "W"):
            continue
        reg = r["field"] | (r["bank"] << 8)
        operand = bytes(r["operand"])
        value = int.from_bytes(operand.ljust(4, b"\0")[:4], "little")
        occ = seen.get(reg, 0)
        seen[reg] = occ + 1
        out.append(
            {
                "at": r["at"],
                "kind": r["kind"],
                "reg": reg,
                "value": value,
                "width": len(operand),
                "occurrence": occ,
            }
        )
    return out


def _f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def scale_registers(mc: bytes, seg: int = 2) -> dict:
    """The float32 quant (occurrence 0) and dequant (occurrence 1) multipliers
    read from ``SCALE_REGS``.

    Returns ``{"quant": [f, ...], "dequant": [f, ...]}`` -- one entry per
    ``(register, occurrence)`` found. On a well-formed elementwise segment the
    quant values are all ``1 / input_scale`` and the dequant values all
    ``output_scale``; a caller can average or cross-check them.
    """
    quant, dequant = [], []
    for w in segment_register_writes(mc, seg):
        if w["reg"] in SCALE_REGS and w["width"] == 4:
            (quant if w["occurrence"] == 0 else dequant).append(_f32(w["value"]))
    return {"quant": quant, "dequant": dequant}


def implied_scales(mc: bytes, seg: int = 2) -> dict:
    """``{"input_scale", "output_scale"}`` implied by the scale registers, i.e.
    ``1 / mean(quant)`` and ``mean(dequant)``. Raises ``ValueError`` if the
    scale registers are absent (not an elementwise ``teng2`` segment of the
    measured form)."""
    try:
        s = scale_registers(mc, seg)
    except (AssertionError, IndexError, ValueError) as exc:
        raise ValueError(f"not a decodable mcode segment: {exc}") from exc
    if not s["quant"] or not s["dequant"]:
        raise ValueError("no teng2 scale registers (0x0f60/70/80) found in segment")
    q = sum(s["quant"]) / len(s["quant"])
    d = sum(s["dequant"]) / len(s["dequant"])
    return {"input_scale": 1.0 / q, "output_scale": d}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        mc = load_mcode(path)
        writes = segment_register_writes(mc)
        s = scale_registers(mc)
        print(
            f"== {os.path.basename(path)}: {len(writes)} register writes in segment 2"
        )
        print(f"   quant multipliers (1/input_scale): {s['quant']}")
        print(f"   dequant multipliers (output_scale): {s['dequant']}")
        try:
            imp = implied_scales(mc)
            print(
                f"   implied input_scale={imp['input_scale']:.6g} output_scale={imp['output_scale']:.6g}"
            )
        except ValueError as exc:
            print(f"   {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
