"""Read the quantization registers of a compiled two-input elementwise op
(``Add``, ``Sub``, ``Mul``, ``Div``) off its AX650 mcode and ``npu_params``.

Device-confirmed semantics (``docs/axera-binary-op-register-semantics.md``):
each value below was patched by a known factor in one compiled model and run
on the AX8850 with x-only, z-only and both-input stimuli; the effect was
measured per output lane (flat element index ``mod 4``).

TENG program (``segments()[2]``), for all four ops -- every scale-like
quantity is written as **four per-lane copies** 0x10 apart (lane 0 first,
lanes 1-3 after it). The lane-1..3 copies are ``V`` records or 7-byte
compressed short units (``S``, ``p=3``); the lane-0 copy is a ``W`` record or
an ``S`` unit that also carries the register address:

* ``1/x_scale`` at ``0x0f50..0x0f80``: input ``x``'s float->int8 quantize
  multiplier. Halving one lane's copy halved exactly ``x``'s contribution in
  exactly that lane.
* ``1/z_scale``, written next (same register block, second pass): input
  ``z``'s quantize multiplier; same per-lane, per-input behaviour. It is
  sometimes stored in a shorter compressed form this reader cannot decode
  (then ``z_quant`` is ``None`` and ``z_scale`` is derived another way).
* ``Mul`` only: ``y_scale / (x_scale * z_scale)`` at ``0x0fd0..0x1000`` -- a
  per-lane **divisor** (doubling it halved that lane's product).
* ``y_scale``, written last: the int8->float dequantize multiplier.
* ``Div`` has no ratio register; its quotient scales as
  ``(1/x) / (1/z)`` per lane (halving the ``1/z`` copy doubled that lane).

``npu_params`` header, ``Add``/``Sub`` only: Q15 requant multipliers applied
to the *raw* zero-point-included int8 codes, uniformly across all lanes.
When ``x_scale/y_scale == z_scale/y_scale`` the compiler emits one shared
word (applied to ``q_x + q_z``) and no CV program; otherwise it emits
``x/y`` and ``z/y`` as two words and adds a CV-unit program (``cv3``,
``segments()[3]``, CV EU6). ``Sub`` stores ``z/y`` positive; the subtraction
is applied elsewhere, so ``Add`` and ``Sub`` cannot be told apart from these
values -- the caller passes the op.

``mcode.decode`` does not always tokenize the lane-0 copy (``W`` record or
address-carrying short unit) as a value write, so a run may list 3 copies
(lanes 1-3) instead of 4; the recovered values are unaffected.

This is a reader, not an emitter. It does not decode the zero-point short
units, the CV program's registers, or the variable-length compressed value
forms (see the doc for what is and isn't confirmed).
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

TENG_SEG = 2
CV_SEG = 3
X_QUANT_REGS = (0x0F50, 0x0F60, 0x0F70, 0x0F80)
MUL_DIVISOR_REGS = (0x0FD0, 0x0FE0, 0x0FF0, 0x1000)
Q15 = 32768.0


def load(path: str) -> tuple[bytes, bytes]:
    """``(mcode, npu_params)`` from an ``.axmodel`` (optionally gzipped)."""
    data = (
        gzip.open(path, "rb").read()
        if path.endswith(".gz")
        else open(path, "rb").read()
    )
    model = onnx.load_model_from_string(data)
    mc = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )
    params = next(
        (bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"),
        b"",
    )
    return mc, params


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _records(mc: bytes, seg: int) -> list[dict]:
    _, segs = mcode.segments(mc)
    pos, length, _ = segs[seg]
    return mcode.decode(mc, start=pos, end=pos + length, **mcode.FULL_RULE)


def _carrier(r: dict) -> tuple[int | None, int] | None:
    """``(register_or_None, value32)`` if record ``r`` writes a full 32-bit value."""
    if r["kind"] in ("V", "W") and len(r["operand"]) == 4:
        return r["field"] | (r["bank"] << 8), int.from_bytes(
            bytes(r["operand"]), "little"
        )
    if r["kind"] == "S":
        payload = bytes(r["payload"])
        if len(payload) == 4:
            return None, int.from_bytes(payload, "little")
        if len(payload) == 6:  # address-carrying first copy: [reg lo][reg hi][value32]
            return int.from_bytes(payload[:2], "little"), int.from_bytes(
                payload[2:], "little"
            )
    return None


def float_runs(mc: bytes, seg: int = TENG_SEG, min_len: int = 3) -> list[dict]:
    """Runs of consecutive records writing the same float32 value (the per-lane
    copies), in stream order: ``{value, bits, count, offsets, regs, kinds}``.
    Only plausible scale magnitudes (1e-8..1e8) are kept."""
    runs: list[dict] = []
    cur: dict | None = None
    for r in _records(mc, seg):
        c = _carrier(r)
        if c is None:
            cur = None
            continue
        reg, bits = c
        if cur is not None and cur["bits"] == bits:
            cur["count"] += 1
            cur["offsets"].append(r["at"])
            cur["regs"].append(reg)
            cur["kinds"].append(r["kind"])
            continue
        cur = {
            "bits": bits,
            "count": 1,
            "offsets": [r["at"]],
            "regs": [reg],
            "kinds": [r["kind"]],
        }
        runs.append(cur)
    out = []
    for run in runs:
        v = _f32(run["bits"])
        if run["count"] >= min_len and 1e-8 < abs(v) < 1e8 and v == v:
            out.append({**run, "value": v})
    return out


def has_cv_program(mc: bytes) -> bool:
    """Whether the CV-unit segment carries a real program (more than the
    two-write sync/end stub seen on single-ratio ops)."""
    _, segs = mcode.segments(mc)
    if len(segs) <= CV_SEG:
        return False
    writes = [r for r in _records(mc, CV_SEG) if r["kind"] in ("V", "W")]
    return len(writes) > 2


def decode(mc: bytes, params: bytes, op: str) -> dict:
    """Quantization values of a compiled ``op`` (``"add"``/``"sub"``/``"mul"``/
    ``"div"``) and the scales they imply. Raises ``ValueError`` if the TENG
    segment does not have the measured form."""
    op = op.lower()
    if op not in ("add", "sub", "mul", "div"):
        raise ValueError(f"unsupported op {op!r}")
    runs = float_runs(mc)
    if len(runs) < 2:
        raise ValueError("fewer than two per-lane float runs in the TENG segment")
    x_run = next((r for r in runs if set(r["regs"]) & set(X_QUANT_REGS)), None)
    if x_run is None:
        raise ValueError("no 1/x_scale run on registers 0x0f50..0x0f80")
    y_run = runs[-1]
    div_run = None
    if op == "mul":
        div_run = next(
            (r for r in runs if set(r["regs"]) & set(MUL_DIVISOR_REGS)), None
        )
    xi, yi = runs.index(x_run), runs.index(y_run)
    stop = runs.index(div_run) if div_run is not None else yi
    between = [r for r in runs[xi + 1 : stop] if r is not div_run]
    z_run = between[0] if between else None

    out: dict = {
        "op": op,
        "x_quant": x_run["value"],
        "z_quant": z_run["value"] if z_run else None,
        "y_dequant": y_run["value"],
        "mul_divisor": div_run["value"] if div_run else None,
        "cv_program": has_cv_program(mc),
        "q15": None,
    }
    x_scale = 1.0 / out["x_quant"]
    y_scale = out["y_dequant"]
    z_scale = 1.0 / out["z_quant"] if out["z_quant"] else None

    if op in ("add", "sub") and len(params) >= 4:
        w0, w1 = struct.unpack_from("<2H", params, 0)
        if out["cv_program"]:
            out["q15"] = {"x_over_y": w0 / Q15, "z_over_y": w1 / Q15}
            if z_scale is None:
                z_scale = out["q15"]["z_over_y"] * y_scale
        else:
            out["q15"] = {"shared": w0 / Q15}
            if z_scale is None:
                z_scale = out["q15"]["shared"] * y_scale
    if op == "mul" and z_scale is None and out["mul_divisor"]:
        z_scale = y_scale / (x_scale * out["mul_divisor"])
    out["scales"] = {"x": x_scale, "z": z_scale, "y": y_scale}
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("usage: binary_op_register_decode.py OP MODEL.axmodel[.gz] ...")
        return 2
    op = argv[0]
    for path in argv[1:]:
        mc, params = load(path)
        d = decode(mc, params, op)
        print(f"== {os.path.basename(path)} ({op})")
        for k in (
            "x_quant",
            "z_quant",
            "mul_divisor",
            "y_dequant",
            "cv_program",
            "q15",
            "scales",
        ):
            print(f"   {k}: {d[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
