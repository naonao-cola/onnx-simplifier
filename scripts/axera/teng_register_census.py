"""Register-write census of the compiled elementwise-op programs in a ResNet18
training step, across engineered calibration variants.

``docs/axera-pulsar-v1-mining.md`` (PR #1818) showed ``mcode.py``'s 8-byte
``V`` records are register writes ``[verb][unit][reg_lo][reg_hi][value32]``, and
PR #1831 (``docs/axera-teng2-register-decode.md``) used that to find Relu's
quant/dequant scale registers. This module applies the same lens to every
elementwise-type op the step uses (Mul, Add, Sub, Div, Sqrt, Relu,
Greater->Cast, ReduceSum), offline:

* ``build``: compile one standalone model per (op, calibration variant) with
  Pulsar2. Calibration data are *engineered* -- every sample is the same
  ``linspace(lo, hi)`` array -- so each input's min/max is known exactly.
* ``census``: parse every segment of every build as a list of register writes
  keyed by ``(segment, kind, register, occurrence)`` (robust to the stream
  re-flowing around variable-length short units), find the registers whose
  value changes with calibration, and test each against scale formulas built
  from the quantization parameters Pulsar2 actually chose (read from the
  build's own ``quant/quant_axmodel.onnx``, filtered by op type).

No device is used. See ``docs/axera-teng-register-census.md`` for results.

Usage::

    teng_register_census.py build WORK_ROOT [OP ...]
    teng_register_census.py census WORK_ROOT [--json OUT.json]
"""

from __future__ import annotations

import gzip
import io
import itertools
import json
import math
import os
import struct
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
from onnx import parser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"

SEGMENT_ENGINE = {0: "CONV", 1: "CONV", 2: "TENG", 3: "CV", 4: "SDMA"}
"""Segment -> engine, per ``docs/axera-step-attribution.md``'s stream order and
the device logs of PRs #1821/#1823/#1831 (segment 2 on TENG EU9, segment 3 on
CV EU6)."""

_SYM1 = (-1.0, 1.0)
_SYM4 = (-4.0, 4.0)
_ASYM = (-0.5, 1.5)
_POS1 = (0.25, 1.0)
_POS4 = (1.0, 4.0)

OPS: dict[str, dict] = {
    "relu": {
        "graph": "(float[1,64,56,56] x) => (float[1,64,56,56] y) { y = Relu(x) }",
        "inputs": {"x": (1, 64, 56, 56)},
        "variants": {
            "s1": {"x": _SYM1},
            "s4": {"x": _SYM4},
            "asym": {"x": _ASYM},
            # Same scale as s1 (2/255), zero point 0 instead of 128.
            "zp0": {"x": (0.0, 2.0)},
        },
    },
    "add": {
        "graph": "(float[1,64,56,56] x, float[1,64,56,56] z) => (float[1,64,56,56] y)"
        " { y = Add(x, z) }",
        "inputs": {"x": (1, 64, 56, 56), "z": (1, 64, 56, 56)},
        "variants": {
            "s1": {"x": _SYM1, "z": _SYM1},
            "s4": {"x": _SYM4, "z": _SYM4},
            "mix": {"x": _SYM1, "z": _SYM4},
            "asym": {"x": _ASYM, "z": (-3.0, 1.0)},
            # mix/asym give power-of-two scale ratios; r3 gives 3.
            "r3": {"x": _SYM1, "z": (-3.0, 3.0)},
            # Same input scales as s1, both zero points 64 instead of 128.
            "zp": {"x": _ASYM, "z": _ASYM},
        },
    },
    "mul": {
        "graph": "(float[1,64,56,56] x, float[1,64,56,56] z) => (float[1,64,56,56] y)"
        " { y = Mul(x, z) }",
        "inputs": {"x": (1, 64, 56, 56), "z": (1, 64, 56, 56)},
        "variants": {
            "s1": {"x": _SYM1, "z": _SYM1},
            "s4": {"x": _SYM4, "z": _SYM4},
            "mix": {"x": _SYM1, "z": _SYM4},
            "asym": {"x": _ASYM, "z": (-3.0, 1.0)},
            "r3": {"x": _SYM1, "z": (-3.0, 3.0)},
            # The symmetric variants all give y/(x*z) = 255/4; this one does not.
            "pos": {"x": (0.5, 1.0), "z": (0.5, 2.0)},
        },
    },
    "sub": {
        "graph": "(float[64,64,3,3] x, float[64,64,3,3] z) => (float[64,64,3,3] y)"
        " { y = Sub(x, z) }",
        "inputs": {"x": (64, 64, 3, 3), "z": (64, 64, 3, 3)},
        # With identical x and z arrays at equal ranges, x - z is exactly zero and
        # MinMax gives y_scale = 0; rolling z keeps its min/max but not the pairing.
        "roll": {"z": 1 / 3},
        "variants": {
            "s1": {"x": _SYM1, "z": _SYM1},
            "s4": {"x": _SYM4, "z": _SYM4},
            "mix": {"x": _SYM1, "z": _SYM4},
            "asym": {"x": _ASYM, "z": (-3.0, 1.0)},
            "r3": {"x": _SYM1, "z": (-3.0, 3.0)},
            "zp": {"x": _ASYM, "z": _ASYM},
        },
    },
    "div": {
        # Denominators kept positive, as in the step's Adam update (m / sqrt(v)).
        "graph": "(float[64,64,3,3] x, float[64,64,3,3] z) => (float[64,64,3,3] y)"
        " { y = Div(x, z) }",
        "inputs": {"x": (64, 64, 3, 3), "z": (64, 64, 3, 3)},
        "variants": {
            "s1": {"x": _SYM1, "z": _POS1},
            "s4": {"x": _SYM4, "z": _POS4},
            "mix": {"x": _SYM1, "z": _POS4},
            "asym": {"x": _ASYM, "z": (0.5, 2.0)},
            "r3": {"x": _SYM1, "z": (0.5, 3.0)},
        },
    },
    "sqrt": {
        "graph": "(float[64,64,3,3] x) => (float[64,64,3,3] y) { y = Sqrt(x) }",
        "inputs": {"x": (64, 64, 3, 3)},
        "variants": {
            "p1": {"x": (0.0, 1.0)},
            "p4": {"x": (0.0, 4.0)},
            "p9": {"x": (0.25, 9.0)},
        },
    },
    "gtcast": {
        # The step's form: Greater(x, 0-d constant) -> Cast(to=FLOAT), the Relu mask.
        "graph": "(float[1,64,56,56] x) => (float[1,64,56,56] y) {"
        " c = Constant <value = float {0.0}> ()  b = Greater(x, c)"
        "  y = Cast <to = 1> (b) }",
        "inputs": {"x": (1, 64, 56, 56)},
        "variants": {"s1": {"x": _SYM1}, "s4": {"x": _SYM4}, "asym": {"x": _ASYM}},
    },
    "reducesum": {
        # The step's form: ReduceSum over axis 2, keepdims=0.
        "graph": "(float[1,1,64,3136] x) => (float[1,1,3136] y) {"
        " a = Constant <value = int64[1] {2}> ()"
        "  y = ReduceSum <keepdims = 0> (x, a) }",
        "inputs": {"x": (1, 1, 64, 3136)},
        "variants": {"s1": {"x": _SYM1}, "s4": {"x": _SYM4}, "asym": {"x": _ASYM}},
    },
}


# --------------------------------------------------------------------- build


def _model(op: str) -> onnx.ModelProto:
    text = f'<ir_version: 8, opset_import: ["" : 13]> g {OPS[op]["graph"]}'
    return parser.parse_model(text)


def _calib_array(shape, lo: float, hi: float) -> np.ndarray:
    n = int(np.prod(shape))
    return np.linspace(lo, hi, n, dtype=np.float64).astype(np.float32).reshape(shape)


def build_one(root: str, op: str, variant: str) -> int:
    """Compile ``(op, variant)`` to ``<root>/<op>_<variant>/out/compiled.axmodel``."""
    wd = os.path.join(root, f"{op}_{variant}")
    if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
        return 0
    os.makedirs(os.path.join(wd, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(wd, "config"), exist_ok=True)
    onnx.save(_model(op), os.path.join(wd, "t.onnx"))
    spec = OPS[op]
    cfg_inputs = []
    for name, shape in spec["inputs"].items():
        lo, hi = spec["variants"][variant][name]
        arr = _calib_array(shape, lo, hi)
        if name in spec.get("roll", {}):
            flat = arr.reshape(-1)
            arr = np.roll(flat, int(len(flat) * spec["roll"][name])).reshape(shape)
        with tarfile.open(os.path.join(wd, "dataset", f"{name}.tar"), "w") as tar:
            for k in range(2):
                buf = io.BytesIO()
                np.save(buf, arr)
                info = tarfile.TarInfo(f"{k}.npy")
                info.size = len(buf.getvalue())
                buf.seek(0)
                tar.addfile(info, buf)
        cfg_inputs.append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./dataset/{name}.tar",
                "calibration_format": "Numpy",
                "calibration_size": 2,
            }
        )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": cfg_inputs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    with open(os.path.join(wd, "config", "c.json"), "w") as f:
        json.dump(config, f)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"census-{op}-{variant}-{os.getpid()}",
            "-v",
            f"{wd}:/data",
            _IMAGE,
            "pulsar2",
            "build",
            "--target_hardware",
            "AX650",
            "--input",
            "t.onnx",
            "--output_dir",
            "out",
            "--config",
            "config/c.json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        with open(os.path.join(wd, "build.log"), "w") as f:
            f.write(result.stdout[-4000:] + result.stderr[-4000:])
    return result.returncode


# --------------------------------------------------------------- extraction


def load_axmodel(path: str) -> onnx.ModelProto:
    data = (
        gzip.open(path, "rb").read()
        if path.endswith(".gz")
        else open(path, "rb").read()
    )
    return onnx.load_model_from_string(data)


def mcode_of(model: onnx.ModelProto) -> bytes:
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


NOISE_WINDOW = (301, 326)
"""Absolute mcode offsets that differ between identical rebuilds (compiler
noise; see ``scripts/axera/README.md``). Writes starting here are dropped."""


def register_writes(mc: bytes, include_short: bool = True) -> list[dict]:
    """Every value-carrying record in every segment, in stream order.

    ``V``/``W`` records are register writes (PR #1818): ``reg`` is
    ``field | bank << 8`` and ``value`` the operand as little-endian uint32.
    With ``include_short``, compressed short units (``S``, ``[p][payload]
    [tag][reg]``) are included too -- PR #1837 device-confirmed that 7-byte
    ``p=3`` units are register writes as well. Their register byte is not a
    full address, so they are keyed by ``(tag, reg)`` as a string
    ``"S:tt:rr"``; a 6-byte payload's first two bytes are an explicit register
    address (``[reg lo][reg hi][value32]``, the lane-0 form) and are keyed by
    that address.

    Each entry: ``seg``, ``at``, ``kind``, ``reg`` (int for ``V``/``W`` and
    address-carrying ``S``, else the ``"S:tt:rr"`` string), ``value``
    (uint32, zero-padded), ``width`` (payload/operand bytes), ``length``
    (bytes the record occupies in the stream), and ``occurrence`` (earlier
    writes of the same ``(seg, reg)``). Records in the compiler-noise window
    are dropped.
    """
    _, segs = mcode.segments(mc)
    out = []
    for seg, (pos, length, _) in enumerate(segs):
        records = mcode.decode(mc, start=pos, end=pos + length, **mcode.FULL_RULE)
        seen: dict = {}
        for i, r in enumerate(records):
            if NOISE_WINDOW[0] <= r["at"] < NOISE_WINDOW[1]:
                continue
            nxt = records[i + 1]["at"] if i + 1 < len(records) else pos + length
            if r["kind"] in ("V", "W"):
                reg = r["field"] | (r["bank"] << 8)
                data = bytes(r["operand"])
            elif r["kind"] == "S" and include_short:
                payload = bytes(r["payload"])
                if len(payload) == 6:
                    reg = int.from_bytes(payload[:2], "little")
                    data = payload[2:]
                else:
                    reg = f"S:{r['tag']:02x}:{r['reg']:02x}"
                    data = payload
            else:
                continue
            occ = seen.get(reg, 0)
            seen[reg] = occ + 1
            out.append(
                {
                    "seg": seg,
                    "at": r["at"],
                    "kind": r["kind"],
                    "reg": reg,
                    "value": int.from_bytes(data.ljust(4, b"\0")[:4], "little"),
                    "width": len(data),
                    "length": nxt - r["at"],
                    "occurrence": occ,
                }
            )
    return out


def _reg_label(reg) -> str:
    return f"0x{reg:04x}" if isinstance(reg, int) else reg


def segment_summary(mc: bytes) -> list[dict]:
    """Per segment: allocated size, non-padding content length, and record mix."""
    _, segs = mcode.segments(mc)
    out = []
    for seg, (pos, length, _) in enumerate(segs):
        body = mc[pos : pos + length]
        content = max((i for i, v in enumerate(body) if v), default=-1) + 1
        kinds: dict[str, int] = {}
        for r in mcode.decode(mc, start=pos, end=pos + length, **mcode.FULL_RULE):
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        out.append({"seg": seg, "size": length, "content": content, "kinds": kinds})
    return out


def quant_params(quant_onnx: onnx.ModelProto) -> dict:
    """``{"tensors": {name: (scale, zp)}, "ops": [(op_type, attrs)]}`` from a
    build's ``quant_axmodel.onnx``.

    Input tensors take their parameters from ``AxQuantizeLinear`` outputs, and
    output tensors from ``AxDequantizeLinear`` inputs -- read per op type, never
    from a merged all-nodes dict (see the project's
    axera-quant-model-input-scales-overwrite-bug note).
    """
    tensors: dict[str, tuple[float, int]] = {}
    ops = []
    for n in quant_onnx.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        clean = {
            k: (
                v.decode()
                if isinstance(v, bytes)
                else list(v)
                if isinstance(v, list)
                else v
            )
            for k, v in attrs.items()
        }
        ops.append((n.op_type, clean))
        if n.op_type == "AxQuantizeLinear":
            tensors[n.input[0]] = (
                float(attrs["output_scales"][0]),
                int(attrs["output_zeropoints"][0]),
            )
        elif n.op_type == "AxDequantizeLinear":
            tensors[n.output[0]] = (
                float(attrs["input_scales"][0]),
                int(attrs["input_zeropoints"][0]),
            )
    inits = {i.name for i in quant_onnx.graph.initializer}
    del inits
    return {"tensors": tensors, "ops": ops}


# ------------------------------------------------------------------ analysis


def _f32(v: int) -> float:
    return struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]


def scale_monomials(tensors: dict[str, tuple[float, int]]) -> dict[str, float]:
    """Every product ``prod(s_t ** e_t)`` with ``e_t in {-1, 0, 1}`` over the
    tensors' scales (the identity excluded), named like ``s_x/s_y``."""
    names = sorted(tensors)
    out = {}
    for exps in itertools.product((-1, 0, 1), repeat=len(names)):
        if not any(exps):
            continue
        val = 1.0
        num, den = [], []
        for name, e in zip(names, exps):
            if e:
                val *= tensors[name][0] ** e
                (num if e > 0 else den).append(f"s_{name}")
        label = "*".join(num) if num else "1"
        if den:
            label += "/" + "*".join(den)
        out[label] = val
    return out


def match_register(
    values: list[int], params: list[dict], zero_points_only: bool = False
) -> list[str]:
    """Formulas consistent with a register's per-variant ``values``.

    Tries: the value as float32 equal (to float32 precision) to a scale
    monomial or its negation; the value, or its low or high 16-bit half, equal
    to a zero point; and the value equal to ``round(monomial * 2**k)`` for one
    fixed ``k`` across all variants (fixed-point, e.g. Q15). A formula counts
    only if it holds for *every* variant.
    """
    hits = []
    monos = [scale_monomials(p["tensors"]) for p in params]
    for label in [] if zero_points_only else monos[0]:
        for sign, tag in ((1.0, ""), (-1.0, "-")):
            ok = True
            for v, mono in zip(values, monos):
                got = _f32(v)
                want = sign * mono[label]
                if not (math.isfinite(got) and abs(got - want) <= 2e-6 * abs(want)):
                    ok = False
                    break
            if ok:
                hits.append(f"f32 {tag}{label}")
        for k in range(8, 31):
            ok = True
            for v, mono in zip(values, monos):
                want = round(mono[label] * (1 << k))
                if not (0 < want < (1 << 32)) or v != want:
                    ok = False
                    break
            if ok:
                hits.append(f"round({label} * 2^{k})")
    names = sorted(params[0]["tensors"])
    for name in names:
        zps = [p["tensors"][name][1] for p in params]
        if len(set(zps)) < 2:
            continue
        for part, fn in (
            ("value", lambda v: v),
            ("lo16", lambda v: v & 0xFFFF),
            ("hi16", lambda v: v >> 16),
            ("byte0", lambda v: v & 0xFF),
        ):
            if [fn(v) for v in values] == zps:
                hits.append(f"{part} = zp_{name}")
    return hits


def census(builds: dict[str, dict]) -> dict:
    """Diff one op's builds (``{variant: {"mc": bytes, "params": {...}}}``).

    Returns per-key rows for every register write whose value differs across
    variants (keyed by ``(seg, kind, reg, occurrence)``), each with its values,
    float32 readings and matched formulas; keys present in only some variants;
    and the per-segment content lengths (a length change means the stream
    re-flowed).
    """
    variants = sorted(builds)
    per = {}
    for v in variants:
        per[v] = {
            (w["seg"], w["kind"], w["reg"], w["occurrence"]): w
            for w in register_writes(builds[v]["mc"])
        }

    def order(key):
        seg, kind, reg, occ = key
        return (seg, kind, str(reg) if not isinstance(reg, int) else f"{reg:06x}", occ)

    common = set.intersection(*(set(p) for p in per.values()))
    partial = set.union(*(set(p) for p in per.values())) - common
    rows = []
    params = [builds[v]["params"] for v in variants]
    for key in sorted(common, key=order):
        vals = [per[v][key]["value"] for v in variants]
        widths = [per[v][key]["width"] for v in variants]
        if len(set(vals)) < 2 and len(set(widths)) < 2:
            continue
        seg, kind, reg, occ = key
        rows.append(
            {
                "seg": seg,
                "engine": SEGMENT_ENGINE.get(seg, "?"),
                "kind": kind,
                "reg": _reg_label(reg),
                "occurrence": occ,
                "widths": widths,
                "lengths": [per[v][key]["length"] for v in variants],
                "values": [f"0x{x:08x}" for x in vals],
                "f32": [_f32(x) for x in vals],
                "formulas": match_register(vals, params)
                if min(widths) == 4
                else match_register(vals, params, zero_points_only=True),
            }
        )
    lengths = {
        v: [s["content"] for s in segment_summary(builds[v]["mc"])] for v in variants
    }
    reflow = sorted(
        {
            seg
            for seg in range(len(lengths[variants[0]]))
            if len({lengths[v][seg] for v in variants if seg < len(lengths[v])}) > 1
        }
    )
    return {
        "variants": variants,
        "changed": rows,
        "partial_keys": [
            f"seg{s}:{k}:{_reg_label(r)}#{o}"
            for s, k, r, o in sorted(partial, key=order)
        ],
        "segment_content": lengths,
        "reflowed_segments": reflow,
        "quant_ops": [
            sorted({op for op, _ in builds[v]["params"]["ops"]}) for v in variants
        ],
        "scales": {v: builds[v]["params"]["tensors"] for v in variants},
    }


def _npu_params(model: onnx.ModelProto) -> bytes:
    return next(
        (bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"),
        b"",
    )


def locate(builds: dict[str, dict], tol: float = 2e-6, header_words: int = 4) -> dict:
    """Where each scale monomial is stored, per variant.

    For every monomial of the build's quantization scales (``scale_monomials``)
    this finds every 4-byte value record (``V``/``W``/``S``) whose float32
    equals it within ``tol`` relative, and every uint16 among the first
    ``header_words`` of ``npu_params`` equal to its Q15 encoding
    ``round(m * 32768)``. A match in a variant where another monomial has the
    same value is *ambiguous* there; the per-label ``discriminating`` count
    is the number of variants where it matched and was unique.

    Returns ``{label: {"locations": [...], "found": n, "discriminating": n,
    "calibration_dependent": bool}}`` for labels found in at least one variant.
    """
    variants = sorted(builds)
    per_variant = {}
    for v in variants:
        monos = {
            k: m
            for k, m in scale_monomials(builds[v]["params"]["tensors"]).items()
            if math.isfinite(m) and m != 0.0 and abs(m - 1.0) > 1e-9
        }
        writes = [w for w in register_writes(builds[v]["mc"]) if w["width"] == 4]
        header = builds[v].get("npu_params", b"")[: 2 * header_words]
        words = struct.unpack(f"<{len(header) // 2}H", header) if header else ()
        hits: dict[str, list[str]] = {}
        for label, m in monos.items():
            locs = []
            for w in writes:
                got = _f32(w["value"])
                if math.isfinite(got) and abs(got - m) <= tol * abs(m):
                    locs.append(f"seg{w['seg']}:{w['kind']}:{_reg_label(w['reg'])}")
            if 0 < m < 2:
                q15 = round(m * 32768)
                locs += [
                    f"npu_params[w{i}]:Q15" for i, x in enumerate(words) if x == q15
                ]
            if locs:
                hits[label] = locs
        unique = {
            label
            for label, m in monos.items()
            if not any(
                abs(m - m2) <= tol * abs(m) for k2, m2 in monos.items() if k2 != label
            )
        }
        per_variant[v] = (hits, unique, monos)
    out = {}
    labels = sorted({label for h, _, _ in per_variant.values() for label in h})
    for label in labels:
        locs = sorted({x for v in variants for x in per_variant[v][0].get(label, [])})
        found = [v for v in variants if label in per_variant[v][0]]
        disc = [v for v in found if label in per_variant[v][1]]
        vals = {
            round(per_variant[v][2][label], 9)
            for v in variants
            if label in per_variant[v][2]
        }
        out[label] = {
            "locations": locs,
            "found": len(found),
            "discriminating": len(disc),
            "calibration_dependent": len(vals) > 1,
        }
    return out


def load_build(wd: str) -> dict:
    model = load_axmodel(os.path.join(wd, "out", "compiled.axmodel"))
    quant = onnx.load(
        os.path.join(wd, "out", "quant", "quant_axmodel.onnx"), load_external_data=False
    )
    return {
        "mc": mcode_of(model),
        "npu_params": _npu_params(model),
        "params": quant_params(quant),
    }


def census_root(root: str) -> dict:
    report = {}
    for op, spec in OPS.items():
        builds = {}
        for variant in spec["variants"]:
            wd = os.path.join(root, f"{op}_{variant}")
            if os.path.exists(os.path.join(wd, "out", "compiled.axmodel")):
                builds[variant] = load_build(wd)
        if len(builds) >= 2:
            report[op] = census(builds)
            report[op]["located"] = locate(builds)
            report[op]["mcode_identical_outside_seg0"] = identical_outside(builds, (0,))
    return report


def identical_outside(builds: dict[str, dict], skip_segments=(0,)) -> bool:
    """Whether every variant's mcode is byte-identical to the others once the
    ``skip_segments`` are blanked (segment 0 holds scheduling noise that
    varies between builds even with no quantization at all -- see the doc)."""
    blobs = []
    for b in builds.values():
        mc = bytearray(b["mc"])
        _, segs = mcode.segments(bytes(mc))
        for s in skip_segments:
            pos, length, _ = segs[s]
            mc[pos : pos + length] = bytes(length)
        blobs.append(bytes(mc))
    return len(set(blobs)) == 1


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("build", "census"):
        print(__doc__)
        return 2
    root = argv[1]
    if argv[0] == "build":
        ops = argv[2:] or list(OPS)
        jobs = [(op, v) for op in ops for v in OPS[op]["variants"]]
        with ThreadPoolExecutor(3) as pool:
            for (op, v), rc in zip(jobs, pool.map(lambda j: build_one(root, *j), jobs)):
                print(op, v, "ok" if rc == 0 else f"rc={rc}", flush=True)
        return 0
    report = census_root(root)
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump(report, f, indent=1, default=str)
    for op, r in report.items():
        print(
            f"== {op}: variants {r['variants']}, reflowed segments {r['reflowed_segments']},"
            f" identical outside seg0: {r['mcode_identical_outside_seg0']}"
        )
        for label, d in r["located"].items():
            if not d["calibration_dependent"]:
                continue
            print(
                f"   {label}: found {d['found']}/{len(r['variants'])}"
                f" (discriminating {d['discriminating']}) at {d['locations']}"
            )
        if "--located-only" in argv:
            continue
        for row in r["changed"]:
            print(
                f"   seg{row['seg']}({row['engine']}) {row['kind']} {row['reg']}"
                f"#{row['occurrence']} w{row['widths']} {row['values']}:"
                f" {row['formulas'] or 'UNEXPLAINED'}"
            )
        if r["partial_keys"]:
            print(f"   keys in only some variants: {r['partial_keys'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
