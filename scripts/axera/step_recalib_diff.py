"""Classify how a calibration change alters a compiled training step, to decide
whether the step can be recalibrated by patching instead of recompiling.

Build the same step graph twice with different calibration data and hand the
two ``.axmodel`` files to :func:`classify`. It compares the MCode engine queue
by engine queue (``mcode.segments``) at the record level (``mcode.decode``),
not byte by byte, and diffs ``npu_params``:

* A segment is **unchanged** when its bytes match outside the rebuild-noise
  window, which is always the tail of segment 0 (``noise_bytes``).
* Inside a changed segment, records are aligned by *shape* (kind + register
  address), ignoring values. Aligned records whose 4-byte value changed are
  **literal changes**; with each build's recorded scales
  (``quant_axmodel.json``) a literal is **explained** when both old and new
  values are a scale, its reciprocal, a ratio, or a product of two scales.
* Anything the alignment cannot pair up is a **structural** change: a record
  appeared, vanished, or changed form. On this project's builds these are
  value-dependent re-encodings (a compressed register write whose byte count
  depends on the value) and range-dependent integer tables whose entry count
  changes -- see ``docs/axera-step-recalibrate.md``.

The verdict is ``equivalent`` (noise only), ``patchable`` (every change is a
fixed-width literal explained by the scales), or ``not-patchable`` with the
reasons. This is an analysis tool: it does not patch anything, because no
real calibration change measured so far leaves the compute segment's encoding
intact.

Usage::

    step_recalib_diff.py REF.axmodel[.gz] NEW.axmodel[.gz] [REF_SCALES.json NEW_SCALES.json]

where a scales JSON is ``{"scales": [...]}`` or a Pulsar2 ``quant_axmodel.json``.
"""

from __future__ import annotations

import bisect
import collections
import difflib
import gzip
import itertools
import json
import os
import struct
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

NOISE_TAIL = 64
"""Identical rebuilds differ only inside the last ``NOISE_TAIL`` bytes of
segment 0 (bytes 301-325 of a single-op model, whose segment 0 is 64 bytes;
3124-3140 of the toy step; 25830-25851 of the ResNet18 head step)."""


def noise_bytes(mc: bytes) -> range:
    """The rebuild-noise byte range of one mcode stream: segment 0's tail."""
    _, segs = mcode.segments(mc)
    pos, length, _ = segs[0]
    return range(max(pos, pos + length - NOISE_TAIL), pos + length)


def load_step(path: str) -> tuple[dict[str, bytes], bytes]:
    """``({neu_key: mcode}, npu_params)`` from an ``.axmodel`` (optionally gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    inits = {i.name: bytes(i.raw_data) for i in model.graph.initializer}
    mcodes = {k: v for k, v in inits.items() if k.endswith("_neu")}
    return mcodes, inits.get("npu_params", b"")


def load_scales(path: str) -> list[float]:
    """Positive per-tensor scales from ``{"scales": [...]}`` or a ``quant_axmodel.json``."""
    with open(path) as f:
        data = json.load(f)
    if "scales" in data:
        return sorted(data["scales"])
    return sorted(
        {
            v["scale"][0]
            for v in data["values"].values()
            if v.get("scale") and v["scale"][0] > 0
        }
    )


def segment_records(mc: bytes, seg: int) -> list[dict]:
    _, segs = mcode.segments(mc)
    pos, length, _ = segs[seg]
    return mcode.decode(mc, start=pos, end=pos + length, **mcode.FULL_RULE)


def record_key(r: dict) -> tuple:
    """A record's shape, ignoring its value: kind plus register/tag."""
    kind = r["kind"]
    if kind in ("V", "W"):
        return (kind, r["field"] | (r["bank"] << 8))
    if kind == "S":
        return (kind, r["tag"], r["p"])
    if kind == "B":
        return (kind, r["tag"])
    return (kind,)


def _value(r: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in r.items() if k != "at"))


def literal(r: dict) -> bytes | None:
    """The 4-byte value a record carries (a register-write operand or a 4-byte
    short-unit payload), or ``None``."""
    if r["kind"] in ("V", "W") and len(r["operand"]) == 4:
        return bytes(r["operand"])
    if r["kind"] == "S" and len(r["payload"]) == 4:
        return bytes(r["payload"])
    return None


def compare_segment(a: bytes, b: bytes, seg: int) -> dict:
    """Record-level comparison of one engine queue across two builds."""
    _, sa = mcode.segments(a)
    _, sb = mcode.segments(b)
    pa, la, _ = sa[seg]
    pb, lb, _ = sb[seg]
    noise = noise_bytes(a)
    # Compare each segment at its own offset: an earlier segment growing
    # shifts every later one without changing its content.
    if la == lb and all(
        a[pa + i] == b[pb + i] for i in range(la) if pa + i not in noise
    ):
        return {"changed": False, "size": (la, lb)}
    ra, rb = segment_records(a, seg), segment_records(b, seg)
    matcher = difflib.SequenceMatcher(
        None, [record_key(r) for r in ra], [record_key(r) for r in rb], autojunk=False
    )
    structural = []
    value_diffs = collections.Counter()
    literals = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            structural.append((tag, i2 - i1, j2 - j1))
            continue
        for x, y in zip(ra[i1:i2], rb[j1:j2]):
            if x["at"] in noise:
                continue
            if _value(x) == _value(y):
                continue
            value_diffs[x["kind"]] += 1
            lx, ly = literal(x), literal(y)
            if lx is not None and ly is not None and lx != ly:
                literals.append((x["at"], lx, ly))
    return {
        "changed": True,
        "size": (la, lb),
        "records": (len(ra), len(rb)),
        "structural": structural,
        "value_diffs": dict(value_diffs),
        "literals": literals,
    }


def scale_functions(scales: list[float]) -> list[float]:
    """Every scale, reciprocal, pairwise ratio, product and reciprocal product."""
    out = []
    for s in scales:
        out += [s, 1.0 / s]
    for x, y in itertools.permutations(scales, 2):
        out.append(x / y)
    for x, y in itertools.combinations_with_replacement(scales, 2):
        out += [x * y, 1.0 / (x * y)]
    return sorted(out)


def _explained(value: float, functions: list[float], tol: float = 2e-6) -> bool:
    if not 1e-12 < abs(value) < 1e12:
        return False
    i = bisect.bisect_left(functions, value)
    return any(
        0 <= j < len(functions) and abs(functions[j] - value) <= tol * abs(value)
        for j in (i - 1, i)
    )


def explain_literals(literals, scales_a, scales_b) -> tuple[int, int]:
    """``(explained, total)``: literals whose old and new float values are both
    scale functions of their own build."""
    fa, fb = scale_functions(scales_a), scale_functions(scales_b)
    hits = 0
    for _, la, lb in literals:
        va = struct.unpack("<f", la)[0]
        vb = struct.unpack("<f", lb)[0]
        hits += _explained(va, fa) and _explained(vb, fb)
    return hits, len(literals)


def params_runs(a: bytes, b: bytes, gap: int = 4) -> list[tuple[int, int]]:
    """``(start, length)`` runs of differing ``npu_params`` bytes (equal lengths only)."""
    runs: list[list[int]] = []
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            if runs and i <= runs[-1][1] + gap:
                runs[-1][1] = i
            else:
                runs.append([i, i])
    return [(s, e - s + 1) for s, e in runs]


def classify(ref: str, new: str, scales_ref=None, scales_new=None) -> dict:
    """Compare two builds of one step that differ only in calibration."""
    ma, pa = load_step(ref)
    mb, pb = load_step(new)
    reasons = []
    segments = {}
    for key in sorted(ma):
        a, b = ma[key], mb[key]
        _, sa = mcode.segments(a)
        for seg in range(len(sa)):
            c = compare_segment(a, b, seg)
            segments[(key, seg)] = c
            if not c["changed"]:
                continue
            if c["size"][0] != c["size"][1]:
                reasons.append(
                    f"{key} segment {seg} changed size {c['size'][0]} -> {c['size'][1]}"
                )
            if c["structural"]:
                reasons.append(
                    f"{key} segment {seg}: {len(c['structural'])} structural changes "
                    f"(records {c['records'][0]} -> {c['records'][1]})"
                )
            if scales_ref is not None and scales_new is not None:
                hit, tot = explain_literals(c["literals"], scales_ref, scales_new)
                c["explained"] = (hit, tot)
                if hit < tot:
                    reasons.append(
                        f"{key} segment {seg}: {tot - hit} of {tot} changed literals not scale functions"
                    )
            other = sum(c["value_diffs"].values()) - len(c["literals"])
            if other:
                reasons.append(
                    f"{key} segment {seg}: {other} changed non-literal records"
                )
    runs = params_runs(pa, pb) if len(pa) == len(pb) else None
    if runs is None:
        reasons.append(f"npu_params changed size {len(pa)} -> {len(pb)}")
    changed = [k for k, c in segments.items() if c["changed"]]
    if not changed and not runs and runs is not None:
        verdict = "equivalent"
    elif not reasons:
        verdict = "patchable"
    else:
        verdict = "not-patchable"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "changed_segments": changed,
        "segments": segments,
        "npu_params_runs": runs,
    }


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 4):
        print(__doc__)
        return 2
    sr = load_scales(argv[2]) if len(argv) == 4 else None
    sn = load_scales(argv[3]) if len(argv) == 4 else None
    result = classify(argv[0], argv[1], sr, sn)
    for (key, seg), c in result["segments"].items():
        if not c["changed"]:
            print(f"{key} seg{seg}: unchanged ({c['size'][0]} B)")
            continue
        line = (
            f"{key} seg{seg}: size {c['size'][0]} -> {c['size'][1]}, records "
            f"{c['records'][0]} -> {c['records'][1]}, structural {len(c['structural'])}, "
            f"value diffs {c['value_diffs']}"
        )
        if "explained" in c:
            line += f", literals explained {c['explained'][0]}/{c['explained'][1]}"
        print(line)
    print(f"npu_params differing runs: {result['npu_params_runs']}")
    print(f"verdict: {result['verdict']}")
    for r in result["reasons"]:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
