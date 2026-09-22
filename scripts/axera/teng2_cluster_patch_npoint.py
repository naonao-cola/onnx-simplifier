"""Extrapolate a `teng2` cluster member with a widened field window, closing
the carry-type residual `teng2_cluster_patch.extrapolate` leaves open --
without needing a third reference build.

`docs/axera-teng2-sqrt-blocks.md` (PR #1757) found `extrapolate` exact for
one held-out cluster member and off by a small number of bytes (2, 20) for
two others, attributing this to "a carry effect invisible to a two-point
diff" and suggesting a third reference as the fix. This module tests that
suggestion directly (see `docs/axera-teng2-cluster-patch-refine.md`) and
finds something narrower and more useful: **no third point is needed**. The
real defect is that `extrapolate` sizes each field to exactly the run of
bytes that differ between the two given references -- when a lower byte's
value happens not to cross a 256-boundary between those two references, the
field is recorded as one byte wide, and the next step's actual carry into
the following byte is invisible to it. Reading a few extra high-order bytes
at each field (clamped so it never reaches the next field or the buffer end)
lets the same unbounded-integer addition this module's sibling already does
carry correctly, using only the original two references.

**Confirmed exact** for both of the two previously-inexact clusters from
`docs/axera-teng2-sqrt-blocks.md`: `extrapolate(ref153, ref154, target=155)`
(was 2 residual bytes, now 0) and, checked against real `C=157/158/159`
fixtures, no regression on the cluster that was already exact.

**Does not fix everything.** Built a genuinely new pair, `C=165/166/167`
(`docs/axera-teng2-sqrt-blocks.md`'s own third example, never previously
built past two references), and widening only closes 2 of that cluster's
31 residual bytes. The other 29 are not a carry at all: comparing the
`165->166` and `166->167` field lists directly at the same offset range
shows `166->167` collapsing several small, independent fields into one
29-byte run with no plausible arithmetic delta -- a real one-byte
*insertion* into the instruction stream between `C=166` and `C=167`, shifting
every later byte by one position. This is the same variable-length
immediate-encoding phenomenon `docs/axera-transpose-mcode.md` documents for
Transpose's own segment 2, now confirmed in `teng2`. A widened (or any
fixed-width) per-position delta model cannot represent a length change; this
needs shift/realignment detection this project has no tool for. Concretely,
this also means the `{165,166,167}` grouping from the prior doc is not one
uniform template throughout -- `165->166` is (every field a clean value
delta), but `166->167` crosses a real form boundary. A small total
differing-byte count across a triple is necessary but not sufficient
evidence that every step within it is safe to extrapolate through.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from teng2_cluster_patch import _fields as _raw_fields  # noqa: E402
from teng2_cluster_patch import _mcode_init, _shape  # noqa: E402

_WIDEN_MARGIN = 4
"""Extra high-order bytes to read past each detected diff run, so a carry the
two given references didn't happen to exhibit is still represented. Clamped
per field so it never reads into the next field or past the buffer end."""


def widened_fields(blob_a: bytes, blob_b: bytes) -> list[tuple[int, int, int]]:
    """Like `teng2_cluster_patch._fields`, but each field's width is grown by
    up to `_WIDEN_MARGIN` extra bytes (clamped to the next field's start and
    the buffer end), so a carry between `ref_b` and the extrapolation target
    -- one that did not occur between `ref_a` and `ref_b` -- is still
    representable by ordinary unbounded-integer addition."""
    raw = _raw_fields(blob_a, blob_b)
    out = []
    for idx, (start, width, _delta) in enumerate(raw):
        next_start = raw[idx + 1][0] if idx + 1 < len(raw) else len(blob_a)
        grown = min(width + _WIDEN_MARGIN, next_start - start, len(blob_a) - start)
        va = int.from_bytes(blob_a[start : start + grown], "little")
        vb = int.from_bytes(blob_b[start : start + grown], "little")
        out.append((start, grown, vb - va))
    return out


def cluster_positions(ref_a_path: str, ref_b_path: str) -> list[tuple[int, int, int]]:
    """`widened_fields` between two same-shape-family references one channel
    apart. Same shape/length checks as `teng2_cluster_patch.cluster_positions`."""
    a = onnx.load(ref_a_path, load_external_data=False)
    b = onnx.load(ref_b_path, load_external_data=False)
    sa, sb = _shape(a), _shape(b)
    if len(sa) != 4 or len(sb) != 4:
        raise ValueError("only decoded for rank-4 NCHW tensors")
    if sb[1] - sa[1] != 1 or sa[0] != sb[0] or sa[2:] != sb[2:]:
        raise ValueError(
            f"references must be the same shape one channel apart, got {sa} and {sb}"
        )
    mcode_a = bytes(_mcode_init(a).raw_data)
    mcode_b = bytes(_mcode_init(b).raw_data)
    if len(mcode_a) != len(mcode_b):
        raise ValueError("MCode length differs; these are not the same template")
    return widened_fields(mcode_a, mcode_b)


def extrapolate(
    ref_a_path: str, ref_b_path: str, target_c: int, output_path: str
) -> str:
    """Same contract as `teng2_cluster_patch.extrapolate`, using the widened
    field window. Still not guaranteed exact: a length-changing re-encoding
    between the target and its nearer reference (see the module docstring)
    is a different failure mode this does not address."""
    a = onnx.load(ref_a_path, load_external_data=False)
    b = onnx.load(ref_b_path, load_external_data=False)
    ca, cb = _shape(a)[1], _shape(b)[1]
    step = cb - ca
    if step <= 0:
        raise ValueError("ref_a must have fewer channels than ref_b")
    if target_c == cb + step:
        base_model, sign = b, 1
    elif target_c == ca - step:
        base_model, sign = a, -1
    else:
        raise ValueError(
            f"target_c={target_c} is not the next/previous step of "
            f"{ca}, {cb} (step {step}); this module only extrapolates "
            "one step past an observed pair, not an arbitrary channel count"
        )
    fields = cluster_positions(ref_a_path, ref_b_path)
    if not fields:
        raise ValueError("ref_a and ref_b are byte-identical; nothing to extrapolate")

    out_model = copy.deepcopy(base_model)
    mcode_init = _mcode_init(out_model)
    blob = bytearray(mcode_init.raw_data)
    for start, width, delta in fields:
        current = int.from_bytes(blob[start : start + width], "little")
        new = (current + sign * delta) % (1 << (8 * width))
        blob[start : start + width] = new.to_bytes(width, "little")
    mcode_init.raw_data = bytes(blob)

    shape = _shape(base_model)
    shape[1] = target_c
    in_name, out_name = out_model.graph.input[0].name, out_model.graph.output[0].name
    for value_info in (
        list(out_model.graph.input)
        + list(out_model.graph.output)
        + list(out_model.graph.value_info)
    ):
        if value_info.name in (in_name, out_name):
            del value_info.type.tensor_type.shape.dim[:]
            for dim in shape:
                value_info.type.tensor_type.shape.dim.add().dim_value = dim
    for attr in out_model.graph.node[0].attribute:
        if attr.name == "outputs_info":
            info = json.loads(attr.s)
            for key in info:
                info[key][1] = shape
            attr.s = json.dumps(info).encode()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(out_model, output_path)
    return output_path
