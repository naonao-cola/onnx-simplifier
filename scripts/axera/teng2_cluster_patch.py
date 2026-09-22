"""Extrapolate a third member of a `teng2` (segment-2 compute microprogram)
template cluster from two adjacent, already-compiled references.

`docs/axera-dma-queue.md` found that segment 2 of a tiled elementwise op's
MCode -- the compute program `teng2`, as opposed to the DMA-load tile table
`dma_tile_predict.py` already decodes -- does not reduce to a per-shape
template: consecutive channel counts are, in general, unrelated programs.
`docs/axera-teng2-sqrt-blocks.md` narrows that finding. It is not quite
true: **some** runs of 2-3 consecutive channel counts (for a fixed batch,
op, and spatial size) do share a template, differing only at a handful of
little-endian integer fields (most 1-2 bytes, contiguous in the blob) whose
value is an exact multiple of the channel-count step -- the same small set
of per-step deltas (1, 14, 28, 49 across the measured clusters) recur. But **which** channel counts fall in the same cluster
is not decoded: two shapes that share every other observable property
(tile-table entry count, segment-2 length) can still be in different
clusters (`C=163` and `C=164` in the measured data, both `entries=4,
len(seg2)=1344`, differ in >99% of segment 2's bytes). So this module
cannot predict cluster membership -- it only *extrapolates*, from two
references the caller has already confirmed are in the same cluster (by
compiling them and observing a small, uniform-looking diff), to the next
integer in that arithmetic sequence.

The differing fields are not confined to segment 2 (there are also a few
in segments 3/4 and the loader tail, matching what
`docs/axera-dma-queue.md`'s Relu sweep found for those regions), so this
scans the whole MCode blob outside the known 301-325 compiler-noise
window, not just segment 2. Differing bytes are merged into contiguous
little-endian fields before computing a delta, so a byte that only changes
because a lower byte carried into it (an early version of this module
missed exactly this, at position 1265 of the 153/154/155 cluster) is
handled correctly.

**Not always exact.** Checked byte-for-byte against Pulsar2's own build for
three held-out targets: `extrapolate(ref157, ref158, target_c=159)` is
exact. `extrapolate(ref153, ref154, target_c=155)` and
`extrapolate(ref165, ref166, target_c=167)` are not -- a small number of
bytes (2 and 20, respectively, out of ~3000) differ, because a field's
value carries between the target and the nearer reference in a way the
two references given did not exhibit between themselves (a byte that is
identical between `ref_a` and `ref_b` can still be wrong at the target if
its lower byte overflows exactly between `ref_b` and the target). This is
a real, open limitation, not a rounding artifact: see
`docs/axera-teng2-sqrt-blocks.md` for the full accounting and what would
be needed to close it (a third reference, or the fields' actual semantics
rather than just their slopes). `npu_params` was exact in all three cases
(copied unchanged from the nearer reference -- valid because every
measured cluster shares one tile-table entry, i.e. the same
`ceil(C/entries)`).
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

_NOISE_START = 301
_NOISE_END = 326


def _shape(model: onnx.ModelProto) -> list[int]:
    return [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]


def _mcode_init(model: onnx.ModelProto):
    return next(i for i in model.graph.initializer if i.name.endswith("_neu"))


def _fields(blob_a: bytes, blob_b: bytes) -> list[tuple[int, int, int]]:
    """``(start, width, delta)`` for each contiguous run of differing bytes,
    outside the noise window. ``delta`` is ``b``'s little-endian value minus
    ``a``'s, as an unbounded Python int (not wrapped -- the caller decides how
    to re-encode it, since a target further from ``a``/``b`` than one step
    would need the wraparound a single step might not exhibit).
    """
    diffs = [
        i
        for i in range(len(blob_a))
        if blob_a[i] != blob_b[i] and not (_NOISE_START <= i < _NOISE_END)
    ]
    runs: list[list[int]] = []
    for i in diffs:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    fields = []
    for run in runs:
        start, width = run[0], len(run)
        va = int.from_bytes(blob_a[start : start + width], "little")
        vb = int.from_bytes(blob_b[start : start + width], "little")
        fields.append((start, width, vb - va))
    return fields


def cluster_positions(ref_a_path: str, ref_b_path: str) -> list[tuple[int, int, int]]:
    """The differing little-endian fields between ``ref_a`` and ``ref_b``'s MCode,
    as ``(byte_offset, width, delta)``, scanning the whole blob outside the noise
    window (see the module docstring for why this is not segment-2-only).

    Both references must be the same op/shape family, one channel apart.
    Raises ``ValueError`` if the channel counts are not exactly 1 apart, if the
    shapes disagree anywhere but channel count, or if the MCode lengths differ
    (a sure sign they are not the same template).
    """
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
    return _fields(mcode_a, mcode_b)


def extrapolate(
    ref_a_path: str, ref_b_path: str, target_c: int, output_path: str
) -> str:
    """Emit the next (or previous) member of ``ref_a``/``ref_b``'s cluster.

    ``target_c`` must be ``ref_b``'s channel count plus the same step (``ref_b -
    ref_a``, so the next integer after ``ref_b``) or ``ref_a``'s channel count
    minus that step (the one before ``ref_a``) -- this is extrapolation of an
    already-observed arithmetic sequence, not a claim about any other channel
    count. ``npu_params`` is copied unchanged from the nearer reference, which
    is only correct when ``target_c`` shares its tile-table entry; there is no
    general entry-count predictor to check that against here (the ordinary
    4/8/16/32 doubling rule in ``dma_tile_predict.py`` does not cover every
    entry count this module's own doc measured, e.g. ``entries=7``), so this
    is a real, undischarged assumption the caller should be aware of.
    """
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
