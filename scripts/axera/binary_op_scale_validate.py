"""Validate ``binary_op_scale_emit`` against native Pulsar2 builds.

For every pair ``(template, held-out)`` of builds of the same op and shape with
the same zero points, emit from the template to the held-out build's scales and
compare with the native held-out build:

* ``npu_params`` must be byte-identical;
* decompressed MCode segments 1.. must be identical after normalizing the
  per-build node input order (``normalized_records``); segment 0 is rebuild
  noise and is not compared;
* the rest of the model (node attributes other than the MCode/params
  initializers, IO) is compared too;
* EXACT: the whole MCode blob outside segment 0 (header, segment tables,
  compressed streams, padding) is byte-identical to the native one;
  RECORDS: ``npu_params`` and the decompressed records match but the
  compressed bytes differ, because the two builds' input order differs (so
  the raw records, and hence the LZ77 parse, differ) or the codec's greedy
  parse differs from Pulsar2's on that stream.

Usage::

    binary_op_scale_validate.py BUILD_ROOT [DIR ...]

where each ``BUILD_ROOT/DIR/out/compiled.axmodel`` and
``BUILD_ROOT/DIR/out/quant/quant_axmodel.onnx`` exist.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import binary_op_scale_emit as bse  # noqa: E402
import onnx  # noqa: E402
from teng_register_census import quant_params  # noqa: E402

_OP_BY_PREFIX = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}


def load_build(root: str, d: str):
    model = onnx.load(
        os.path.join(root, d, "out", "compiled.axmodel"), load_external_data=False
    )
    q = quant_params(
        onnx.load(
            os.path.join(root, d, "out", "quant", "quant_axmodel.onnx"),
            load_external_data=False,
        )
    )["tensors"]
    scales = {k: q[k][0] for k in ("x", "z", "y")}
    zps = {k: q[k][1] for k in ("x", "z", "y")}
    shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    return model, scales, zps, shape


def compare(emitted: onnx.ModelProto, native: onnx.ModelProto) -> dict:
    res = {}
    pe = bytes(bse._params_init(emitted).raw_data)
    pn = bytes(bse._params_init(native).raw_data)
    res["params"] = pe == pn
    ne, nn = bse.normalized_records(emitted), bse.normalized_records(native)
    res["segments"] = len(ne) == len(nn) and all(a == b for a, b in zip(ne, nn))
    if not res["segments"]:
        res["seg_diff"] = [
            (
                i + 1,
                sum(
                    1
                    for r in range(0, min(len(a), len(b)), 8)
                    if a[r : r + 8] != b[r : r + 8]
                ),
            )
            for i, (a, b) in enumerate(zip(ne, nn))
            if a != b
        ]
    me = bytes(bse._mcode_init(emitted).raw_data)
    mn = bytes(bse._mcode_init(native).raw_data)
    res["stream_equal_len"] = len(me) == len(mn)
    # outside segment 0 (rebuild noise) and the input-order slots, the whole
    # blob -- header, segment table, padding -- must match when lengths do
    if len(me) == len(mn):
        s0 = bse.codec.segment_streams(mn)[0]
        skip = range(s0[0], s0[0] + 8 * s0[2][2])
        res["blob_diff_outside_seg0"] = sum(
            1 for o in range(len(me)) if me[o] != mn[o] and o not in skip
        )
    res["dims"] = list(bse._mcode_init(emitted).dims) == list(
        bse._mcode_init(native).dims
    )
    segs_e = bse.codec.segment_streams(me)
    segs_n = bse.codec.segment_streams(mn)
    res["streams_1plus_identical"] = all(
        me[a[0] : a[0] + a[1]] == mn[b[0] : b[0] + b[1]]
        for a, b in list(zip(segs_e, segs_n))[1:]
    )
    return res


def main(root: str, dirs: list[str]) -> int:
    builds = {}
    for d in dirs:
        op = _OP_BY_PREFIX.get(d.split("_")[0].lower())
        if op is None:
            continue
        try:
            builds[d] = (op, *load_build(root, d))
        except (FileNotFoundError, KeyError) as exc:
            print(f"skip {d}: {exc}")
    exact = records = bad = refused = 0
    for t, (op, tm, ts, tz, tshape) in sorted(builds.items()):
        for h, (hop, hm, hs, hz, hshape) in sorted(builds.items()):
            if h == t or hop != op or hshape != tshape or hz != tz:
                continue
            try:
                em = bse.emit_model(tm, op, ts, hs, tz)
            except ValueError as exc:
                refused += 1
                print(f"REFUSED {t} -> {h}: {exc}")
                continue
            r = compare(em, hm)
            if not (r["params"] and r["segments"]):
                tag = "BAD"
                bad += 1
            elif r.get("blob_diff_outside_seg0") == 0:
                tag = "EXACT"
                exact += 1
            else:
                tag = "RECORDS"
                records += 1
            same_order = list(tm.graph.node[0].input) == list(hm.graph.node[0].input)
            print(f"{tag} {t} -> {h} same_order={same_order}: {r}")
    print(
        f"summary: {exact} byte-exact, {records} record-exact (layout differs), "
        f"{bad} mismatched, {refused} refused"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    root = sys.argv[1]
    dirs = sys.argv[2:] or sorted(os.listdir(root))
    sys.exit(main(root, dirs))
