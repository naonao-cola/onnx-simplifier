#!/usr/bin/env python3
"""Patch recalibrated scales into a compiled `.axmodel` instead of rebuilding it.

A training phase swap recompiles the step graph with new calibration data,
but a rebuild spends ~60 s mostly re-running the NPU backend scheduler --
which does not depend on the calibration values at all. What actually varies
with calibration is small: a handful of scale slots in the mcode (bfloat16
reciprocals, bfloat16/float32 direct scales), verified by building the same
graph at 1x/2x/4x calibration amplitude and matching every moving byte
against the quant tables. Everything else that moves is scheduler noise,
ordering, or structural divergence no value patch can reach.

So instead of rebuilding per phase: compute the new table (plain MinMax
over the new calibration data -- what calibration *is*), write the new
encodings over the old ones in place, and reload. On an attention probe
this reproduces the rebuilt artifact's replay numerics bit-exactly
(322 dB patched-vs-rebuilt; both 26.71 dB vs float), with `mcode.check()`
clean. See the README's "Patching calibration instead of recompiling"
section for the full evidence, the slot map, and what is explicitly *not*
covered (weight slots for changed weights -- those travel in the resident
state files already -- and the scheduling-divergent bytes, which no value
edit can reach).

What this does *not* replace is a card run: static checks (form, replay)
cannot see an address that happens to share a scale's bits. Every patched
offset and old/new value is reported, so a reviewer (or a first card run)
can audit exactly what moved. Do not train on a patched artifact that has
not run cleanly on device at least once per graph shape.

Usage::

    patch_scales.py base.axmodel --old-table old/quant/quant_axmodel.json \\
        --new-table new/quant/quant_axmodel.json -o patched.axmodel
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def bf16_trunc(x: float) -> int:
    """`x` as bfloat16 bits (top 16 of float32, truncated toward zero --
    exactly how the compiler stores reciprocal scales)."""
    return struct.unpack("<H", struct.pack("<f", np.float32(x))[2:])[0]


def table_scales(table_path: str) -> Dict[str, float]:
    """`{tensor: scale}` for every scaled entry of a quant table."""
    with open(table_path) as f:
        doc = json.load(f)
    out = {}
    for cfg in doc["tensor_configs"].values():
        for tensor, entry in cfg.items():
            value = doc["values"].get(str(entry.get("hash")))
            if value and value.get("scale"):
                out.setdefault(tensor, float(value["scale"][0]))
    return out


def mcode_bytes(model: onnx.ModelProto) -> Tuple[bytes, str]:
    """An `.axmodel`'s mcode blob and its initializer name."""
    info = json.loads(
        next(
            a
            for n in model.graph.node
            if n.op_type == "neu mode"
            for a in n.attribute
            if a.name == "npu_graph_info"
        ).s.decode()
    )
    key = info["dotneus"][0]["neu_key"]
    init = next(i for i in model.graph.initializer if i.name == key)
    return bytes(numpy_helper.to_array(init).tobytes()), key


def find_scale_slots(
    blob: bytes, old: Dict[str, float], new: Dict[str, float]
) -> List[Tuple[int, str, str, bytes, bytes]]:
    """`(offset, tensor, encoding, old_bytes, new_bytes)` for every slot.

    A slot is a position whose current bytes encode the OLD table value
    (bfloat16 direct or reciprocal, or float32 direct) for a tensor whose
    scale CHANGED. Unchanged scales are never touched, so coincidentally
    matching constants (the 127.5 dequant midpoint, allocator output) are
    safe by construction. Overlapping matches keep the widest (a float32
    slot contains its own bfloat16 truncation as nested bytes); matches
    claimed by two different tensors are ambiguous and dropped, not
    guessed.
    """
    hits: Dict[int, Tuple[int, str, str, bytes, bytes]] = {}
    claimed: Dict[int, set] = {}
    for tensor, old_scale in old.items():
        new_scale = new.get(tensor)
        if new_scale is None or new_scale == old_scale:
            continue
        cands = [
            (
                "bf16",
                struct.pack("<H", bf16_trunc(old_scale)),
                struct.pack("<H", bf16_trunc(new_scale)),
            ),
            (
                "bf16recip",
                struct.pack("<H", bf16_trunc(1 / old_scale)),
                struct.pack("<H", bf16_trunc(1 / new_scale)),
            ),
            (
                "f32",
                struct.pack("<f", np.float32(old_scale)),
                struct.pack("<f", np.float32(new_scale)),
            ),
        ]
        for encoding, old_b, new_b in cands:
            start = 0
            while True:
                at = blob.find(old_b, start)
                if at < 0:
                    break
                claimed.setdefault(at, set()).add((tensor, new_b))
                prev = hits.get(at)
                if prev is None or len(old_b) > len(prev[3]):
                    hits[at] = (at, tensor, encoding, old_b, new_b)
                start = at + 1
    # A float32 slot contains its own bfloat16 truncation as nested bytes;
    # both match, but only the widest is a slot. Greedily keep widest-first
    # so nested matches never double-patch. Differently-tensored matches at
    # one offset are ambiguous (two scales, one byte pattern) and dropped.
    ordered = sorted(hits.values(), key=lambda h: (-len(h[3]), h[0]))
    kept: List[Tuple[int, str, str, bytes, bytes]] = []
    for hit in ordered:
        at, tensor, _, old_b, new_b = hit
        if len({t for t, nb in claimed[at]}) > 1:
            continue
        if all(at + len(old_b) <= k or k + len(b) <= at for k, _, _, b, _ in kept):
            kept.append(hit)
    return sorted(kept)


def patch_model(
    model_path: str, old_table: str, new_table: str, output_path: str
) -> List[Tuple[int, str, str, bytes, bytes]]:
    """Write `new_table`'s scales over `old_table`'s slots in `model_path`.

    Returns the patched slots for the audit trail. Raises `ValueError` if
    no slot matched (nothing to patch usually means the tables did not
    actually change).
    """
    model = onnx.load(model_path)
    blob, key = mcode_bytes(model)
    old, new = table_scales(old_table), table_scales(new_table)
    slots = find_scale_slots(blob, old, new)
    if not slots:
        raise ValueError("no scale slots matched; tables may be identical")
    patched = bytearray(blob)
    for offset, _, _, old_b, new_b in slots:
        assert bytes(patched[offset : offset + len(old_b)]) == old_b
        patched[offset : offset + len(old_b)] = new_b
    init = next(i for i in model.graph.initializer if i.name == key)
    init.CopyFrom(
        numpy_helper.from_array(
            np.frombuffer(bytes(patched), dtype=np.uint16).copy(), key
        )
    )
    onnx.save(model, output_path)
    return slots


def verify_against_rebuild(
    patched_model: str, rebuild_model: str
) -> List[Tuple[int, bytes, bytes]]:
    """Byte diffs between a patched artifact and a real rebuild of the new
    table, as `(offset, patched, rebuilt)`. Empty means the patch
    reproduced the rebuild byte-exactly. Anything listed is either
    scheduler noise/ordering (compare against an identical-input rebuild
    pair to tell) or a slot the patch missed -- both worth knowing before
    trusting patched artifacts for training. Needs the rebuild, so this is
    for qualifying a new graph shape once, not for every phase."""
    patched, _ = mcode_bytes(onnx.load(patched_model))
    rebuilt, _ = mcode_bytes(onnx.load(rebuild_model))
    assert len(patched) == len(rebuilt), (len(patched), len(rebuilt))
    return [
        (i, patched[i : i + 1], rebuilt[i : i + 1])
        for i in range(len(patched))
        if patched[i] != rebuilt[i]
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("model", help="compiled .axmodel from the old table")
    argp.add_argument("--old-table", required=True)
    argp.add_argument("--new-table", required=True)
    argp.add_argument("-o", "--output", required=True)
    argp.add_argument(
        "--verify",
        metavar="REBUILT",
        default=None,
        help="a real rebuild from the new table to diff against",
    )
    args = argp.parse_args(argv)

    import mcode

    slots = patch_model(args.model, args.old_table, args.new_table, args.output)
    patched = mcode.mcodes_of(args.output)[0][1]
    bad = mcode.check(patched)
    print(f"patched {len(slots)} slots:")
    for offset, tensor, encoding, old_b, new_b in slots:
        print(f"  @{offset}: {tensor} {encoding} {old_b.hex()} -> {new_b.hex()}")
    print("check:", bad if bad else "clean")
    if args.verify is not None:
        diffs = verify_against_rebuild(args.output, args.verify)
        print(f"vs rebuild: {len(diffs)} differing bytes")
        for offset, was, is_now in diffs[:20]:
            print(f"  @{offset}: {was.hex()} -> {is_now.hex()}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
