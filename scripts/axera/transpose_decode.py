"""Decode-side analysis of the standalone AX650 ``Transpose`` sweep.

Reads builds made by ``transpose_sweep.py`` (``A_1x1x<R>x<C>`` directories under
a work root) and reproduces the tables in ``docs/axera-transpose-mcode.md``'s
"Third pass" section: the size immediate by alignment class, and the block
structure of the unaligned-C family at R=16. Characterization only; nothing here
predicts or emits MCode.

Usage::

    transpose_decode.py size-field WORK_ROOT     # 16-bit size immediate by (C, R) alignment
    transpose_decode.py blocks WORK_ROOT [R]     # varying bytes per block of 8 unaligned C
"""

from __future__ import annotations

import collections
import os
import re
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_NAME = re.compile(r"^A_1x1x(\d+)x(\d+)$")


def segment2(root: str, r: int, c: int) -> bytes | None:
    """Segment 2 content (trailing zero padding stripped) of an untiled build.

    Returns None if the build is missing or tiled (segments 1 and 3 grew past 32 B).
    """
    path = os.path.join(root, f"A_1x1x{r}x{c}", "out", "compiled.axmodel")
    if not os.path.exists(path):
        return None
    model = onnx.load(path, load_external_data=False)
    blob = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )
    segs = mcode.segments(blob)[1]
    if len(segs) != 5 or segs[1][1] != 32 or segs[3][1] != 32:
        return None
    start, size = segs[2][0], segs[2][1]
    body = blob[start : start + size]
    return body[: max((i for i, v in enumerate(body) if v), default=-1) + 1]


def shapes(root: str) -> list[tuple[int, int]]:
    found = set()
    for name in os.listdir(root):
        match = _NAME.match(name)
        if match:
            found.add((int(match.group(1)), int(match.group(2))))
    return sorted(found)


def size_field_table(root: str) -> dict:
    """Count shapes whose stream holds 4*R*C-1 or 4*R*C as a little-endian uint16."""
    table: dict = collections.defaultdict(collections.Counter)
    for r, c in shapes(root):
        body = segment2(root, r, c)
        if body is None:
            continue
        cls = (
            "C aligned" if c % 8 == 0 else "C unaligned",
            "R aligned" if r % 8 == 0 else "R unaligned",
        )
        for label, value in (("4RC-1", 4 * r * c - 1), ("4RC", 4 * r * c)):
            if value >= 65536:
                table[(cls, label)][">=64K (not tested)"] += 1
            else:
                hit = bytes([value & 255, value >> 8]) in body
                table[(cls, label)]["present" if hit else "absent"] += 1
    return table


def block_table(root: str, r: int = 16) -> None:
    """For each block C=8k+1..8k+7, print the bytes that vary inside the block."""
    for start in range(9, 65, 8):
        cs = list(range(start, start + 7))
        streams = {c: segment2(root, r, c) for c in cs}
        if any(s is None or len(s) != len(streams[cs[0]]) for s in streams.values()):
            print(f"block C={start}..{start + 6}: missing or unequal lengths")
            continue
        pos = [
            i
            for i in range(len(streams[cs[0]]))
            if len({streams[c][i] for c in cs}) > 1
        ]
        print(f"block C={start}..{start + 6}: {len(pos)} varying positions")
        if len(pos) <= 4:
            for c in cs:
                got = bytes(streams[c][p] for p in pos)
                print(f"  C={c:3d} bytes={got.hex(' ')} (4RC-1 = {4 * r * c - 1})")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("size-field", "blocks"):
        print(__doc__)
        return 2
    if argv[0] == "size-field":
        for key, counts in sorted(size_field_table(argv[1]).items()):
            print(key, dict(counts))
    else:
        block_table(argv[1], int(argv[2]) if len(argv) > 2 else 16)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
