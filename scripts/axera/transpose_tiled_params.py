"""Predict the ``npu_params`` tile table of a tiled AX650 ``Transpose``.

A standalone float32 ``Transpose`` larger than 64 KiB is compiled by Pulsar2 7.0-lite into
a tiled program whose ``npu_params`` is a table of byte offsets. This module reproduces
that table exactly for the measured part of two families: ``Transpose(x[1,1,R,C],
perm=0,1,3,2)`` and ``Transpose(x[1,R,C,K], perm=0,2,1,3)`` (an ``[R, C]`` grid of
``4*K``-byte elements), plus 2-D ``perm=1,0``. It predicts *only* the table, not the MCode,
which is not predictable (see ``docs/axera-transpose-tiled.md``), so this is a
characterization tool and not an emitter. Everything outside the measured domain raises
``ValueError``.

The rule, for an ``[R, C]`` grid of ``e``-byte elements (``e = 4`` unless stated):

* Tiled iff ``e*R*C > 65536``. Otherwise ``npu_params`` is ten zero words.
* C a multiple of 8: tiles run along C. With ``tc = max(8, (C // 4) // 8 * 8)`` columns per
  tile there are ``ceil(C / tc)`` tiles; tile ``j`` starts at input offset ``j*tc*e`` and
  output offset ``j*tc*e*R``.
* A tile of ``rr`` rows holds at most 131072 bytes, counting each element padded to a power
  of two. When one chunk of ``R`` rows does not fit, the rows are cut into ``m`` chunks of
  ``ceil(R / m)`` rows, adding ``i*ceil(R/m)*e*C`` to the input offsets. For each ``tc``
  (``C/4``, then halved down to 8) try ``m`` = 1, 2, 3 and take the first that fits.
  Validated for 4-byte elements and ``C`` a power of two >= 32; for wider elements only
  ``m = 1`` is accepted.
* C not a multiple of 8 (4-byte elements) with R a multiple of 8 and ``C <= 644``: tiles run
  along R, ``tr = max(8, (R // 4) // 8 * 8)`` rows per tile, input offset ``i*tr*4*C`` and
  output offset ``i*tr*4``, valid while ``tr * C <= 15680``.
* Each group is written in the iteration order of a CPython ``set`` of its offsets,
  inserted column-major (``j`` outer, ``i`` inner): offsets that collide in the hash table
  keep insertion order. The table is ``input group + output group`` repeated five times.

Not covered (raises ``ValueError``): the C-tiled plan for unaligned C (its tile size and the
switch from R to C are not decoded), ``m >= 4``, row chunking of wide elements, R and C both
unaligned, and any batch ``N > 1`` (a different table form).
"""

from __future__ import annotations

import struct

_TILE_BYTES = 131072
_R_TILE_ELEMENTS = 15680


def _padded(elem_bytes: int) -> int:
    """Element size rounded up to a power of two: what a tile's size is counted in."""
    return 1 << (elem_bytes - 1).bit_length()


def is_tiled(r: int, c: int, elem_bytes: int = 4) -> bool:
    return elem_bytes * r * c > 65536


def _set_order(values: list[int]) -> list[int]:
    """Iteration order of a CPython ``set`` built by inserting ``values`` in the given order."""
    return list(set(values))


def c_tile_plan(r: int, c: int, elem_bytes: int = 4) -> tuple[int, int, int]:
    """``(tc, k, m)``: columns per tile, column tiles, and row chunks of ``ceil(R / m)`` rows."""
    if c % 8:
        raise ValueError("only validated for C a multiple of 8")
    pad = _padded(elem_bytes)
    tc = max(8, (c // 4) // 8 * 8)
    if r * tc * pad <= _TILE_BYTES:
        return tc, -(-c // tc), 1
    if c < 32 or c & (c - 1):
        raise ValueError(
            "multi-chunk plans are only validated for C a power of two >= 32"
        )
    # Try up to three row chunks at each column-tile size, halving the tile in between;
    # only at 8 columns may the chunk count keep growing.
    while True:
        for m in (1, 2, 3):
            if -(-r // m) * tc * pad <= _TILE_BYTES:
                return tc, c // tc, m
        if tc == 8:
            raise ValueError("more than three row chunks: output table not decoded")
        tc //= 2


def r_tile_plan(r: int, c: int) -> tuple[int, int]:
    """``(tr, k)``: rows per tile and tile count, for C not a multiple of 8, R a multiple of 8.

    Only the measured region is accepted: ``C <= 644`` and a row tile of at most 15680
    elements. Beyond that Pulsar2 tiles along C with a rule that is not decoded.
    """
    if r % 8 or not 8 < c <= 644:
        raise ValueError("R-tiled plan is only validated for R % 8 == 0 and C <= 644")
    tr = max(8, (r // 4) // 8 * 8)
    if tr * c > _R_TILE_ELEMENTS:
        raise ValueError("row tile too large: Pulsar2 tiles along C here (not decoded)")
    return tr, -(-r // tr)


def predict_words(r: int, c: int, elem_bytes: int = 4) -> list[int]:
    """The uint32 words of ``npu_params`` for transposing an ``[R, C]`` grid of elements.

    ``elem_bytes`` is 4 for ``Transpose(x[1,1,R,C], perm=0,1,3,2)`` and ``4*K`` for
    ``Transpose(x[1,R,C,K], perm=0,2,1,3)``.
    """
    if not is_tiled(r, c, elem_bytes):
        return [0] * 10
    e = elem_bytes
    if c % 8:
        if e != 4:
            raise ValueError("R-tiled plan is only validated for 4-byte elements")
        tr, k = r_tile_plan(r, c)
        src = [i * tr * 4 * c for i in range(k)]
        dst = [i * tr * 4 for i in range(k)]
        return (_set_order(src) + _set_order(dst)) * 5
    tc, k, m = c_tile_plan(r, c, e)
    if e != 4 and m > 1:
        # Wide elements: this rule picked more row chunks than Pulsar2 did in 5 of the 45
        # measured plans, so only the single-chunk plan is trusted.
        raise ValueError(
            "row chunking is not validated for elements other than 4 bytes"
        )
    rr = -(-r // m)
    # Column-major insertion: offsets that collide in the set's hash table keep insertion order.
    src = [i * rr * e * c + j * tc * e for j in range(k) for i in range(m)]
    dst = [j * tc * e * r for j in range(k)]
    return (_set_order(src) + _set_order(dst)) * 5


def predict_params(r: int, c: int, elem_bytes: int = 4) -> bytes:
    words = predict_words(r, c, elem_bytes)
    return struct.pack(f"<{len(words)}I", *words)


def _load_params(path: str) -> bytes:
    import onnx

    model = onnx.load(path, load_external_data=False)
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


def _grid(name: str) -> tuple[int, int, int] | None:
    """``(R, C, elem_bytes)`` for a sweep directory name, or None if it is another family.

    ``A_1x1xRxC`` (perm 0,1,3,2) and ``C_RxC`` (2-D, perm 1,0) move 4-byte elements;
    ``B_1xRxCxK`` (perm 0,2,1,3) moves ``4*K``-byte elements.
    """
    import re

    m = re.match(r"^A_1x1x(\d+)x(\d+)$", name) or re.match(r"^C_(\d+)x(\d+)$", name)
    if m:
        return int(m.group(1)), int(m.group(2)), 4
    m = re.match(r"^B_1x(\d+)x(\d+)x(\d+)$", name)
    if m:
        return int(m.group(1)), int(m.group(2)), 4 * int(m.group(3))
    return None


def check(work_root: str) -> int:
    """Compare predictions with every ``A_``, ``B_`` and ``C_`` build under ``work_root``.

    Prints exact / refused / wrong counts (a refusal is a ``ValueError``, never a guess) and
    returns the number of wrong predictions.
    """
    import os

    counts = {"exact": 0, "refused": 0, "wrong": 0}
    wrong = []
    for name in sorted(os.listdir(work_root)):
        grid = _grid(name)
        path = os.path.join(work_root, name, "out", "compiled.axmodel")
        if grid is None or not os.path.exists(path):
            continue
        try:
            ok = predict_params(*grid) == _load_params(path)
        except ValueError:
            counts["refused"] += 1
            continue
        counts["exact" if ok else "wrong"] += 1
        if not ok:
            wrong.append(name)
    print(counts, "wrong:", wrong)
    return counts["wrong"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "check":
        raise SystemExit("usage: transpose_tiled_params.py check WORK_ROOT")
    raise SystemExit(1 if check(sys.argv[2]) else 0)
