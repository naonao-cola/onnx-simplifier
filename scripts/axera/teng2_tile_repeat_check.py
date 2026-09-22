"""Checks the "does `teng2` repeat once per DMA tile?" hypothesis against a
compiled standalone elementwise op (`Relu`), and reproduces the adjacent
finding about segment 4.

`docs/axera-dma-queue.md` decoded the `npu_params` tile table for a
standalone `Relu` (`dma_tile_predict.py`) but left segment 2 -- `teng2`, the
compute program -- undecoded. Segment 4 in that same doc's segment layout is
`sdma4`, the DMA queue itself, already shown (in
`docs/axera-step-attribution.md`) to split into one `0xa3`-terminated group
per DMA task in a real training step. The natural next question: does
`teng2` (or `sdma4`) repeat, once per `npu_params` tile entry, the way
`transpose_tiled_params.py`'s tiled Transpose segment 4 does?

Two measurements, per compiled `.axmodel`:

* **Segment 2 (`teng2`) periodicity at period = size / n.** Byte
  self-similarity (fraction of byte `i` equal to byte `i + period`) and a
  record-signature check (decode with `mcode.FULL_RULE`, split the record
  list into `n` equal blocks by record count, compare block shapes). Both
  measure whether `teng2` is `n` copies of one per-tile program; see
  `docs/axera-teng2-tiled-repeat.md` for why the answer is no.
* **Segment 4 (`sdma4`) size and `0xa3`/verb counts.** Whether the DMA queue
  has real content (task-structured, `0xa3`-terminated groups) or is a bare
  32-byte placeholder -- which, empirically, depends on total tensor bytes,
  not on `n`.

Usage::

    teng2_tile_repeat_check.py WORK_ROOT   # one row per Relu build under it
    teng2_tile_repeat_check.py FIXTURE.axmodel.gz  # a single gzipped model
"""

from __future__ import annotations

import gzip
import os
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode as mcode_mod  # noqa: E402


def _load_model(path: str) -> onnx.ModelProto:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return onnx.load_model_from_string(f.read())
    return onnx.load(path, load_external_data=False)


def load_mcode_params_shape(path: str) -> tuple[bytes, bytes, list[int]]:
    model = _load_model(path)
    inits = {i.name: i for i in model.graph.initializer}
    mcode = next(
        bytes(i.raw_data) for name, i in inits.items() if name.endswith("_neu")
    )
    params = bytes(inits["npu_params"].raw_data)
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return mcode, params, shape


def n_entries(params: bytes) -> int:
    if len(params) % 40:
        raise ValueError(
            f"npu_params is not a whole number of 40-byte entries: {len(params)}"
        )
    return len(params) // 40


def self_similarity(data: bytes, period: int) -> float | None:
    """Fraction of bytes equal to the byte one `period` earlier; None if `period` doesn't fit."""
    if period <= 0 or period >= len(data):
        return None
    count = len(data) - period
    equal = sum(1 for i in range(count) if data[i] == data[i + period])
    return equal / count


def _record_shape(record: dict) -> tuple:
    """A record's form, ignoring the operand/payload bytes most likely to hold a per-tile
    address or size immediate -- what a genuine per-tile repeat would still share."""
    kind = record["kind"]
    if kind == "V":
        return (kind, record["verb"], record["field"], record["bank"])
    if kind == "S":
        return (kind, record["p"], record["tag"])
    if kind == "B":
        return (kind, record["tag"])
    return (kind,)


def teng2_periodicity(mcode: bytes, params: bytes) -> dict:
    """Segment-2 self-similarity and record-block match at period = size/n."""
    n = n_entries(params)
    _, segs = mcode_mod.segments(mcode)
    start, size, _ = segs[2]
    seg2 = mcode[start : start + size]
    result = {
        "n": n,
        "seg2_size": size,
        "byte_selfsim_at_size_over_n": None,
        "record_blocks_matching_first": None,
        "record_count": None,
    }
    if n and size % n == 0:
        result["byte_selfsim_at_size_over_n"] = self_similarity(seg2, size // n)
    records = mcode_mod.decode(
        mcode, start=start, end=start + size, **mcode_mod.FULL_RULE
    )
    result["record_count"] = len(records)
    if n and len(records) % n == 0:
        period = len(records) // n
        shapes = [_record_shape(r) for r in records]
        blocks = [shapes[i * period : (i + 1) * period] for i in range(n)]
        result["record_blocks_matching_first"] = sum(
            1 for b in blocks if b == blocks[0]
        )
    return result


def sdma4_summary(mcode: bytes) -> dict:
    """Segment-4 size and its `0xa1/0xa2/0xa3/0xa8/0xa9` verb counts."""
    _, segs = mcode_mod.segments(mcode)
    start, size, _ = segs[4]
    records = mcode_mod.decode(
        mcode, start=start, end=start + size, **mcode_mod.FULL_RULE
    )
    verb_counts = {}
    for record in records:
        if record["kind"] == "V":
            verb_counts[record["verb"]] = verb_counts.get(record["verb"], 0) + 1
    return {"seg4_size": size, "empty": size <= 64, "verb_counts": verb_counts}


def report_row(name: str, path: str) -> str:
    mcode, params, shape = load_mcode_params_shape(path)
    total_bytes = 1
    for dim in shape:
        total_bytes *= dim
    total_bytes *= 4
    teng2 = teng2_periodicity(mcode, params)
    sdma4 = sdma4_summary(mcode)
    return (
        f"{name:24s} shape={shape!s:20s} bytes={total_bytes:9d} n={teng2['n']:3d} "
        f"teng2_selfsim={teng2['byte_selfsim_at_size_over_n']} "
        f"teng2_record_blocks_matching={teng2['record_blocks_matching_first']}/{teng2['n']} "
        f"sdma4_size={sdma4['seg4_size']:5d} sdma4_empty={sdma4['empty']}"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    target = argv[0]
    if os.path.isdir(target):
        for name in sorted(os.listdir(target)):
            path = os.path.join(target, name, "out", "compiled.axmodel")
            if os.path.exists(path):
                print(report_row(name, path))
    else:
        print(report_row(os.path.basename(target), target))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
