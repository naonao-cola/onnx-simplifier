"""Verification helper for `docs/axera-gather-compose-real-scale.md`.

Confirms, on real ResNet18-stem-scale fixtures (Gather then an elementwise
`Mul`-by-mask, 32,768 indices -- the same structure and real index/mask data
as the ResNet18 training step's stem Gather, one un-chunked slice of it), that
a compiled composed model's `npu_params` still holds the Gather's indices as
the first N contiguous little-endian uint32 words, the same layout
`memory_emit.py`'s standalone emitters rely on. This module does not add a
new generation capability -- `memory_emit.py` already retargets exactly this
layout -- it only pins the layout down with a regression test on a real,
composed (not standalone) reference, which had never been checked before.
"""

from __future__ import annotations

import struct

import onnx


def gather_indices(axmodel_path: str, count: int) -> tuple[int, ...]:
    """The first ``count`` uint32 words of ``npu_params`` -- the Gather's indices,
    for a compiled model whose graph is (equivalent to) Gather followed only by
    ops that do not themselves hold an index table before it in `npu_params`."""
    model = onnx.load(axmodel_path, load_external_data=False)
    table = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )
    words = struct.unpack(f"<{len(table) // 4}I", table)
    return words[:count]


def patch_indices(reference_path: str, output_path: str, indices) -> str:
    """Rewrite only the leading index words of `npu_params`, keeping everything
    else -- MCode, scales, the mask constant -- from the reference. The same
    operation `memory_emit.py`'s `emit_gather_last_axis_axmodel` performs for a
    standalone Gather, applied here to a composed reference."""
    model = onnx.load(reference_path, load_external_data=False)
    table = next(i for i in model.graph.initializer if i.name == "npu_params")
    words = list(struct.unpack(f"<{len(table.raw_data) // 4}I", bytes(table.raw_data)))
    indices = list(indices)
    if len(indices) > len(words):
        raise ValueError("more indices than npu_params words")
    words[: len(indices)] = indices
    table.raw_data = struct.pack(f"<{len(words)}I", *words)
    onnx.save(model, output_path)
    return output_path
