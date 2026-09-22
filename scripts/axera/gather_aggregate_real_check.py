"""Verification helper for `docs/axera-gather-aggregate-real.md`.

Confirms, on a real ResNet18-step-scale composed reference (`Gather(x[16,1,
512,49], idx[441], axis=3) -> Mul(mask) -> Reshape([16,1,4608,49]) -> MatMul(
w[1,1,512,4608], reshaped)`, the exact op sequence and real index/mask data
around the training step's `Gather_48`/`Mul_49`/`Reshape_50`/`MatMul_54`),
that index retargeting through an aggregating (`MatMul`, contracting over the
gathered axis) consumer reproduces `docs/axera-compose.md`'s toy-scale
finding at real scale: a narrowly-calibrated reference clips when retargeted
to indices whose selected activations fall outside the calibration range, and
calibrating the *input* tensor's full valid range (not the aggregate output,
and not specifically the target indices) fixes it.

`patch_indices` in `gather_compose_real_check.py` (PR #1759) assumes
`npu_params` is a whole number of uint32 words -- `len(table.raw_data) // 4`
truncates and silently drops trailing bytes otherwise. This composed
reference's table is 10,861 bytes (one byte over 2,715 words), so that
function raises `struct.error` on it. `patch_indices_bytesafe` below fixes
this by writing the new index bytes directly into the table's raw bytes
in place, leaving every byte after the indices -- including the odd one --
untouched.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import onnx


def patch_indices_bytesafe(
    reference_path: str, output_path: str, indices: Sequence[int]
) -> str:
    """Rewrite only the leading ``len(indices) * 4`` bytes of ``npu_params``
    (the Gather's indices, little-endian uint32) and leave every other byte
    of the table -- MCode, quantization scales, and any trailing byte past a
    whole word -- untouched. Works whether or not the table's total length is
    a multiple of 4 bytes, unlike `gather_compose_real_check.patch_indices`.
    """
    model = onnx.load(reference_path, load_external_data=False)
    table = next(i for i in model.graph.initializer if i.name == "npu_params")
    raw = bytearray(table.raw_data)
    packed = struct.pack(f"<{len(indices)}I", *indices)
    if len(packed) > len(raw):
        raise ValueError("more indices than npu_params bytes")
    raw[: len(packed)] = packed
    table.raw_data = bytes(raw)
    onnx.save(model, output_path)
    return output_path


def leading_indices(axmodel_path: str, count: int) -> tuple[int, ...]:
    """The first ``count`` uint32 words of a compiled model's ``npu_params``."""
    model = onnx.load(axmodel_path, load_external_data=False)
    table = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )
    words = struct.unpack(f"<{count}I", table[: count * 4])
    return words
