"""Does the ResNet18 stem Gather's real leading `Reshape` (`Reshape_466`,
`[16,3,224,224] -> [16,1,3,50176]`) change anything about the already-working
14-chunk `Gather+Mul` composition (`docs/axera-stem-gather-rechunk.md`, PR
#1767)? Answer, recorded in `docs/axera-stem-gather-reshape.md`: the compiled
artifact including the real `Reshape` works correctly end to end on device
(max err 0.0098 across 29.5M outputs, the same noise floor every other
in-range Gather check in this project reports), but its `npu_params` index
encoding is NOT simply the Reshape-free template shifted by a fixed offset --
best word-alignment match is 1.5%, not the ~100% a shift-only difference
would give. The existing `memory_emit.py` / `stem_gather_compose_emit.py`
"rewrite the leading N words" retargeting trick does not apply unmodified to
this fuller, more-real composition; a working retargeting recipe for this
exact chain has not been found.

This module holds only the static (no Docker/device) part: confirming the
index-layout mismatch against the two committed reference fixtures.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESHAPE_TEMPLATE = os.path.join(
    _HERE, "fixtures", "stem_gather_reshape", "reshape14_reference.axmodel.gz"
)
_RESHAPE_FREE_TEMPLATE = os.path.join(
    _HERE, "fixtures", "gather_compose_real", "stem_rechunk14_reference.axmodel.gz"
)

_N_IDX = 614656


def _npu_params(gz_path: str) -> bytes:
    with gzip.open(gz_path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


def best_shift_match(max_shift_words: int = 40) -> tuple[int, float]:
    """`(shift_words, fraction_matching)` for the best word-alignment of the
    Reshape-graph's `npu_params` against the Reshape-free reference's index
    region, over shifts `0..max_shift_words`. A simple "small block inserted
    at the front" layout would give a match near 1.0 at the shift equal to
    that block's word length (20, matching the 80-byte total length
    difference); this project's actual measured result is ~0.015 at every
    shift tried.
    """
    reshape_table = _npu_params(_RESHAPE_TEMPLATE)
    reference_words = np.frombuffer(
        _npu_params(_RESHAPE_FREE_TEMPLATE)[: _N_IDX * 4], dtype="<u4"
    )
    best = (0, 0.0)
    for shift_words in range(max_shift_words):
        offset = shift_words * 4
        segment_bytes = reshape_table[offset : offset + _N_IDX * 4]
        if len(segment_bytes) != _N_IDX * 4:
            continue
        segment = np.frombuffer(segment_bytes, dtype="<u4")
        match = float((segment == reference_words).mean())
        if match > best[1]:
            best = (shift_words, match)
    return best


def table_length_diff() -> int:
    """Bytes by which the Reshape graph's `npu_params` is longer than the
    Reshape-free reference's (20 words / 80 bytes, at the time this was
    measured)."""
    return len(_npu_params(_RESHAPE_TEMPLATE)) - len(
        _npu_params(_RESHAPE_FREE_TEMPLATE)
    )


if __name__ == "__main__":
    shift, match = best_shift_match()
    print(f"best shift: {shift} words, {match * 100:.2f}% index words match")
    print(f"table length diff: {table_length_diff()} bytes")
