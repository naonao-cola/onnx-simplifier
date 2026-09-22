"""Does a standalone Conv weight-code table survive real-graph composition?

Answer: **the weight-code region does, exactly, for the terminal Conv in a
fused chain; the requantisation scaffold does not.** Two real 4-op chains
(`Conv -> Add(skip) -> Relu -> Conv`, the real ResNet18 stage-1 residual
pattern, `[16,64,56,56]`, `Conv(64,64,3x3)` both times) were compiled with
Pulsar2, using the reference/holdout weight pairs already committed by
`conv_weight_learn.py` (PR #1769), with the two weight sets in opposite
graph positions between the two builds (`chain1`: reference first / holdout
last; `chain2`: holdout first / reference last -- a swap control).

Findings, see `docs/axera-conv-compose-real.md` for the full writeup:

* Composition fuses the whole chain into **one** `neu mode` node, same as
  every other op's composition check in this project (Transpose, Gather).
* The **terminal** Conv (nothing consumes its output further -- here, the
  second Conv, whose output is the graph's own output) keeps the exact
  standalone weight-code table layout: `emitter.emit_table` applied to the
  standalone reference table with that Conv's own weights reproduces the
  composed table's weight-code bytes with **zero mismatches**, at a fixed
  byte offset (`TERMINAL_OFFSET`, empirically located by sliding a window
  and scoring Hamming distance against the masked weight-code bits).
  Confirmed in **both** chains, with the weight sets swapped -- it is a
  property of chain *position*, not of which specific weights are used.
* The **interior** Conv (its output feeds `Add`+`Relu` before the next op)
  does **not** match the standalone layout anywhere in the composed table.
* Patching a fresh weight set into the terminal Conv's located region (full
  pipeline: `emitter.emit_table` for the code bits,
  `conv_weight_learn.requant_block_biased` for the scaffold, using
  `conv_weight_learn.build_scales` to read that Conv's own x/y scale from
  the composed build's own `quant_axmodel.json`) produced **incorrect**
  device output: mean error 0.79 against a control's 0.014, not explained
  by calibration-range clipping (only ~3e-7 of the true output fell outside
  the representable range). The weight-code write itself was independently
  confirmed correct (zero mismatches) for the same patch, so the scaffold
  computation -- not the code layout -- is where this breaks. This is a
  real, unresolved gap, not glossed over as success.

This module holds the offset-finding/verification logic used to produce
those findings, reproducible without Docker or a device from the two
committed fixtures.
"""

from __future__ import annotations

import gzip
import io
import os
import sys

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import emitter  # noqa: E402

_FIXTURES = os.path.join(_HERE, "fixtures", "conv_compose_real")
_CONV_LEARN_FIXTURES = os.path.join(_HERE, "fixtures", "conv_weight_learn")

TERMINAL_OFFSET = 37380
"""Byte offset of the terminal Conv's weight-code+scaffold region within the composed
table, empirically located and identical across both swap-control chains."""


def _load_gz_model(path: str) -> onnx.ModelProto:
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


def _load_gz_npy(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.load(io.BytesIO(f.read()))


def npu_params(model: onnx.ModelProto) -> np.ndarray:
    init = next(i for i in model.graph.initializer if i.name == "npu_params")
    return np.frombuffer(bytes(init.raw_data), dtype=np.uint8).copy()


def standalone_reference() -> tuple[np.ndarray, np.ndarray]:
    """`(table, origin)`: the standalone `Conv(64,64,3,3)` reference table (PR #1769)
    and its learned weight-code bit-origin map."""
    model = _load_gz_model(os.path.join(_CONV_LEARN_FIXTURES, "reference.axmodel.gz"))
    table = npu_params(model)
    origin = np.load(
        os.path.join(_CONV_LEARN_FIXTURES, "stage1_map.npz"), allow_pickle=True
    )["origin"]
    return table, origin


def code_byte_mask(table_len: int, origin: np.ndarray) -> np.ndarray:
    """Bytes of the standalone table that carry a weight-code bit (deterministic,
    weight-dependent) rather than scaffold/constant bytes."""
    mask = np.zeros(table_len, dtype=bool)
    for bit in range(table_len * 8):
        if origin[bit] >= 0:
            mask[bit // 8] = True
    return mask


def find_weight_code_offset(
    composed_table: np.ndarray, standalone_table: np.ndarray, origin: np.ndarray, w
) -> tuple[int, int]:
    """`(best_shift, mismatches)`: the byte offset within `composed_table` whose
    weight-code bytes best match `w`'s predicted encoding (Hamming distance over the
    masked weight-code bytes only, sliding a `len(standalone_table)`-wide window)."""
    predicted = emitter.emit_table(standalone_table, origin, emitter.codes_of(w))
    mask = code_byte_mask(len(standalone_table), origin)
    best = None
    max_shift = len(composed_table) - len(standalone_table)
    for shift in range(0, max_shift + 1):
        window = composed_table[shift : shift + len(standalone_table)]
        mismatches = int(np.count_nonzero((window != predicted) & mask))
        if best is None or mismatches < best[1]:
            best = (shift, mismatches)
    return best


def check() -> int:
    """Reproduce the terminal-vs-interior finding on both committed fixtures."""
    standalone_table, origin = standalone_reference()
    ref_w = _load_gz_npy(os.path.join(_CONV_LEARN_FIXTURES, "reference_w.npy.gz"))
    hold_w = _load_gz_npy(os.path.join(_CONV_LEARN_FIXTURES, "holdout_w.npy.gz"))

    mismatches = 0
    for name, terminal_w, interior_w in (
        ("chain1 (reference first, holdout terminal)", hold_w, ref_w),
        ("chain2 (holdout first, reference terminal)", ref_w, hold_w),
    ):
        gz = "chain1.axmodel.gz" if "chain1" in name else "chain2.axmodel.gz"
        composed_table = npu_params(_load_gz_model(os.path.join(_FIXTURES, gz)))

        shift, mism = find_weight_code_offset(
            composed_table, standalone_table, origin, terminal_w
        )
        ok = shift == TERMINAL_OFFSET and mism == 0
        print(
            f"{name}: terminal weight best shift={shift} mismatches={mism} {'OK' if ok else 'MISMATCH'}"
        )
        mismatches += not ok

        _, mism_wrong = find_weight_code_offset(
            composed_table, standalone_table, origin, interior_w
        )
        # The interior weight should NOT find a clean match anywhere.
        interior_ok = mism_wrong > len(standalone_table) // 4
        print(
            f"{name}: interior weight best mismatches={mism_wrong} {'OK (no match, as expected)' if interior_ok else 'UNEXPECTED MATCH'}"
        )
        mismatches += not interior_ok

    return mismatches


if __name__ == "__main__":
    sys.exit(1 if check() else 0)
