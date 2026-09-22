"""Regression coverage for the 128->256 downsample Conv weight-learn emitter.

No Docker/device needed: these check the *emitted* table against committed
reference fixtures. `docs/axera-conv-weight-learn-128to256.md` has the real
Pulsar2-build and AX8850-device evidence this is standing in for.
"""

import gzip
import json
import os
import sys

import numpy as np
import onnx

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import emitter  # noqa: E402
from conv_128to256_tiled import (  # noqa: E402
    P1_BLOCK_AT,
    P1_BLOCK_LEN,
    P3_TABLE_LEN,
    emit_p1_table,
    emit_p3_table,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_learn_128to256")


def _load_gz_model(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _table_of(model, name="npu_params"):
    return next(
        np.frombuffer(bytes(i.raw_data), dtype=np.uint8)
        for i in model.graph.initializer
        if i.name == name
    )


def _quant_scale_zero(gz_json_name, tensor):
    with gzip.open(os.path.join(_FIXTURES, gz_json_name), "rt") as f:
        doc = json.load(f)
    tc = list(doc["tensor_configs"].values())[0][tensor]
    v = doc["values"][str(tc["hash"])]
    return float(v["scale"][0]), float(v["zero_point"][0])


def test_p1_1x1_weight_code_and_scaffold_match_native():
    """Full pipeline: k=46-build-learned weight-code map + the one contiguous
    scaffold block reproduces the real compiled table for held-out weights,
    within the float32 rounding `conv_bias_requant.py`'s formula documents
    everywhere else in this project."""
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "p1_map.npz")
    )
    assert tuple(shape) == (256, 128, 1, 1)

    reference = _load_gz_model("p1_reference.axmodel.gz")
    ref_table = _table_of(reference)
    native = _load_gz_model("p1_holdout_native.axmodel.gz")
    native_table = _table_of(native)

    w = np.load(os.path.join(_FIXTURES, "p1_holdout_w.npy"))
    b = np.load(os.path.join(_FIXTURES, "p1_holdout_b.npy"))
    x_scale, x_zero = _quant_scale_zero("p1_reference_quant.json.gz", "x")
    y_scale, y_zero = _quant_scale_zero("p1_holdout_quant.json.gz", "y")

    emitted = emit_p1_table(ref_table, origin, w, b, x_scale, x_zero, y_scale, y_zero)
    assert emitted.shape == native_table.shape

    outside = np.ones(len(emitted), dtype=bool)
    outside[P1_BLOCK_AT : P1_BLOCK_AT + P1_BLOCK_LEN] = False
    assert np.array_equal(emitted[outside], native_table[outside])

    m_e = emitted[P1_BLOCK_AT + 1024 : P1_BLOCK_AT + 2048].view("<f4")
    m_n = native_table[P1_BLOCK_AT + 1024 : P1_BLOCK_AT + 2048].view("<f4")
    assert np.abs(m_e.astype(np.float64) - m_n.astype(np.float64)).max() < 1e-6

    bias_e = emitted[P1_BLOCK_AT : P1_BLOCK_AT + 1024].view("<f4")
    bias_n = native_table[P1_BLOCK_AT : P1_BLOCK_AT + 1024].view("<f4")
    assert np.abs(bias_e.astype(np.float64) - bias_n.astype(np.float64)).max() < 1e-3


def test_p3_3x3_weight_code_and_all_8_tiles_match_native():
    """Full pipeline for the harder shape: 8-tile scaffold, k=54-build weight
    codes (0 collisions)."""
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "p3_map.npz")
    )
    assert tuple(shape) == (256, 128, 3, 3)

    reference = _load_gz_model("p3_reference.axmodel.gz")
    ref_table = _table_of(reference)
    native = _load_gz_model("p3_holdout_native.axmodel.gz")
    native_table = _table_of(native)
    assert len(native_table) == P3_TABLE_LEN

    w = np.load(os.path.join(_FIXTURES, "p3_holdout_w.npy"))
    b = np.load(os.path.join(_FIXTURES, "p3_holdout_b.npy"))
    x_scale, x_zero = _quant_scale_zero("p3_reference_quant.json.gz", "x")
    y_scale, y_zero = _quant_scale_zero("p3_holdout_quant.json.gz", "y")

    emitted = emit_p3_table(ref_table, origin, w, b, x_scale, x_zero, y_scale, y_zero)
    assert emitted.shape == native_table.shape

    scaffold_mask = np.zeros(len(emitted), dtype=bool)
    for tile in range(8):
        tile_start = tile * 37120
        scaffold_mask[tile_start + 36864 : tile_start + 37120] = True
    outside = ~scaffold_mask
    assert np.array_equal(emitted[outside], native_table[outside])

    # the 480-byte trailer past the last tile is part of "outside" and must
    # also be byte-exact (it's constant across builds, copied from the
    # reference untouched).
    assert np.array_equal(
        emitted[8 * 37120 :],
        native_table[8 * 37120 :],
    )

    max_m_diff = 0.0
    max_bias_diff = 0.0
    for tile in range(8):
        tile_start = tile * 37120
        bias_at = tile_start + 36864
        m_at = bias_at + 128
        m_e = emitted[m_at : m_at + 128].view("<f4").astype(np.float64)
        m_n = native_table[m_at : m_at + 128].view("<f4").astype(np.float64)
        bias_e = emitted[bias_at : bias_at + 128].view("<f4").astype(np.float64)
        bias_n = native_table[bias_at : bias_at + 128].view("<f4").astype(np.float64)
        max_m_diff = max(max_m_diff, np.abs(m_e - m_n).max())
        max_bias_diff = max(max_bias_diff, np.abs(bias_e - bias_n).max())
    assert max_m_diff < 1e-6
    assert max_bias_diff < 1e-3


def test_conv_scaffold_arithmetic_tobytes_bug_reproduces():
    """Pins the real bug this module's docstring reports in the merged
    `conv_scaffold_arithmetic.emit_conv_table_tiled`: assigning a raw
    `bytes` object to a `uint8` ndarray slice raises, it does not copy
    bytes. Confirms the bug is real (not a misreading) and that the
    `np.frombuffer` fix this module uses instead avoids it."""
    import pytest

    table = np.zeros(64, dtype=np.uint8)
    payload = np.arange(8, dtype=np.float32)
    with pytest.raises(ValueError):
        table[0:32] = payload.tobytes()
    # the fix:
    table[0:32] = np.frombuffer(payload.tobytes(), dtype=np.uint8)
    assert np.array_equal(table[0:32].view("<f4"), payload)
