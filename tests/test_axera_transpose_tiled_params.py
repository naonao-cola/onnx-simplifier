"""The tile-table predictor for tiled AX650 ``Transpose`` builds.

The fixture holds ``npu_params`` tables read from real Pulsar2 7.0-lite builds (one period
of each, plus the total word count). Twenty-six of them are from a set built after the rule
was frozen; see ``docs/axera-transpose-tiled.md``.
"""

import gzip
import json
import os
import struct
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import transpose_tiled_params as tp  # noqa: E402

_FIXTURE = os.path.join(_AXERA_DIR, "fixtures", "transpose_tiled_params.json.gz")
with gzip.open(_FIXTURE, "rt") as _f:
    _MEASURED = json.load(_f)


def _shape(key):
    """``(R, C, elem_bytes)``: keys are ``RxC`` for 4-byte elements or ``RxCxE``."""
    parts = [int(p) for p in key.split("x")]
    return parts[0], parts[1], parts[2] if len(parts) == 3 else 4


@pytest.mark.parametrize("key", sorted(_MEASURED))
def test_predictor_reproduces_measured_table(key):
    r, c, e = _shape(key)
    measured = _MEASURED[key]
    period = measured["period"]
    expected = period * (measured["words"] // len(period))
    assert len(expected) == measured["words"]
    assert tp.predict_words(r, c, e) == expected
    assert tp.predict_params(r, c, e) == struct.pack(f"<{len(expected)}I", *expected)


def test_fixture_covers_tiled_and_untiled_shapes():
    tiled = [k for k, v in _MEASURED.items() if v["words"] > 10]
    untiled = [k for k, v in _MEASURED.items() if v["words"] <= 10]
    assert len(tiled) >= 60 and len(untiled) >= 10
    assert any(k.count("x") == 2 for k in tiled)  # wide elements are covered too
    assert all(tp.is_tiled(*_shape(k)) for k in tiled)
    assert not any(tp.is_tiled(*_shape(k)) for k in untiled)


@pytest.mark.parametrize(
    "r,c,expected",
    [
        (1024, 16, False),  # exactly 65536 bytes
        (1025, 16, True),  # 65600 bytes
        (128, 128, False),
        (129, 128, True),
        (129, 127, False),  # 65532 bytes
    ],
)
def test_untiled_threshold_is_64_kib(r, c, expected):
    assert tp.is_tiled(r, c) is expected
    if not expected:
        assert tp.predict_words(r, c) == [0] * 10


@pytest.mark.parametrize(
    "r,c,plan",
    [
        (264, 64, (16, 4, 1)),  # four column tiles of 16
        (1025, 16, (8, 2, 1)),  # tile size is never below 8 columns
        (2400, 64, (16, 4, 2)),  # tile would hold 38400 elements: two row chunks
        (6400, 64, (8, 8, 2)),  # chunks of 3200 rows do not fit 16 columns: halve
        (1300, 256, (64, 4, 3)),  # three uneven chunks (434, 434, 432 rows)
        (9600, 64, (8, 8, 3)),
        (3200, 128, (16, 8, 2)),
    ],
)
def test_column_tile_plan(r, c, plan):
    assert tp.c_tile_plan(r, c) == plan


@pytest.mark.parametrize(
    "r,c,plan",
    [(336, 49, (80, 5)), (352, 49, (88, 4)), (1280, 49, (320, 4)), (104, 164, (24, 5))],
)
def test_row_tile_plan(r, c, plan):
    assert tp.r_tile_plan(r, c) == plan


def test_single_tile_offsets_use_column_major_set_order():
    # Two row chunks of 1200 rows and four column tiles of 16: offsets that collide in
    # the hash table keep insertion order, so this is column-major, not ascending.
    words = tp.predict_words(2400, 64)
    period = words[: len(words) // 5]
    assert period[:8] == [0, 307200, 64, 307264, 128, 307328, 192, 307392]
    assert period[8:] == [0, 153600, 307200, 460800]


def test_tables_repeat_the_base_period_five_times():
    for key in _MEASURED:
        words = tp.predict_words(*_shape(key))
        if len(words) > 10:
            base = words[: len(words) // 5]
            assert words == base * 5


@pytest.mark.parametrize(
    "r,c,e",
    [
        (128, 128, 36),  # a two-chunk plan of 36-byte elements: not validated
        (512, 512, 36),  # the ResNet18 weight transposes are in this class
        (364, 32, 36),  # padded-cap plan disagrees with what Pulsar2 built
    ],
)
def test_wide_element_row_chunking_refuses(r, c, e):
    assert tp.is_tiled(r, c, e)
    with pytest.raises(ValueError):
        tp.predict_words(r, c, e)


def test_wide_element_offsets_use_the_real_element_size():
    # [1,64,64,9]: 64x64 grid of 36-byte elements, four column tiles of 16.
    words = tp.predict_words(64, 64, 36)
    assert sorted(words[:4]) == [0, 576, 1152, 1728]  # j * 16 columns * 36 bytes
    assert sorted(words[4:8]) == [0, 36864, 73728, 110592]  # j * 16 * 36 * 64 rows


@pytest.mark.parametrize(
    "r,c",
    [
        (101, 201),  # neither R nor C a multiple of 8
        (100, 164),  # C unaligned, R unaligned
        (1536, 49),  # Pulsar2 tiles along C here; rule not decoded
        (72, 969),  # likewise
        (144, 538),
        (12800, 64),  # four row chunks: output table has extra words
        (128, 1568),  # C not a power of two in the multi-chunk regime
        (1600, 96),  # likewise
    ],
)
def test_out_of_domain_shapes_refuse_instead_of_guessing(r, c):
    assert tp.is_tiled(r, c)
    with pytest.raises(ValueError):
        tp.predict_words(r, c)
