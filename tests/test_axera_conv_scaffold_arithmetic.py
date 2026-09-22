"""Regression coverage for the deterministic wide-Cout Conv scaffold layout."""

import gzip
import os
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from conv_scaffold_arithmetic import (  # noqa: E402
    scaffold_bytes,
    tile_offsets,
    tile_width,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_scaffold_arithmetic")


def _load_table(gz_name):
    with gzip.open(os.path.join(_FIXTURES, gz_name), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return np.frombuffer(
        bytes(
            next(i.raw_data for i in model.graph.initializer if i.name == "npu_params")
        ),
        dtype=np.uint8,
    ).copy()


def _load_meta(npz_name):
    z = np.load(os.path.join(_FIXTURES, npz_name))
    return dict(z)


@pytest.mark.parametrize(
    "axmodel_gz,meta_npz",
    [
        ("conv128_build1.axmodel.gz", "conv128_build1_meta.npz"),
        ("conv128_build2.axmodel.gz", "conv128_build2_meta.npz"),
    ],
)
def test_tile_offsets_locate_the_real_scaffold_blind(axmodel_gz, meta_npz):
    """No search: read the formula's predicted offsets directly and compare to a real
    compiled table, for two independent Pulsar2 builds with different random weights."""
    import emitter

    table = _load_table(axmodel_gz)
    meta = _load_meta(meta_npz)
    w, b = meta["w"], meta["b"]
    x_scale, x_zero = float(meta["x_scale"]), float(meta["x_zero"])
    y_scale, y_zero = float(meta["y_scale"]), float(meta["y_zero"])
    w_scale = meta["w_scale"]

    cout, cin, kh, kw = w.shape
    assert kh == kw
    codes = emitter.codes_of(w)
    bias_all, m_all = scaffold_bytes(
        codes, x_scale, x_zero, y_scale, y_zero, w_scale, b
    )

    offsets, total = tile_offsets(cout, cin, kh)
    assert offsets == [
        (0, 32, 0, 36864, 36992),
        (32, 32, 37120, 73984, 74112),
        (64, 32, 74240, 111104, 111232),
        (96, 32, 111360, 148224, 148352),
    ]

    for channel, width, _weight_start, bias_start, m_start in offsets:
        table_bias = table[bias_start : bias_start + 4 * width].view(np.float32)
        table_m = table[m_start : m_start + 4 * width].view(np.float32)
        np.testing.assert_array_equal(table_m, m_all[channel : channel + width])
        np.testing.assert_allclose(
            table_bias, bias_all[channel : channel + width], atol=2e-4, rtol=0
        )


def test_tile_width_matches_every_checked_shape():
    # Contiguous (single-tile) shapes the already-merged emitters found working:
    assert (
        tile_width(cin=64, k=3, cout=64) == 64
    )  # docs/axera-conv-weight-learn-stem.md
    assert (
        tile_width(cin=64, k=1, cout=128) == 128
    )  # 1x1 downsample, block_at=9216 case
    # Tiled shapes:
    assert tile_width(cin=128, k=3, cout=128) == 32
    assert tile_width(cin=256, k=3, cout=256) == 16
    assert tile_width(cin=512, k=3, cout=512) == 8


def test_256_256_predicted_run_count_matches_published_finding():
    """docs/axera-conv-weight-learn-wide.md independently reported 512 scattered
    ambiguous-byte runs for Conv(256,256,3,3)'s scaffold, found via bit-permutation
    learning, with no reference to tile geometry. This checks the formula predicts
    the same count: 16 tiles of 16 channels, 2 runs (bias, M) per channel."""
    offsets, _ = tile_offsets(cout=256, cin=256, k=3)
    assert len(offsets) == 16
    assert all(width == 16 for _, width, *_ in offsets)
    assert len(offsets) * 16 * 2 == 512
