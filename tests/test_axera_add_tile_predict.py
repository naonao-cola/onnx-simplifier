"""Regression coverage for the two-input elementwise (Add) DMA tile table predictor."""

import gzip
import os
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from add_tile_predict import predict_params  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "add_tiles")


def _npu_params(gz_name):
    with gzip.open(os.path.join(_FIXTURES, gz_name), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


# Scales are the per-tensor calibration scales Pulsar2 wrote to that build's own
# out/quant/quant_axmodel.json (tensor_configs["y"][k]["hash"] -> values[hash]["scale"][0]);
# not derivable from shape alone, unlike the single-input Relu/Transpose tile tables.
@pytest.mark.parametrize(
    "gz_name,shape,x_scale,z_scale,y_scale",
    [
        # Untiled, symmetric ranges: x/y and z/y round to the same Q15 word (dedup, 2-byte header).
        (
            "add_1x32x32x32.axmodel.gz",
            [1, 32, 32, 32],
            0.00705871544778347,
            0.007058681920170784,
            0.014049092307686806,
        ),
        # Untiled, x/y and z/y round to different Q15 words (4-byte header).
        (
            "add_1x16x8x8.axmodel.gz",
            [1, 16, 8, 8],
            0.007058057934045792,
            0.00705726258456707,
            0.013801783323287964,
        ),
        # Tiled, 4 entries, dedup header.
        (
            "add_1x64x56x56.axmodel.gz",
            [1, 64, 56, 56],
            0.007058816496282816,
            0.007058818358927965,
            0.014106313697993755,
        ),
        # Tiled, 8 entries.
        (
            "add_4x64x56x56.axmodel.gz",
            [4, 64, 56, 56],
            0.007058820221573114,
            0.007058819755911827,
            0.014107690192759037,
        ),
        # Tiled, 32 entries -- the largest N*C (1024) still below the split threshold.
        (
            "add_16x64x56x56.axmodel.gz",
            [16, 64, 56, 56],
            0.007058822084218264,
            0.007058819755911827,
            0.01411121804267168,
        ),
        # Deliberately asymmetric, non-overlapping input ranges: the header's two Q15
        # words are far apart (0x73f5 and 0x0ce2), confirming it isn't a coincidence.
        (
            "add_asym_1x16x16x16.axmodel.gz",
            [1, 16, 16, 16],
            0.007058156654238701,
            0.0007841793121770024,
            0.007791136857122183,
        ),
        # Held out: a shape (8x64x28x28), calibration ranges ((-0.3,0.6)/(-1.5,0.2)),
        # and RNG seeds (42/99) not used anywhere else in this decode.
        (
            "add_holdout_8x64x28x28.axmodel.gz",
            [8, 64, 28, 28],
            0.003529408248141408,
            0.006666665896773338,
            0.01018680538982153,
        ),
    ],
)
def test_predict_params_matches_compiled_fixture(
    gz_name, shape, x_scale, z_scale, y_scale
):
    assert predict_params(shape, x_scale, z_scale, y_scale) == _npu_params(gz_name)


def test_predict_params_rejects_split_regime():
    # N*C == 2048: the table becomes an undecoded three-way read/write mix.
    with pytest.raises(ValueError, match="splits into an undecoded"):
        predict_params([16, 128, 28, 28], 0.007, 0.007, 0.014)


def test_predict_params_rejects_split_regime_fixture():
    # The real compiled split-regime model this project has on hand, with its own
    # scales: confirms the rejection is not just a synthetic-number exercise, and
    # that its npu_params really is longer than the (rejected) simple prediction
    # would be.
    real = _npu_params("add_16x128x28x28.axmodel.gz")
    assert len(real) == 1602
    with pytest.raises(ValueError, match="splits into an undecoded"):
        predict_params(
            [16, 128, 28, 28],
            0.007058822084218264,
            0.007058819755911827,
            0.01411121804267168,
        )


def test_predict_params_rejects_channel_one():
    with pytest.raises(ValueError):
        predict_params([16, 1, 64, 3136], 0.007, 0.007, 0.014)


def test_predict_params_rejects_non_rank4():
    with pytest.raises(ValueError):
        predict_params([64, 56, 56], 0.007, 0.007, 0.014)
