"""Regression coverage for the elementwise-op DMA tile table predictor."""

import gzip
import os
import struct
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from dma_tile_predict import predict_params  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "dma_tiles")


def _npu_params(gz_name):
    with gzip.open(os.path.join(_FIXTURES, gz_name), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


@pytest.mark.parametrize(
    "gz_name,shape",
    [
        ("relu_1x16x32x32.axmodel.gz", [1, 16, 32, 32]),  # untiled: all-zero table
        ("relu_1x64x56x56.axmodel.gz", [1, 64, 56, 56]),  # 4 entries
        ("relu_1x64x56x56_rb1.axmodel.gz", [1, 64, 56, 56]),  # same shape, a rebuild
        ("relu_1x192x56x56.axmodel.gz", [1, 192, 56, 56]),  # 8 entries
        ("relu_4x64x56x56.axmodel.gz", [4, 64, 56, 56]),  # 8 entries, batch > 1
        ("relu_1x384x56x56.axmodel.gz", [1, 384, 56, 56]),  # 16 entries
        ("relu_16x128x28x28.axmodel.gz", [16, 128, 28, 28]),  # 16 entries
        ("relu_16x64x56x56.axmodel.gz", [16, 64, 56, 56]),  # 32 entries
    ],
)
def test_predict_params_matches_compiled_fixture(gz_name, shape):
    assert predict_params(shape) == _npu_params(gz_name)


def test_predict_params_untiled_is_all_zero():
    words = predict_params([1, 16, 8, 8])
    assert words == struct.pack("<10I", *([0] * 10))


@pytest.mark.parametrize(
    "shape",
    [
        [1, 33, 56, 56],  # channel count not a multiple of 4
        [3, 64, 56, 56],  # batch not a power of two
        [16, 1, 64, 3136],  # flattened middle dims don't match the literal shape
        [64, 56, 56],  # not rank 4
    ],
)
def test_predict_params_rejects_undecoded_shapes(shape):
    with pytest.raises(ValueError):
        predict_params(shape)
