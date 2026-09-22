"""Regression coverage for the first ReduceSum npu_params decode."""

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

from reducesum_decode import predict_main_offsets  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "reducesum_decode")


def _npu_params(gz_name):
    with gzip.open(os.path.join(_FIXTURES, gz_name), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    inits = {i.name: i for i in model.graph.initializer}
    return bytes(inits["npu_params"].raw_data)


@pytest.mark.parametrize(
    "gz_name,n,row",
    [
        ("a_16x1x64x576_ax0.axmodel.gz", 16, 64 * 576),
        ("f_16x1x128x784_ax0_synth.axmodel.gz", 16, 128 * 784),
    ],
)
def test_predicted_offsets_are_a_subset_of_the_real_table(gz_name, n, row):
    real = _npu_params(gz_name)
    words = struct.unpack(f"<{len(real) // 4}I", real)
    predicted = set(predict_main_offsets(n, row))
    assert predicted <= set(words)


def test_untiled_shapes_have_all_zero_params():
    for gz_name in ("c_16x1000_ax0.axmodel.gz", "d_16x1000_ax1.axmodel.gz"):
        real = _npu_params(gz_name)
        assert real == bytes(len(real))


def test_two_axis_reduction_does_not_match_the_single_axis_formula():
    # [16,1,64,3136] reduced over axes (0,3): the real graph's harder case.
    # The single-axis formula is not claimed to apply here -- confirm that
    # directly rather than silently, so a future fix to the predictor has a
    # regression test to satisfy.
    real = _npu_params("b_16x1x64x3136_ax03.axmodel.gz")
    words = struct.unpack(f"<{len(real) // 4}I", real)
    predicted = set(predict_main_offsets(16, 64 * 3136))
    assert not predicted <= set(words)
