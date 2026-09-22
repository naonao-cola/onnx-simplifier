"""Regression coverage for the Sub/Mul/Div two-input elementwise DMA tile
table predictor."""

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

from elementwise_two_input_tile_predict import predict_params  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "elementwise_two_input_tiles")


def _npu_params(gz_name):
    with gzip.open(os.path.join(_FIXTURES, gz_name), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


# Scales are each build's own out/quant/quant_axmodel.json values (see
# add_tile_predict.py's test file for how they're read out), not derivable from
# shape alone. Div's y_scale is a implausible constant (78431376.0) in every build
# here except the differently-ranged holdout -- likely the same tensor_configs
# hash-lookup picking up an unrelated intermediate tensor that this project's
# axera-quant-model-input-scales-overwrite-bug note already flagged as a recurring
# pitfall elsewhere. It is passed through unchanged below: Div's predictor never
# reads y_scale (see the module docstring -- Div has no header to compute it for),
# so the anomaly does not affect these results, but it is not independently
# explained here either.
@pytest.mark.parametrize(
    "op,gz_name,shape,x_scale,z_scale,y_scale",
    [
        # Sub: identical formula to Add (header + 15x body cycle).
        (
            "Sub",
            "sub_1x32x32x32.axmodel.gz",
            [1, 32, 32, 32],
            0.0070587280206382275,
            0.007058791816234589,
            0.014106064103543758,
        ),
        (
            "Sub",
            "sub_1x16x8x8_asym.axmodel.gz",
            [1, 16, 8, 8],
            0.007056748494505882,
            0.0007842928753234446,
            0.007761236280202866,
        ),
        (
            "Sub",
            "sub_1x64x56x56.axmodel.gz",
            [1, 64, 56, 56],
            0.007058808114379644,
            0.007058808580040932,
            0.014110462740063667,
        ),
        (
            "Sub",
            "sub_4x64x56x56.axmodel.gz",
            [4, 64, 56, 56],
            0.007058816496282816,
            0.007058822549879551,
            0.014110462740063667,
        ),
        (
            "Sub",
            "sub_16x64x56x56.axmodel.gz",
            [16, 64, 56, 56],
            0.007058820687234402,
            0.007058823015540838,
            0.014110462740063667,
        ),
        # Held out: shape and calibration ranges not used to derive the Sub formula
        # (it's inherited unmodified from Add's own already-validated one).
        (
            "Sub",
            "sub_holdout_8x64x28x28.axmodel.gz",
            [8, 64, 28, 28],
            0.003529408248141408,
            0.006666665896773338,
            0.010191429406404495,
        ),
        # Mul: no header, 15x body cycle only.
        (
            "Mul",
            "mul_1x32x32x32.axmodel.gz",
            [1, 32, 32, 32],
            0.0070587280206382275,
            0.007058791816234589,
            0.0063285427168011665,
        ),
        (
            "Mul",
            "mul_1x16x8x8_asym.axmodel.gz",
            [1, 16, 8, 8],
            0.007056748494505882,
            0.0007842928753234446,
            0.0006907600909471512,
        ),
        (
            "Mul",
            "mul_1x64x56x56.axmodel.gz",
            [1, 64, 56, 56],
            0.007058808114379644,
            0.007058808580040932,
            0.006348197814077139,
        ),
        (
            "Mul",
            "mul_4x64x56x56.axmodel.gz",
            [4, 64, 56, 56],
            0.007058816496282816,
            0.007058822549879551,
            0.006348197814077139,
        ),
        (
            "Mul",
            "mul_16x64x56x56.axmodel.gz",
            [16, 64, 56, 56],
            0.007058820687234402,
            0.007058823015540838,
            0.006348197814077139,
        ),
        (
            "Mul",
            "mul_holdout_8x64x28x28.axmodel.gz",
            [8, 64, 28, 28],
            0.003529408248141408,
            0.006666665896773338,
            0.0052894242107868195,
        ),
        # Div: no header, 15x body cycle only.
        (
            "Div",
            "div_1x32x32x32.axmodel.gz",
            [1, 32, 32, 32],
            0.0070587280206382275,
            0.007058791816234589,
            78431376.0,
        ),
        (
            "Div",
            "div_1x16x8x8_asym.axmodel.gz",
            [1, 16, 8, 8],
            0.007056748494505882,
            0.0007843000930733979,
            78431376.0,
        ),
        (
            "Div",
            "div_1x64x56x56.axmodel.gz",
            [1, 64, 56, 56],
            0.007058808114379644,
            0.007058808580040932,
            78431376.0,
        ),
        (
            "Div",
            "div_4x64x56x56.axmodel.gz",
            [4, 64, 56, 56],
            0.007058816496282816,
            0.007058822549879551,
            78431376.0,
        ),
        (
            "Div",
            "div_16x64x56x56.axmodel.gz",
            [16, 64, 56, 56],
            0.007058820687234402,
            0.007058823015540838,
            78431376.0,
        ),
        (
            "Div",
            "div_holdout_8x64x28x28.axmodel.gz",
            [8, 64, 28, 28],
            0.003529408248141408,
            0.005882352590560913,
            0.01759788766503334,
        ),
    ],
)
def test_predict_params_matches_compiled_fixture(
    op, gz_name, shape, x_scale, z_scale, y_scale
):
    assert predict_params(op, shape, x_scale, z_scale, y_scale) == _npu_params(gz_name)


def test_predict_params_rejects_unknown_op():
    with pytest.raises(ValueError, match="unknown op"):
        predict_params("Pow", [1, 8, 8, 8], 0.007, 0.007, 0.014)


@pytest.mark.parametrize("op", ["Sub", "Mul", "Div"])
def test_predict_params_rejects_split_regime(op):
    with pytest.raises(ValueError, match="splits into an undecoded"):
        predict_params(op, [16, 128, 28, 28], 0.007, 0.007, 0.014)


@pytest.mark.parametrize("op", ["Sub", "Mul", "Div"])
def test_predict_params_rejects_non_rank4(op):
    with pytest.raises(ValueError, match="rank-4"):
        predict_params(op, [8, 8, 8], 0.007, 0.007, 0.014)
