"""Tests for the opt-in ``rewrite_gridsample_to_gather`` pass.

Every model is a single 2-D ``GridSample`` node, built with the ONNX text
format parser (``onnx.parser``, no torch dependency) per CLAUDE.md's
convention for this repo's tests. Each is run through
``onnxsim.simplify(..., extra_optimizers=["rewrite_gridsample_to_gather"])``,
which numerically equivalence-checks the rewritten graph against the original
``GridSample`` node (via onnxruntime, or the onnx reference evaluator when
onnxruntime is not installed) on an actual random ``grid`` *input* -- not a
folded constant -- generated with a wide enough range to include values
outside ``[-1, 1]`` on every test, so padding-mode handling (not just
in-bounds sampling) is always exercised. ``input_data`` pins that grid (and
``X``) so the same values are used for both the original and rewritten graphs
-- see ``tests/test_pocket_tts.py``'s ``check_n=1`` / fixed-``input_data``
idiom, which this mirrors.
"""

import collections

import numpy as np
import pytest
from onnx import parser

import onnxsim


def _model(
    x_shape,
    grid_shape,
    out_shape,
    mode,
    padding_mode,
    align_corners,
    opset=20,
    ir_version=10,
):
    body = f"""
    <
      ir_version: {ir_version},
      opset_import: ["": {opset}]
    >
    agraph (float{x_shape} X, float{grid_shape} grid) => (float{out_shape} Y)
    {{
      Y = GridSample <mode="{mode}", padding_mode="{padding_mode}", align_corners={align_corners}> (X, grid)
    }}
    """
    return parser.parse_model(body)


def _rand_grid(rng, shape, lo=-1.6, hi=1.6):
    # A wide enough range that, with overwhelming probability, some values
    # fall outside [-1, 1] -- so zeros/border/reflection padding-mode
    # handling is actually exercised, not just in-bounds bilinear/nearest
    # sampling.
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def _simplify_and_check(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gridsample_to_gather"],
    )
    assert check_ok, "rewritten graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "GridSample" not in op_types, op_types
    assert "GatherND" in op_types, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# bilinear ("linear") x {align_corners} x {padding_mode}
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("align_corners", [0, 1])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_linear_all_padding_modes(padding_mode, align_corners):
    rng = np.random.RandomState(0)
    X = rng.randn(2, 3, 5, 7).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode="linear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# nearest x {padding_mode}, one align_corners setting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_nearest_all_padding_modes(padding_mode):
    rng = np.random.RandomState(1)
    X = rng.randn(2, 3, 5, 7).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode="nearest",
        padding_mode=padding_mode,
        align_corners=0,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# Grid values outside [-1, 1] -- explicit, dedicated coverage of padding-mode
# handling beyond what the wide random range above already exercises
# incidentally: a grid built entirely from out-of-range values (some barely
# so, some far enough to need more than one reflection fold).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_out_of_range_grid_values(padding_mode):
    X = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    # A mix of just-out-of-range and far-out-of-range (multi-reflection)
    # coordinates on both axes, broadcast across the batch/output grid.
    base = np.array(
        [
            [-1.2, 1.3],
            [2.5, -2.7],
            [5.9, -6.1],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    grid = np.broadcast_to(base, (2, 4, 4, 2)).copy()

    model = _model(
        "[2,3,4,4]",
        "[2,4,4,2]",
        "[2,3,4,4]",
        mode="linear",
        padding_mode=padding_mode,
        align_corners=0,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# Dynamic (symbolic) H/W -- the pass must derive H/W from Shape(X) at
# runtime, never assume a static shape.
# --------------------------------------------------------------------------- #


def test_dynamic_input_shape():
    rng = np.random.RandomState(2)
    X = rng.randn(2, 3, 9, 11).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[N,3,H,W]",
        "[N,4,6,2]",
        "[N,3,4,6]",
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


def test_dynamic_input_shape_reflection_nearest():
    # Also cover the reflection/nearest combination under a dynamic shape --
    # H/W feed directly into the per-axis reflect bounds and the rounding
    # path, both of which must stay purely runtime-derived.
    rng = np.random.RandomState(3)
    X = rng.randn(1, 2, 6, 8).astype(np.float32)
    grid = _rand_grid(rng, (1, 3, 5, 2))

    model = _model(
        "[N,2,H,W]",
        "[N,3,5,2]",
        "[N,2,3,5]",
        mode="nearest",
        padding_mode="reflection",
        align_corners=1,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})
