"""Regression coverage for the Transpose real-graph composition check."""

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

from transpose_compose_real_check import (  # noqa: E402
    activation_chain_model,
    weight_chain_model,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "transpose_compose_real")


@pytest.mark.parametrize(
    "fixture,expected_inputs,expected_output",
    [
        ("weight_chain_1x64x64x9.axmodel.gz", ["w"], [1, 1, 64, 576]),
        (
            "activation_chain_16x1x576x3136.axmodel.gz",
            ["mul_out", "other"],
            [16, 1, 64, 576],
        ),
    ],
)
def test_composed_fixture_is_a_single_fused_node(
    fixture, expected_inputs, expected_output
):
    with gzip.open(os.path.join(_FIXTURES, fixture), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    # Pulsar2 fuses the whole Reshape/Transpose/(Reshape|MatMul) chain into one
    # compiled unit -- there is no separately addressable "Transpose" node left.
    assert [n.op_type for n in model.graph.node] == ["neu mode"]
    assert [v.name for v in model.graph.input] == expected_inputs
    out = model.graph.output[0]
    assert [d.dim_value for d in out.type.tensor_type.shape.dim] == expected_output


def test_weight_chain_model_matches_the_real_step_graph_shapes():
    model = weight_chain_model()
    assert [n.op_type for n in model.graph.node] == ["Reshape", "Transpose", "Reshape"]
    transpose = model.graph.node[1]
    assert list(next(a.ints for a in transpose.attribute if a.name == "perm")) == [
        0,
        2,
        1,
        3,
    ]
    assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] == [
        64,
        64,
        3,
        3,
    ]


def test_activation_chain_model_matches_the_real_step_graph_shapes():
    model = activation_chain_model()
    assert [n.op_type for n in model.graph.node] == ["Reshape", "Transpose", "MatMul"]
    transpose = model.graph.node[1]
    assert list(next(a.ints for a in transpose.attribute if a.name == "perm")) == [
        0,
        1,
        3,
        2,
    ]
    shapes = [
        [d.dim_value for d in v.type.tensor_type.shape.dim] for v in model.graph.input
    ]
    assert shapes == [[16, 1, 64, 28224], [16, 1, 64, 3136]]
