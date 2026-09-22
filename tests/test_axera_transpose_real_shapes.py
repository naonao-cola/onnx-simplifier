"""Regression coverage for the real-ResNet18-shape Transpose template library."""

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

from transpose_real_shapes import (  # noqa: E402
    TEMPLATES,
    load_template_bytes,
    template_path,
)

_FIXTURE_DIR = os.path.join(_AXERA_DIR, "fixtures", "transpose_real")

# The 23 distinct (shape, perm) pairs of the real ResNet18 training step's 41
# Transpose node-instances (tallied via onnx.shape_inference over
# t6-r18fold/step.onnx), with each pair's node-instance count.
_STEP_PAIRS = {
    ((1, 64, 64, 9), (0, 2, 1, 3)): 4,
    ((16, 1, 576, 3136), (0, 1, 3, 2)): 4,
    ((1, 512, 512, 9), (0, 2, 1, 3)): 3,
    ((16, 1, 4608, 49), (0, 1, 3, 2)): 3,
    ((1, 256, 256, 9), (0, 2, 1, 3)): 3,
    ((16, 1, 2304, 196), (0, 1, 3, 2)): 3,
    ((1, 128, 128, 9), (0, 2, 1, 3)): 3,
    ((16, 1, 1152, 784), (0, 1, 3, 2)): 3,
    ((16, 512), (1, 0)): 1,
    ((512, 1000), (1, 0)): 1,
    ((1, 512, 256, 9), (0, 2, 1, 3)): 1,
    ((16, 1, 2304, 49), (0, 1, 3, 2)): 1,
    ((1, 512, 256, 1), (0, 2, 1, 3)): 1,
    ((16, 1, 256, 49), (0, 1, 3, 2)): 1,
    ((1, 256, 128, 9), (0, 2, 1, 3)): 1,
    ((16, 1, 1152, 196), (0, 1, 3, 2)): 1,
    ((1, 256, 128, 1), (0, 2, 1, 3)): 1,
    ((16, 1, 128, 196), (0, 1, 3, 2)): 1,
    ((1, 128, 64, 9), (0, 2, 1, 3)): 1,
    ((16, 1, 576, 784), (0, 1, 3, 2)): 1,
    ((1, 128, 64, 1), (0, 2, 1, 3)): 1,
    ((16, 1, 64, 784), (0, 1, 3, 2)): 1,
    ((16, 1, 147, 12544), (0, 1, 3, 2)): 1,
}

_NOISE_START, _NOISE_END = 301, 326


def test_all_23_real_step_pairs_are_covered():
    assert set(TEMPLATES) == set(_STEP_PAIRS)


def test_covers_all_41_node_instances():
    assert sum(_STEP_PAIRS.values()) == 41
    assert sum(1 for k in _STEP_PAIRS if k in TEMPLATES) == len(_STEP_PAIRS)


@pytest.mark.parametrize("shape,perm", sorted(_STEP_PAIRS))
def test_template_loads_and_matches_shape(shape, perm):
    data = load_template_bytes(shape, perm)
    model = onnx.load_model_from_string(data)
    assert [n.op_type for n in model.graph.node] == ["neu mode"]
    node = model.graph.node[0]
    assert list(node.input) == ["x"]
    assert list(node.output) == ["y"]
    in_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert in_dims == list(shape)
    assert out_dims == [shape[p] for p in perm]


def test_template_path_matches_load_template_bytes():
    shape, perm = (1, 64, 64, 9), (0, 2, 1, 3)
    path = template_path(shape, perm)
    assert os.path.exists(path)
    with gzip.open(path, "rb") as f:
        assert f.read() == load_template_bytes(shape, perm)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 8, 9),  # a plausible but unmeasured weight shape
        (16, 1, 64, 3136),  # a plausible but unmeasured activation shape
    ],
)
def test_unmeasured_shape_is_rejected(shape):
    perm = (0, 2, 1, 3) if shape[1] == shape[2] else (0, 1, 3, 2)
    with pytest.raises(ValueError, match="no verified Transpose template"):
        template_path(shape, perm)


def _mcode_and_params(data: bytes):
    model = onnx.load_model_from_string(data)
    inits = {i.name: i for i in model.graph.initializer}
    mcode = next(bytes(i.raw_data) for n, i in inits.items() if n.endswith("_neu"))
    params = bytes(inits["npu_params"].raw_data) if "npu_params" in inits else b""
    return mcode, params


@pytest.mark.parametrize(
    "shape,perm,oracle_name",
    [
        (
            (1, 64, 64, 9),
            (0, 2, 1, 3),
            "transpose_real_oracle_1x64x64x9_perm0213.axmodel.gz",
        ),
        (
            (16, 1, 576, 3136),
            (0, 1, 3, 2),
            "transpose_real_oracle_16x1x576x3136_perm0132.axmodel.gz",
        ),
        (
            (16, 512),
            (1, 0),
            "transpose_real_oracle_16x512_perm10.axmodel.gz",
        ),
    ],
)
def test_template_reproduces_an_independent_rebuild(shape, perm, oracle_name):
    """Each committed template was spot-checked against a FRESH, independent
    Pulsar2 rebuild of the same (shape, perm) -- confirming the committed
    fixture is a trustworthy, reproducible template rather than a one-off
    build artifact. The oracle here is that independent rebuild's own
    compiled output, committed alongside the template.
    """
    template_mc, template_pa = _mcode_and_params(load_template_bytes(shape, perm))
    with gzip.open(os.path.join(_FIXTURE_DIR, oracle_name), "rb") as f:
        oracle_data = f.read()
    oracle_mc, oracle_pa = _mcode_and_params(oracle_data)

    assert template_pa == oracle_pa

    assert len(template_mc) == len(oracle_mc)
    diffs = [i for i in range(len(template_mc)) if template_mc[i] != oracle_mc[i]]
    outside_noise = [i for i in diffs if not (_NOISE_START <= i < _NOISE_END)]
    assert outside_noise == []
