#!/usr/bin/env python3
"""Edgeai-side model suite: the shared suite plus one TIDL-specific fixture.

Re-exports `scripts/common/synthetic_models.py` (see that module) and adds
`edgeai_dynamic_batch_leaf`: a graph with a symbolic batch dimension, to
exercise `tidl_ops.has_dynamic_shape` -- TIDL requires every input shape to
be fully static (see `tidl_ops.py`'s docstring), a constraint none of the
shared suite's fixed-shape models trip.
"""

from __future__ import annotations

import os
import sys

import onnx
from onnx import TensorProto, helper

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only keep scripts/ on sys.path for the duration of this import -- see
# scripts/axera/models.py's docstring for why (namespace-package shadowing
# for directories like scripts/rfdetr with no __init__.py).
_inserted = _SCRIPTS_DIR not in sys.path
if _inserted:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from common.synthetic_models import (  # noqa: E402,F401
        all_models as _shared_all_models,
        build as _shared_build,
        conv_bn_relu,
        foldable_shape_reshape,
        matmul_bias_tanh,
        names as _shared_names,
        redundant_transpose,
        sigmoid_mul_swish,
    )
finally:
    if _inserted:
        sys.path.remove(_SCRIPTS_DIR)

_DYNAMIC_BATCH_LEAF_NAME = "edgeai_dynamic_batch_leaf"


def edgeai_dynamic_batch_leaf() -> onnx.ModelProto:
    """Conv -> Relu with a symbolic ("N") batch dimension on the input.

    Deliberately not run through the shared `_model` helper (which always
    builds fully-static shapes): the point of this fixture is the one
    dimension every other model in this suite lacks.
    """
    w = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 3, 3, 3], [0.0] * (4 * 3 * 3 * 3)
    )
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    x_type = helper.make_tensor_value_info("x", TensorProto.FLOAT, None)
    x_type.type.tensor_type.shape.dim.add().dim_param = "N"
    for d in (3, 8, 8):
        x_type.type.tensor_type.shape.dim.add().dim_value = d
    y_type = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    y_type.type.tensor_type.shape.dim.add().dim_param = "N"
    for d in (4, 8, 8):
        y_type.type.tensor_type.shape.dim.add().dim_value = d
    graph = helper.make_graph(nodes, _DYNAMIC_BATCH_LEAF_NAME, [x_type], [y_type], [w])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


def all_models():
    models = _shared_all_models()
    models[_DYNAMIC_BATCH_LEAF_NAME] = edgeai_dynamic_batch_leaf()
    return models


def names():
    return [*_shared_names(), _DYNAMIC_BATCH_LEAF_NAME]


def build(name: str) -> onnx.ModelProto:
    if name == _DYNAMIC_BATCH_LEAF_NAME:
        return edgeai_dynamic_batch_leaf()
    return _shared_build(name)
