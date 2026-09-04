#!/usr/bin/env python3
"""Axera-side model suite: the shared suite plus one Axera-specific fixture.

Re-exports `scripts/common/synthetic_models.py` (see that module) and adds
`axera_npu_compiled_leaf`, a minimal synthetic reproduction of the *real*
compiled-NPU-subgraph node shape confirmed against an actual AX650N and a
real `AXERA-TECH/YOLOv8` `.axmodel` (see `pulsar2_ops.py`'s docstring): a
single `op_type="neu mode"` node whose only declared input is the graph
input, with an initializer it depends on purely via a JSON-encoded attribute
reference rather than a graph edge. This exercises
`pulsar2_ops.has_out_of_band_npu_data` / `missing_npu_data` and the
`pulsar2_unsafe_for_simplify` worker path in CI, without needing the real
device.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only keep scripts/ on sys.path for the duration of this import: scripts/
# also holds directories like rfdetr/ with no __init__.py, which Python 3
# treats as importable namespace packages. Leaving scripts/ on sys.path for
# the rest of the process would make `import rfdetr` "succeed" as that empty
# namespace package instead of skipping via pytest.importorskip, and shadow
# the real one everywhere else it's checked for.
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

_AXERA_LEAF_NAME = "axera_npu_compiled_leaf"


def axera_npu_compiled_leaf() -> onnx.ModelProto:
    """Reproduces the real `neu mode` node shape (see this module's docstring).

    Deliberately not run through `onnx.checker.check_model`: `neu mode` has
    no registered schema (that's the point), same as the real file, which
    onnxsim itself only accepts via its own custom-operator tolerance, not
    plain ``onnx.checker``.
    """
    params = numpy_helper.from_array(np.zeros(64, dtype=np.uint8), "npu_params")
    graph_info = json.dumps(
        {
            "name": "leaf",
            "dotneus": [
                {
                    "neu_key": "npu_params",
                    "batch": 1,
                    "extra_inputs": [
                        {"name": "params", "const_data_key": "npu_params"}
                    ],
                }
            ],
        }
    )
    outputs_info = json.dumps({"y": ["FP32", [1, 4]]})
    node = helper.make_node(
        "neu mode",
        ["x"],
        ["y"],
        name="leaf",
        neu_name="leaf",
        npu_graph_info=graph_info,
        outputs_info=outputs_info,
        version=1,
    )
    graph = helper.make_graph(
        [node],
        _AXERA_LEAF_NAME,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [params],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )


def names() -> list:
    return [*_shared_names(), _AXERA_LEAF_NAME]


def build(name: str) -> onnx.ModelProto:
    if name == _AXERA_LEAF_NAME:
        return axera_npu_compiled_leaf()
    return _shared_build(name)


def all_models() -> dict:
    return {**_shared_all_models(), _AXERA_LEAF_NAME: axera_npu_compiled_leaf()}


if __name__ == "__main__":
    for n, m in all_models().items():
        print(f"{n:24} {len(m.graph.node)} nodes")
