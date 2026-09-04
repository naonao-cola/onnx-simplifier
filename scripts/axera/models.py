#!/usr/bin/env python3
"""Axera-side model suite: the shared suite plus two Axera-specific fixtures.

Re-exports `scripts/common/synthetic_models.py` (see that module) and adds:

- `axera_npu_compiled_leaf`: a minimal synthetic reproduction of the *real*
  compiled-NPU-subgraph node shape confirmed against an actual AX650N and a
  real `AXERA-TECH/YOLOv8` `.axmodel` (see `pulsar2_ops.py`'s docstring): a
  single `op_type="neu mode"` node whose only declared input is the graph
  input, with an initializer it depends on purely via a JSON-encoded
  attribute reference rather than a graph edge.
- `axera_llm_layer_leaf`: the real per-transformer-layer `.axmodel` shape
  confirmed against a real `Qwen/Qwen3-0.6B` build (see
  `pulsar2_docker.llm_build()`'s docstring): **two** `neu mode` nodes in one
  graph (decode + prefill), sharing one `npu_params` initializer, each with
  its own `K_cache`/`V_cache` graph inputs and `*_out` outputs.

Both exercise `pulsar2_ops.has_out_of_band_npu_data`/`missing_npu_data` and
the `pulsar2_unsafe_for_simplify` worker path in CI, without needing the
real device -- `axera_llm_layer_leaf` specifically checks the multi-NPU-node
case the CNN-only leaf never exercises.
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


_AXERA_LLM_LAYER_NAME = "axera_llm_layer_leaf"


def _neu_node(suffix: str, batch_shape: list) -> tuple:
    """One `neu mode` node + its matching value_info, in the confirmed real
    per-layer shape (decode `suffix=""`, prefill `suffix="_1"`)."""
    neu_key = f"subgraph_npu{suffix}_b1_neu"
    graph_info = json.dumps(
        {
            "name": f"subgraph_npu{suffix}",
            "dotneus": [
                {
                    "neu_key": neu_key,
                    "batch": 1,
                    "extra_inputs": [
                        {"name": "params", "const_data_key": "npu_params"}
                    ],
                }
            ],
        }
    )
    out_name = f"output{suffix}"
    outputs_info = json.dumps({out_name: ["BF16", batch_shape]})
    node = helper.make_node(
        "neu mode",
        [f"K_cache{suffix}", f"V_cache{suffix}", f"input{suffix}"],
        [f"K_cache_out{suffix}", f"V_cache_out{suffix}", out_name],
        name=f"subgraph_npu{suffix}",
        neu_name=f"subgraph_npu{suffix}",
        npu_graph_info=graph_info,
        outputs_info=outputs_info,
        version=1,
    )
    return node, neu_key


def axera_llm_layer_leaf() -> onnx.ModelProto:
    """Reproduces the real per-layer LLM `.axmodel` shape (see this module's
    docstring): two `neu mode` nodes (decode + prefill) sharing one
    `npu_params` initializer, each with its own `K_cache`/`V_cache` I/O.
    """
    decode_node, decode_key = _neu_node("", [1, 1, 8])
    prefill_node, prefill_key = _neu_node("_1", [1, 4, 8])

    # Real files have one initializer per referenced key: the shared weight
    # blob (npu_params) plus each subgraph's own per-neu blob (*_b1_neu).
    params = numpy_helper.from_array(np.zeros(64, dtype=np.uint8), "npu_params")
    decode_neu = numpy_helper.from_array(np.zeros(8, dtype=np.uint8), decode_key)
    prefill_neu = numpy_helper.from_array(np.zeros(8, dtype=np.uint8), prefill_key)

    def cache_io(suffix: str, shape: list) -> tuple:
        return (
            helper.make_tensor_value_info(f"K_cache{suffix}", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info(f"V_cache{suffix}", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info(f"input{suffix}", TensorProto.FLOAT, shape),
        )

    decode_io = cache_io("", [1, 1, 8])
    prefill_io = cache_io("_1", [1, 4, 8])

    graph = helper.make_graph(
        [decode_node, prefill_node],
        _AXERA_LLM_LAYER_NAME,
        [*decode_io, *prefill_io],
        [
            helper.make_tensor_value_info("K_cache_out", TensorProto.FLOAT, [1, 1, 8]),
            helper.make_tensor_value_info("V_cache_out", TensorProto.FLOAT, [1, 1, 8]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 8]),
            helper.make_tensor_value_info(
                "K_cache_out_1", TensorProto.FLOAT, [1, 4, 8]
            ),
            helper.make_tensor_value_info(
                "V_cache_out_1", TensorProto.FLOAT, [1, 4, 8]
            ),
            helper.make_tensor_value_info("output_1", TensorProto.FLOAT, [1, 4, 8]),
        ],
        [params, decode_neu, prefill_neu],
    )
    assert decode_key != prefill_key  # both keys must be distinct, like the real file
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )


def names() -> list:
    return [*_shared_names(), _AXERA_LEAF_NAME, _AXERA_LLM_LAYER_NAME]


def build(name: str) -> onnx.ModelProto:
    if name == _AXERA_LEAF_NAME:
        return axera_npu_compiled_leaf()
    if name == _AXERA_LLM_LAYER_NAME:
        return axera_llm_layer_leaf()
    return _shared_build(name)


def all_models() -> dict:
    return {
        **_shared_all_models(),
        _AXERA_LEAF_NAME: axera_npu_compiled_leaf(),
        _AXERA_LLM_LAYER_NAME: axera_llm_layer_leaf(),
    }


if __name__ == "__main__":
    for n, m in all_models().items():
        print(f"{n:24} {len(m.graph.node)} nodes")
