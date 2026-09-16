#!/usr/bin/env python3
"""Emits ``webgpu_kernel_tuning_multi_conv_fixture.onnx`` -- a plain model
with **two independent, named** ``Conv`` nodes (no custom kernel metadata
attached to either) -- so ``webgpu_kernel_tuner_ui.test.mjs`` can exercise
the "Tune full graph" button's actual batching: tune every Conv node in one
click, then export one model carrying both winners.

Unlike ``make_webgpu_kernel_tuning_fixture.py`` (one Conv node -- proves a
single node's own tune/export round trip), this is deliberately the
smallest fixture that can prove the *batch* logic specifically: more than
one node actually gets tuned, and attachWebgpuKernelSpec's own
bytes-in/bytes-out chaining actually carries more than one winner into a
single exported file, not just "the last one".

No tinygrad import needed here at all -- unlike the single-Conv fixture,
this script never calls onnxsim.webgpu_kernel_tuning itself (the actual
tuning happens live, in the browser, in the test); it only needs a real
model and its ground-truth outputs.

Regenerate (needs ``onnx`` and ``numpy``)::

    pip install onnx numpy
    python3 make_webgpu_kernel_tuning_multi_conv_fixture.py
"""

import json
import os

import numpy as np
import onnx
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))


def _flat(array: np.ndarray):
    return [float(v) for v in array.reshape(-1)]


def main():
    x_shape, w_shape = (1, 4, 16, 16), (4, 4, 3, 3)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(x_shape).astype(np.float32)
    w1 = rng.standard_normal(w_shape).astype(np.float32)
    w2 = rng.standard_normal(w_shape).astype(np.float32)

    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        g (float{list(x_shape)} x) => (float[?,?,?,?] y1, float[?,?,?,?] y2)
        {{
          y1 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w1)
          y2 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w2)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(w1, "w1"))
    model.graph.initializer.append(numpy_helper.from_array(w2, "w2"))
    # onnx.parser never assigns node names -- give both real ones, the same
    # convention make_webgpu_kernel_tuning_fixture.py's own conv_node uses.
    model.graph.node[0].name = "conv_a"
    model.graph.node[1].name = "conv_b"

    onnx.checker.check_model(model)
    y1_ref, y2_ref = ReferenceEvaluator(model).run(None, {"x": x})

    manifest = {
        # "inputs" here means "every tensor a dispatch needs bound as a GPU
        # buffer" -- w1/w2 are ordinary graph initializers in the model
        # itself, but the tuned kernel's own spec binds them as plain tensor
        # bindings by name just like x, so a caller dispatching the exported
        # spec directly (bypassing onnxruntime-web, which would resolve an
        # initializer itself) needs their real values here too.
        "inputs": {
            "x": {"shape": list(x_shape), "data": _flat(x)},
            "w1": {"shape": list(w_shape), "data": _flat(w1)},
            "w2": {"shape": list(w_shape), "data": _flat(w2)},
        },
        "nodes": {
            "conv_a": {"outputName": "y1", "expectedOutput": {"shape": list(y1_ref.shape), "data": _flat(y1_ref)}},
            "conv_b": {"outputName": "y2", "expectedOutput": {"shape": list(y2_ref.shape), "data": _flat(y2_ref)}},
        },
    }

    manifest_path = os.path.join(HERE, "webgpu_kernel_tuning_multi_conv_fixture.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    model_path = os.path.join(HERE, "webgpu_kernel_tuning_multi_conv_fixture.onnx")
    onnx.save(model, model_path)

    print(f"wrote {manifest_path}")
    print(f"wrote {model_path} (two independent Conv nodes: conv_a, conv_b)")


if __name__ == "__main__":
    main()
