#!/usr/bin/env python3
"""Emits ``webgpu_node_tuning_gather_fixture.onnx`` -- a plain model with a
**Gather** node (a real "memory operation": tinygrad schedules it to its own
compute kernel, unlike a pure view op such as Reshape/Transpose -- see
webgpu_kernel_tuner.mjs's own module docstring) feeding a Relu, no custom
kernel metadata attached to either -- so
webgpu_kernel_tuner_ui.test.mjs/webgpu_kernel_profile_ui.test.mjs can exercise
the *generalized* ("any op type, via tinygrad's own OnnxRunner") tuner and
profiler end to end on an op onnxsim has no bespoke Conv-only codegen for at
all.

Gather is a deliberate choice, not just "any non-Conv op": its index input
(``idx``) is an ordinary runtime tensor as far as tinygrad's own kernel
*structure* is concerned (unlike, say, Reshape's shape input, which
tinygrad's OnnxRunner treats as a required python-const -- see
``tinygrad.nn.onnx.required_input_python_consts`` and
webgpu_kernel_tuner.mjs's own two-pass docstring), so this fixture exercises
the ordinary "fresh random leaf tensor" path for a non-float, non-Conv input
too, not just the python-const path a Reshape-shaped fixture would.

No tinygrad import needed here at all -- like
make_webgpu_kernel_tuning_multi_conv_fixture.py, this script never calls
onnxsim.webgpu_kernel_tuning itself (the actual tuning happens live, in the
browser, in the test); it only needs a real model and its ground-truth
outputs.

Regenerate (needs ``onnx`` and ``numpy``)::

    pip install onnx numpy
    python3 make_webgpu_node_tuning_fixture.py
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
    x_shape = (6, 4)
    idx = np.array([0, 2, 4], dtype=np.int64)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(x_shape).astype(np.float32)

    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        g (float{list(x_shape)} x) => (float[?,?] z)
        {{
          y = Gather<axis = 0>(x, idx)
          z = Relu(y)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(idx, "idx"))
    # onnx.parser never assigns node names -- give both real ones, the same
    # convention every other fixture here uses.
    model.graph.node[0].name = "gather_node"
    model.graph.node[1].name = "relu_node"

    # Sanity-check the *whole* model is valid ONNX and Relu(Gather(x, idx))
    # actually runs -- but the test itself tunes and dispatches the isolated
    # "gather_node" kernel alone (its own output is "y", not the graph's own
    # declared output "z"), so the ground truth it needs is Gather's own
    # output, not Relu's. axis=0 Gather is exactly numpy fancy indexing, so
    # this computes that directly rather than pulling "y" out as an
    # intermediate ReferenceEvaluator output.
    onnx.checker.check_model(model)
    ReferenceEvaluator(model).run(None, {"x": x})
    y_ref = x[idx]

    manifest = {
        # "inputs" here means "every tensor a dispatch needs bound as a GPU
        # buffer" -- idx is an ordinary graph initializer in the model
        # itself, but the tuned kernel's own spec binds it as a plain tensor
        # binding by name just like x (see
        # make_webgpu_kernel_tuning_multi_conv_fixture.py's own manifest
        # comment for why).
        "inputs": {
            "x": {"shape": list(x_shape), "data": _flat(x)},
            # dtype "int64" -- unlike every other fixture's own "inputs"
            # (all plain float32 data), Gather's own index input keeps its
            # real ONNX/tinygrad dtype: tinygrad's WGSL renderer has no
            # native 64-bit integer type, so it represents an int64 buffer
            # as two packed 32-bit words per element (low, high) -- exactly
            # a JS BigInt64Array's own little-endian byte layout -- rather
            # than reinterpreting it through Float32Array like every other
            # (real float) input here.
            "idx": {"shape": list(idx.shape), "data": [int(v) for v in idx], "dtype": "int64"},
        },
        "nodeName": "gather_node",
        "outputName": "y",
        "expectedOutput": {"shape": list(y_ref.shape), "data": _flat(y_ref)},
    }

    manifest_path = os.path.join(HERE, "webgpu_node_tuning_gather_fixture.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    model_path = os.path.join(HERE, "webgpu_node_tuning_gather_fixture.onnx")
    onnx.save(model, model_path)

    print(f"wrote {manifest_path}")
    print(f"wrote {model_path} (Gather -> Relu, no attached kernel metadata)")


if __name__ == "__main__":
    main()
