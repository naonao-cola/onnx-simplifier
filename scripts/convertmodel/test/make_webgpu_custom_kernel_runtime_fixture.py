#!/usr/bin/env python3
"""Emits the ``pre``/``post`` ``.onnx`` fixtures and manifest
``webgpu_custom_kernel_runtime.test.mjs`` needs to prove
``webgpu_custom_kernel_runtime.mjs`` can actually run a real model through
onnxruntime-web with one node excised and replaced by a tinygrad-generated
WebGPU program -- the runtime built in
``onnxsim/webgpu_custom_kernel_runtime.py`` (graph surgery) +
``scripts/convertmodel/webgpu_custom_kernel_runtime.mjs`` (session
orchestration + GPU-buffer splice).

The model is ``Relu(x) -> Conv3D -> Relu -> y``: the two ``Relu``s are
ordinary ops onnxruntime-web's WebGPU EP runs natively (they become the
``pre``/``post`` sessions), and the ``Conv3D`` in the middle is the gap
``onnxsim.webgpu_target.check_webgpu_conv3d_support`` flags (real
onnxruntime-web WebGPU builds don't accept a 3-D spatial ``Conv`` --
unverified here directly, but this is exactly the same shape
``webgpu_tinygrad_codegen.test.mjs`` already generates a real kernel for).

Needs the optional ``tinygrad`` package in addition to ``onnx``/``numpy``
(only to *generate* the fixture -- see
``make_webgpu_tinygrad_codegen_fixture.py``'s own docstring for why that's
fine: the committed fixture files are all the CI job reads).

Regenerate::

    pip install onnx numpy 'tinygrad==0.14.0'
    python3 make_webgpu_custom_kernel_runtime_fixture.py
"""

import importlib.util
import json
import os
import sys
import types

import numpy as np
from onnx import numpy_helper, parser

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..", "..")


def _load_onnxsim_module_without_package_init(name, relative_path):
    """See ``make_webgpu_kernel_fixture.py``'s own copy of this helper."""
    if "onnxsim" not in sys.modules:
        stub = types.ModuleType("onnxsim")
        stub.__path__ = []
        sys.modules["onnxsim"] = stub
    spec = importlib.util.spec_from_file_location(
        f"onnxsim.{name}", os.path.join(REPO_ROOT, relative_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"onnxsim.{name}"] = module
    spec.loader.exec_module(module)
    return module


_load_onnxsim_module_without_package_init("_rich_compat", "onnxsim/_rich_compat.py")
_load_onnxsim_module_without_package_init("model_info", "onnxsim/model_info.py")
_load_onnxsim_module_without_package_init("vitisai_target", "onnxsim/vitisai_target.py")
webgpu_kernel_metadata = _load_onnxsim_module_without_package_init(
    "webgpu_kernel_metadata", "onnxsim/webgpu_kernel_metadata.py"
)
codegen = _load_onnxsim_module_without_package_init(
    "webgpu_tinygrad_codegen", "onnxsim/webgpu_tinygrad_codegen.py"
)
runtime = _load_onnxsim_module_without_package_init(
    "webgpu_custom_kernel_runtime", "onnxsim/webgpu_custom_kernel_runtime.py"
)


def _flat(array: np.ndarray):
    return [float(v) for v in array.reshape(-1)]


def main():
    rng = np.random.default_rng(0)
    x_shape, w_shape, k = (1, 2, 4, 4, 4), (2, 2, 3, 3, 3), 3
    # Valid (no padding, unit stride) convolution: out = in - k + 1 per
    # spatial dim -- declared explicitly below rather than left as `[?]`
    # (fine for onnx.reference.ReferenceEvaluator, lenient about
    # declared-vs-inferred shape, but a real onnxruntime-web session's own
    # shape inference rejects a rank mismatch outright once the post
    # sub-graph infers y's real rank (5) against a `[?]`-declared one (1):
    # "Mismatch between number of inferred and declared dimensions.").
    y_shape = (x_shape[0], w_shape[0]) + tuple(d - k + 1 for d in x_shape[2:])
    x = rng.standard_normal(x_shape).astype(np.float32)
    w = rng.standard_normal(w_shape).astype(np.float32)

    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        g (float{list(x_shape)} x) => (float{list(y_shape)} y)
        {{
          pre = Relu(x)
          conv_out = Conv<kernel_shape = [{k}, {k}, {k}]>(pre, w)
          y = Relu(conv_out)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    for node in model.graph.node:
        if "pre" in node.output:
            node.name = "pre_relu"
        elif "conv_out" in node.output:
            node.name = "conv_node"
        elif "y" in node.output:
            node.name = "post_relu"

    import onnx

    onnx.checker.check_model(model)
    codegen.generate_conv_kernel(model, "conv_node")
    spec = webgpu_kernel_metadata.read_webgpu_kernel(model, "conv_node")

    from onnx.reference import ReferenceEvaluator

    (y_ref,) = ReferenceEvaluator(model).run(None, {"x": x})

    split = runtime.split_around_node(model, "conv_node")
    onnx.checker.check_model(split.pre)
    onnx.checker.check_model(split.post)

    # split_around_node's own contract: node inputs the pre-session's own
    # outputs don't cover (here, the Conv weight "w", an initializer -- never
    # a node output, so never a pre boundary) must be supplied directly by
    # the caller. "pre" (the excised node's other input) *is* covered, via
    # the pre-session's own GPU output.
    pre_output_names = {o.name for o in split.pre.graph.output}
    assert pre_output_names == {"pre"}, pre_output_names
    assert list(split.node.input) == ["pre", "w"]
    node_output_name = split.node.output[0]
    # Relu is shape-preserving, so conv_out's shape is exactly y's.
    node_output_shape = y_ref.shape

    pre_path = os.path.join(HERE, "webgpu_custom_kernel_runtime_pre.onnx")
    post_path = os.path.join(HERE, "webgpu_custom_kernel_runtime_post.onnx")
    onnx.save(split.pre, pre_path)
    onnx.save(split.post, post_path)

    manifest = {
        "preFile": os.path.basename(pre_path),
        "postFile": os.path.basename(post_path),
        "nodeName": "conv_node",
        "spec": spec.to_json(),
        "graphInput": {"name": "x", "shape": list(x_shape), "data": _flat(x)},
        "extraInputs": {"w": {"shape": list(w_shape), "data": _flat(w)}},
        "nodeOutputs": [
            {
                "name": node_output_name,
                "dims": list(node_output_shape),
                "dataType": "float32",
            }
        ],
        "finalOutputName": "y",
        "expectedOutput": {"shape": list(y_ref.shape), "data": _flat(y_ref)},
    }
    manifest_path = os.path.join(HERE, "webgpu_custom_kernel_runtime_fixture.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("wrote", pre_path)
    print("wrote", post_path)
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
