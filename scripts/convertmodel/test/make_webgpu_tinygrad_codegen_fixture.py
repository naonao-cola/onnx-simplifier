#!/usr/bin/env python3
"""Emits small ``.onnx`` fixtures with a tinygrad-generated WebGPU program
attached (``onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel`` /
``generate_resize_kernel``), so ``webgpu_tinygrad_codegen.test.mjs`` can
dispatch a *real tinygrad-rendered* kernel against a real WebGPU device and
check its output -- the first time any tinygrad-generated kernel (as
opposed to the hand-written Add kernel ``make_webgpu_kernel_fixture.py``
emits) actually runs on a GPU rather than just being checked numerically
offline against ``onnx.reference.ReferenceEvaluator`` (which is all
``tests/test_webgpu_tinygrad_codegen.py`` does, on the Python side alone).

Unlike ``make_webgpu_kernel_fixture.py``, this needs the optional
``tinygrad`` package (``pip install onnxsim[webgpu-codegen]``) in addition
to ``onnx``/``numpy`` -- but only to *generate* the fixture, run once here,
offline. The committed fixture files (the ``.onnx`` bytes, already carrying
the rendered WGSL, plus this script's ``.json`` manifest of concrete input
values and the expected output) are all ``webgpu_tinygrad_codegen.test.mjs``
reads; that CI job never installs Python or tinygrad, same as
``make_webgpu_kernel_fixture.py``'s own fixture.

Regenerate::

    pip install onnx numpy 'tinygrad==0.14.0'
    python3 make_webgpu_tinygrad_codegen_fixture.py
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
    """See ``make_webgpu_kernel_fixture.py``'s own copy of this helper for
    why: registers a stub "onnxsim" package so the module's own ``from
    onnxsim.x import y`` statements resolve without pulling in
    ``onnxsim/__init__.py`` (and therefore the compiled extension this
    script does not need).
    """
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
_load_onnxsim_module_without_package_init(
    "webgpu_kernel_metadata", "onnxsim/webgpu_kernel_metadata.py"
)
codegen = _load_onnxsim_module_without_package_init(
    "webgpu_tinygrad_codegen", "onnxsim/webgpu_tinygrad_codegen.py"
)


def _flat(array: np.ndarray):
    return [float(v) for v in array.reshape(-1)]


def _make_conv3d_fixture():
    """A tiny 3-D Conv (the gap ``onnxsim.webgpu_target.check_webgpu_conv3d_support``
    flags) -- small enough to keep the fixture JSON small, but a real 5-D
    (NCDHW) computation, not a 2-D one in disguise.
    """
    rng = np.random.default_rng(0)
    x_shape, w_shape = (1, 2, 4, 4, 4), (2, 2, 3, 3, 3)
    x = rng.standard_normal(x_shape).astype(np.float32)
    w = rng.standard_normal(w_shape).astype(np.float32)

    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        g (float{list(x_shape)} x) => (float[?] y)
        {{
          y = Conv<kernel_shape = [3, 3, 3]>(x, w)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    model.graph.node[0].name = "conv3d_node"

    import onnx

    onnx.checker.check_model(model)
    codegen.generate_conv_kernel(model, "conv3d_node")

    from onnx.reference import ReferenceEvaluator

    (y_ref,) = ReferenceEvaluator(model).run(None, {"x": x})

    out_path = os.path.join(HERE, "webgpu_tinygrad_conv3d.onnx")
    onnx.save(model, out_path)
    return {
        "file": "webgpu_tinygrad_conv3d.onnx",
        "nodeName": "conv3d_node",
        "outputName": "y",
        "inputs": {
            "x": {"shape": list(x_shape), "data": _flat(x)},
            "w": {"shape": list(w_shape), "data": _flat(w)},
        },
        "expectedOutput": {"shape": list(y_ref.shape), "data": _flat(y_ref)},
    }, out_path


def _make_resize_fixture():
    """A 4-D ``Resize`` with align_corners downsampling at an exact 2x ratio
    (the gap ``onnxsim.webgpu_target.check_webgpu_resize_support`` flags,
    scoped to the exact-integer-ratio case ``generate_resize_kernel``
    supports -- see that function's own docstring/comment for why).
    """
    rng = np.random.default_rng(1)
    x_shape, scales = (1, 2, 8, 8), [1.0, 1.0, 0.5, 0.5]
    x = rng.standard_normal(x_shape).astype(np.float32)

    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 13]>
        g (float{list(x_shape)} x) => (float[?] y)
        {{
          y = Resize<mode = "linear", coordinate_transformation_mode = "align_corners">(x, , scales)
        }}
        """
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(scales, dtype=np.float32), "scales")
    )
    model.graph.node[0].name = "resize_node"

    import onnx

    onnx.checker.check_model(model)
    codegen.generate_resize_kernel(model, "resize_node")

    from onnx.reference import ReferenceEvaluator

    (y_ref,) = ReferenceEvaluator(model).run(None, {"x": x})

    out_path = os.path.join(HERE, "webgpu_tinygrad_resize.onnx")
    onnx.save(model, out_path)
    return {
        "file": "webgpu_tinygrad_resize.onnx",
        "nodeName": "resize_node",
        "outputName": "y",
        "inputs": {"x": {"shape": list(x_shape), "data": _flat(x)}},
        "expectedOutput": {"shape": list(y_ref.shape), "data": _flat(y_ref)},
    }, out_path


def main():
    conv3d_manifest, conv3d_path = _make_conv3d_fixture()
    resize_manifest, resize_path = _make_resize_fixture()

    manifest = {"conv3d": conv3d_manifest, "resize": resize_manifest}
    manifest_path = os.path.join(HERE, "webgpu_tinygrad_codegen_fixture.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("wrote", conv3d_path)
    print("wrote", resize_path)
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
