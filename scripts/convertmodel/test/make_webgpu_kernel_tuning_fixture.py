#!/usr/bin/env python3
"""Emits a fixture of several *alternative* WebGPU kernel candidates for the
same real ``Conv2D`` -- via ``onnxsim.webgpu_kernel_tuning.generate_kernel_candidates``
-- so ``webgpu_kernel_tuning.test.mjs`` can dispatch every one of them on a
real WebGPU device, time each with ``dispatchWebgpuProgram``'s own
``profile: true`` option, and check both that every candidate computes the
right answer and that the fastest one actually gets picked.

Unlike ``make_webgpu_tinygrad_codegen_fixture.py``'s fixture (a single,
already-attached-to-a-model kernel), this fixture is a plain JSON manifest
of several standalone :class:`~onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec`
candidates -- there's no ONNX node to attach any *one* of them to, since the
whole point is comparing several against each other, not shipping a single
picked winner.

Also saves ``webgpu_kernel_tuning_fixture.onnx`` -- the *same* Conv2D as an
ordinary, standalone ONNX model (no custom kernel spec attached at all) --
so ``webgpu_kernel_tuning_vs_webnn.test.mjs`` can run it through
onnxruntime-web's WebNN execution provider for a same-op comparison against
the tinygrad-tuned candidates above.

Regenerate (needs ``onnx``, ``numpy``, and ``tinygrad``)::

    pip install onnx numpy 'tinygrad==0.14.0'
    python3 make_webgpu_kernel_tuning_fixture.py
"""

import importlib.util
import json
import os
import sys
import types

import numpy as np
import onnx
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

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
_load_onnxsim_module_without_package_init(
    "webgpu_tinygrad_codegen", "onnxsim/webgpu_tinygrad_codegen.py"
)
tuning = _load_onnxsim_module_without_package_init(
    "webgpu_kernel_tuning", "onnxsim/webgpu_kernel_tuning.py"
)


def _flat(array: np.ndarray):
    return [float(v) for v in array.reshape(-1)]


def main():
    # A real Conv2D with a rich tuning search space -- verified directly
    # (see onnxsim/webgpu_kernel_tuning.py's own docstring) that a shape
    # like this one gives tinygrad's own get_kernel_actions dozens of
    # genuinely distinct candidates, unlike this repo's other (minimal
    # 3x3x3) Conv3D fixture.
    x_shape, w_shape = (1, 4, 16, 16), (4, 4, 3, 3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(x_shape).astype(np.float32)
    w = rng.standard_normal(w_shape).astype(np.float32)

    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        g (float{list(x_shape)} x) => (float[?,?,?,?] y)
        {{
          y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))

    (y_ref,) = ReferenceEvaluator(model).run(None, {"x": x})

    from tinygrad import Tensor

    xt = Tensor(x, device="WEBGPU")
    wt = Tensor(w, device="WEBGPU")
    yt = xt.conv2d(wt, padding=1)

    max_candidates = 8
    results = tuning.generate_kernel_candidates(
        {"x": xt, "w": wt, "y": yt}, "y", max_candidates=max_candidates
    )
    assert len(results) == 1, (
        f"expected exactly one scheduled kernel call, got {len(results)}"
    )
    candidates, intermediates = results[0]

    manifest = {
        "outputName": "y",
        "inputs": {
            "x": {"shape": list(x_shape), "data": _flat(x)},
            "w": {"shape": list(w_shape), "data": _flat(w)},
        },
        "expectedOutput": {"shape": list(y_ref.shape), "data": _flat(y_ref)},
        "candidates": [
            {
                "appliedOpts": candidates.applied_opts[i],
                "spec": candidates.spec_for(i, intermediates).to_json(),
            }
            for i in range(len(candidates.steps))
        ],
    }

    manifest_path = os.path.join(HERE, "webgpu_kernel_tuning_fixture.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    model_path = os.path.join(HERE, "webgpu_kernel_tuning_fixture.onnx")
    onnx.save(model, model_path)

    print(f"wrote {manifest_path} with {len(candidates.steps)} candidates")
    print(f"wrote {model_path} (the same Conv2D, as a plain runnable model)")


if __name__ == "__main__":
    main()
