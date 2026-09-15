#!/usr/bin/env python3
"""Emits a small ``.onnx`` fixture with a custom WebGPU kernel attached to
one node via ``onnxsim.webgpu_kernel_metadata.attach_webgpu_kernel``, so
``onnx_node_metadata.test.mjs`` can prove the JS-side hand-rolled protobuf
reader (``scripts/convertmodel/onnx_node_metadata.mjs``) actually agrees
with what the Python side wrote -- not just that each side's own round trip
works in isolation.

Regenerate (only ``onnx`` is needed on the Python side, not a built onnxsim
-- same convention as ``make_ep_placement_fixtures.py``'s own docstring, and
for the same reason: this runs in the wasm/convertmodel CI job, which never
builds onnxsim's compiled Python extension). ``onnxsim/webgpu_kernel_metadata.py``
itself has no dependency on that extension either (only ``onnx`` and
``onnxsim/model_info.py``, also extension-free) -- it's loaded directly by
file path below, bypassing ``onnxsim/__init__.py`` (which *does* pull the
extension in transitively), so this exercises the real, shipped
``attach_webgpu_kernel`` rather than a hand-duplicated stand-in, without
requiring onnxsim to be installed::

    python3 make_webgpu_kernel_fixture.py
"""

import importlib.util
import json
import os
import sys
import types

from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..", "..")


def _load_onnxsim_module_without_package_init(name, relative_path):
    """Loads a single onnxsim module by file path, registering a stub
    "onnxsim" package in ``sys.modules`` first so the module's own
    ``from onnxsim.x import y`` statements resolve without importing the
    real ``onnxsim/__init__.py`` (which pulls in the compiled extension this
    script deliberately avoids needing -- see this file's module docstring).
    """
    if "onnxsim" not in sys.modules:
        stub = types.ModuleType("onnxsim")
        stub.__path__ = []  # marks it as a package for submodule imports
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
webgpu_kernel_metadata = _load_onnxsim_module_without_package_init(
    "webgpu_kernel_metadata", "onnxsim/webgpu_kernel_metadata.py"
)
WebgpuKernelBinding = webgpu_kernel_metadata.WebgpuKernelBinding
WebgpuKernelSpec = webgpu_kernel_metadata.WebgpuKernelSpec
attach_webgpu_kernel = webgpu_kernel_metadata.attach_webgpu_kernel

WGSL = """\
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  c[gid.x] = a[gid.x] + b[gid.x];
}
"""

BINDINGS = [
    WebgpuKernelBinding.for_tensor("a", group=0, binding=0, access="read"),
    WebgpuKernelBinding.for_tensor("b", group=0, binding=1, access="read"),
    WebgpuKernelBinding.for_tensor("c", group=0, binding=2, access="read_write"),
]

N = 256
DISPATCH = (N // 64, 1, 1)


def main():
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[{N}] a, float[{N}] b) => (float[{N}] c)
        {{
          c = Add(a, b)
        }}
        """
    )
    model.graph.node[0].name = "add_node"
    spec = WebgpuKernelSpec.single_step(WGSL, "main", DISPATCH, BINDINGS)
    attach_webgpu_kernel(model, "add_node", spec)

    import onnx

    onnx.checker.check_model(model)
    out_path = os.path.join(HERE, "webgpu_kernel_add.onnx")
    onnx.save(model, out_path)

    manifest = {
        "file": "webgpu_kernel_add.onnx",
        "nodeName": "add_node",
        "n": N,
        "entryPoint": "main",
        "dispatch": list(DISPATCH),
        "bindings": [b.to_json() for b in BINDINGS],
    }
    with open(os.path.join(HERE, "webgpu_kernel_fixture.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("wrote", out_path)
    print("wrote webgpu_kernel_fixture.json")


if __name__ == "__main__":
    main()
