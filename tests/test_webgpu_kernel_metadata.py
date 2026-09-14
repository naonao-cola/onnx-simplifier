"""Tests for ``onnxsim.webgpu_kernel_metadata`` -- the schema for attaching a
custom WebGPU kernel to a node's own ``metadata_props`` and reading it back.

Models are built via the ONNX text format parser (see CLAUDE.md's testing
guidance). ``onnx.parser`` never assigns ``NodeProto.name`` -- confirmed by
inspection, since the text format has no syntax for it -- so ``_named`` sets
it programmatically after parsing, the documented fallback for what the text
form can't express.
"""

import pytest
from onnx import parser

from onnxsim.webgpu_kernel_metadata import (
    WebgpuKernelBinding,
    WebgpuKernelSpec,
    attach_webgpu_kernel,
    list_webgpu_kernels,
    read_webgpu_kernel,
)


def _model(body, opset=17, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _named(model, output_name, node_name):
    """Sets the ``NodeProto.name`` of the node producing ``output_name`` --
    see this file's module docstring for why this can't be done in the text
    form itself.
    """
    for node in model.graph.node:
        if output_name in node.output:
            node.name = node_name
            return model
    raise AssertionError(f"no node producing {output_name!r}")


def _add_model():
    model = _model(
        """
        g (float[4] a, float[4] b) => (float[4] c)
        {
          c = Add(a, b)
        }
        """
    )
    return _named(model, "c", "add_node")


_WGSL = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  c[gid.x] = a[gid.x] + b[gid.x];
}
"""

_BINDINGS = [
    WebgpuKernelBinding(tensor="a", group=0, binding=0, access="read"),
    WebgpuKernelBinding(tensor="b", group=0, binding=1, access="read"),
    WebgpuKernelBinding(tensor="c", group=0, binding=2, access="read_write"),
]


def test_attach_and_read_round_trip():
    model = _add_model()
    attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1, 1), _BINDINGS)

    spec = read_webgpu_kernel(model, "add_node")
    assert spec == WebgpuKernelSpec(
        wgsl=_WGSL, entry_point="main", dispatch=(1, 1, 1), bindings=tuple(_BINDINGS)
    )


def test_attach_returns_mutated_model_for_chaining():
    model = _add_model()
    returned = attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1, 1), _BINDINGS)
    assert returned is model


def test_read_missing_kernel_returns_none():
    model = _add_model()
    assert read_webgpu_kernel(model, "add_node") is None


def test_read_missing_node_returns_none():
    model = _add_model()
    assert read_webgpu_kernel(model, "does_not_exist") is None


def test_attach_unknown_node_name_raises():
    model = _add_model()
    with pytest.raises(ValueError, match="no node named"):
        attach_webgpu_kernel(model, "does_not_exist", _WGSL, "main", (1, 1, 1), _BINDINGS)


def test_attach_empty_node_name_raises():
    model = _add_model()
    with pytest.raises(ValueError, match="non-empty"):
        attach_webgpu_kernel(model, "", _WGSL, "main", (1, 1, 1), _BINDINGS)


def test_attach_wrong_dispatch_length_raises():
    model = _add_model()
    with pytest.raises(ValueError, match="dispatch must have exactly 3"):
        attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1), _BINDINGS)


def test_attach_binding_to_foreign_tensor_raises():
    model = _add_model()
    bad_binding = [WebgpuKernelBinding(tensor="not_a_node_tensor", group=0, binding=0)]
    with pytest.raises(ValueError, match="not one of"):
        attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1, 1), bad_binding)


def test_attach_invalid_access_raises():
    model = _add_model()
    bad_binding = [WebgpuKernelBinding(tensor="a", group=0, binding=0, access="write_only")]
    with pytest.raises(ValueError, match="access"):
        attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1, 1), bad_binding)


def test_attach_overwrites_existing_kernel():
    model = _add_model()
    attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1, 1), _BINDINGS)
    attach_webgpu_kernel(model, "add_node", _WGSL, "main", (2, 3, 4), _BINDINGS)

    spec = read_webgpu_kernel(model, "add_node")
    assert spec.dispatch == (2, 3, 4)
    # exactly one metadata entry, not two stale + fresh copies
    node = model.graph.node[0]
    kernel_entries = [e for e in node.metadata_props if e.key.endswith("webgpu_kernel")]
    assert len(kernel_entries) == 1


def test_list_webgpu_kernels_only_lists_attached_nodes():
    model = _model(
        """
        g (float[4] a, float[4] b, float[4] d) => (float[4] c, float[4] e)
        {
          c = Add(a, b)
          e = Add(c, d)
        }
        """
    )
    _named(model, "c", "add_node")
    _named(model, "e", "second_add")
    attach_webgpu_kernel(model, "add_node", _WGSL, "main", (1, 1, 1), _BINDINGS)

    kernels = list_webgpu_kernels(model)
    assert [name for name, _ in kernels] == ["add_node"]
    assert kernels[0][1].entry_point == "main"


def test_webgpu_kernel_spec_json_round_trip():
    spec = WebgpuKernelSpec(
        wgsl=_WGSL, entry_point="main", dispatch=(1, 2, 3), bindings=tuple(_BINDINGS)
    )
    assert WebgpuKernelSpec.from_json(spec.to_json()) == spec
