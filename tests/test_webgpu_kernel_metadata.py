"""Tests for ``onnxsim.webgpu_kernel_metadata`` -- the schema for attaching a
custom WebGPU program (one or more kernel steps) to a node's own
``metadata_props`` and reading it back.

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
    WebgpuKernelStep,
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
    WebgpuKernelBinding.for_tensor("a", group=0, binding=0, access="read"),
    WebgpuKernelBinding.for_tensor("b", group=0, binding=1, access="read"),
    WebgpuKernelBinding.for_tensor("c", group=0, binding=2, access="read_write"),
]


def _single_step_spec(dispatch=(1, 1, 1), bindings=_BINDINGS):
    return WebgpuKernelSpec.single_step(_WGSL, "main", dispatch, bindings)


def test_attach_and_read_round_trip():
    model = _add_model()
    spec = _single_step_spec()
    attach_webgpu_kernel(model, "add_node", spec)

    assert read_webgpu_kernel(model, "add_node") == spec


def test_attach_returns_mutated_model_for_chaining():
    model = _add_model()
    returned = attach_webgpu_kernel(model, "add_node", _single_step_spec())
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
        attach_webgpu_kernel(model, "does_not_exist", _single_step_spec())


def test_attach_empty_node_name_raises():
    model = _add_model()
    with pytest.raises(ValueError, match="non-empty"):
        attach_webgpu_kernel(model, "", _single_step_spec())


def test_attach_no_steps_raises():
    model = _add_model()
    with pytest.raises(ValueError, match="at least one step"):
        attach_webgpu_kernel(model, "add_node", WebgpuKernelSpec(steps=()))


def test_single_step_wrong_dispatch_length_raises():
    with pytest.raises(ValueError, match="dispatch must have exactly 3"):
        WebgpuKernelSpec.single_step(_WGSL, "main", (1, 1), _BINDINGS)


def test_attach_binding_to_foreign_tensor_raises():
    bad_binding = [
        WebgpuKernelBinding.for_tensor("not_a_node_tensor", group=0, binding=0)
    ]
    model = _add_model()
    with pytest.raises(ValueError, match="not one of"):
        attach_webgpu_kernel(model, "add_node", _single_step_spec(bindings=bad_binding))


def test_attach_undeclared_intermediate_raises():
    bad_binding = [WebgpuKernelBinding.for_intermediate("scratch", group=0, binding=0)]
    model = _add_model()
    with pytest.raises(ValueError, match="not declared"):
        attach_webgpu_kernel(model, "add_node", _single_step_spec(bindings=bad_binding))


def test_attach_invalid_access_raises():
    bad_binding = [
        WebgpuKernelBinding(tensor="a", group=0, binding=0, access="write_only")
    ]
    model = _add_model()
    with pytest.raises(ValueError, match="access"):
        attach_webgpu_kernel(model, "add_node", _single_step_spec(bindings=bad_binding))


def test_attach_binding_needs_exactly_one_source():
    model = _add_model()
    none_set = [WebgpuKernelBinding(group=0, binding=0)]
    with pytest.raises(ValueError, match="exactly one"):
        attach_webgpu_kernel(model, "add_node", _single_step_spec(bindings=none_set))

    both_set = [WebgpuKernelBinding(group=0, binding=0, tensor="a", constant=(1.0,))]
    with pytest.raises(ValueError, match="exactly one"):
        attach_webgpu_kernel(model, "add_node", _single_step_spec(bindings=both_set))


def test_attach_overwrites_existing_kernel():
    model = _add_model()
    attach_webgpu_kernel(model, "add_node", _single_step_spec(dispatch=(1, 1, 1)))
    attach_webgpu_kernel(model, "add_node", _single_step_spec(dispatch=(2, 3, 4)))

    spec = read_webgpu_kernel(model, "add_node")
    assert spec.steps[0].dispatch == (2, 3, 4)
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
    attach_webgpu_kernel(model, "add_node", _single_step_spec())

    kernels = list_webgpu_kernels(model)
    assert [name for name, _ in kernels] == ["add_node"]
    assert kernels[0][1].steps[0].entry_point == "main"


def test_webgpu_kernel_spec_json_round_trip():
    spec = _single_step_spec(dispatch=(1, 2, 3))
    assert WebgpuKernelSpec.from_json(spec.to_json()) == spec


def test_constant_binding_json_round_trips_non_finite_values():
    import json

    binding = WebgpuKernelBinding.for_constant(
        [float("inf"), float("-inf"), float("nan"), 2.5], group=0, binding=0
    )
    # Standard JSON (unlike Python's own lenient json.dumps) has no
    # Infinity/NaN literal -- round-trip through an actual json.dumps/loads
    # pair (not just to_json()/from_json(), which never leave Python) to
    # prove the encoding is real strict-JSON-safe, matching what a browser's
    # JSON.parse will receive.
    raw = json.dumps(binding.to_json())
    restored = WebgpuKernelBinding.from_json(json.loads(raw))
    assert restored.constant[0] == float("inf")
    assert restored.constant[1] == float("-inf")
    assert restored.constant[2] != restored.constant[2]  # NaN != NaN
    assert restored.constant[3] == 2.5


def test_multi_step_program_with_intermediate_and_constant():
    # A two-step program (mirrors what tinygrad schedules for a fused
    # softmax-like op): step 1 writes a scratch buffer, step 2 reads it plus
    # a constant, and writes the real output.
    step1 = WebgpuKernelStep(
        wgsl="/* step1 */",
        entry_point="step1",
        dispatch=(1, 1, 1),
        bindings=(
            WebgpuKernelBinding.for_tensor("a", group=0, binding=0, access="read"),
            WebgpuKernelBinding.for_intermediate("scratch", group=0, binding=1),
        ),
    )
    step2 = WebgpuKernelStep(
        wgsl="/* step2 */",
        entry_point="step2",
        dispatch=(1, 1, 1),
        bindings=(
            WebgpuKernelBinding.for_intermediate(
                "scratch", group=0, binding=0, access="read"
            ),
            WebgpuKernelBinding.for_constant([1.0, 2.0], group=0, binding=1),
            WebgpuKernelBinding.for_tensor(
                "c", group=0, binding=2, access="read_write"
            ),
        ),
    )
    spec = WebgpuKernelSpec(steps=(step1, step2), intermediates={"scratch": 256})

    model = _add_model()
    attach_webgpu_kernel(model, "add_node", spec)
    round_tripped = read_webgpu_kernel(model, "add_node")

    assert round_tripped == spec
    assert len(round_tripped.steps) == 2
    assert round_tripped.intermediates == {"scratch": 256}
    assert round_tripped.steps[1].bindings[1].constant == (1.0, 2.0)
