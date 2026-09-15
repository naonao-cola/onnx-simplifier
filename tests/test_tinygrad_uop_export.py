"""Tests for ``onnxsim.tinygrad_uop_export`` -- checks that exporting a real
tinygrad ``UOp`` graph (the same kind of per-kernel ``Ops.SINK``-rooted AST
``onnxsim.webgpu_tinygrad_codegen`` itself renders to WGSL) produces a
structurally valid, faithful ONNX representation: every ``UOp`` becomes
exactly one node, in the same topological positions, with the same
op/dtype/edges -- not just that ``onnx.checker`` doesn't raise.
"""

import pytest

pytest.importorskip("tinygrad")

import onnx  # noqa: E402

from onnxsim.tinygrad_uop_export import DOMAIN, uop_to_onnx_model  # noqa: E402


def _conv_kernel_ast():
    """A real per-kernel AST -- the same one
    ``onnxsim.webgpu_tinygrad_codegen._lower_tensor_program`` itself passes
    to ``to_program``/``WGSLRenderer`` -- from a small Conv2D, tagged for
    the ``WEBGPU`` device (never actually opened; see that module's own
    docstring for why no real GPU is needed just to schedule/render).
    """
    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops

    x = Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="WEBGPU")
    w = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device="WEBGPU")
    y = x.conv2d(w)
    linear = y.schedule_linear()
    kernel_calls = [
        u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK
    ]
    assert kernel_calls, "expected at least one real compute kernel to be scheduled"
    return kernel_calls[0].src[0]


def test_exports_one_node_per_uop_in_toposort_order():
    ast = _conv_kernel_ast()
    order = list(ast.toposort())

    model = uop_to_onnx_model(ast)
    onnx.checker.check_model(model)

    assert len(model.graph.node) == len(order)

    index = {u: i for i, u in enumerate(order)}
    by_output_name = {node.output[0]: node for node in model.graph.node}
    for u in order:
        node = by_output_name[f"u{index[u]}"]
        assert node.domain == DOMAIN
        assert node.op_type == u.op.name
        assert list(node.input) == [f"u{index[s]}" for s in u.src]
        attrs = {a.name: a for a in node.attribute}
        assert attrs["dtype"].s.decode() == str(u.dtype)


def test_graph_output_is_the_root_uop():
    ast = _conv_kernel_ast()
    order = list(ast.toposort())
    root_index = order.index(ast)

    model = uop_to_onnx_model(ast)

    assert len(model.graph.output) == 1
    assert model.graph.output[0].name == f"u{root_index}"


def test_declares_the_custom_domain_opset():
    model = uop_to_onnx_model(_conv_kernel_ast())
    domains = {opset.domain: opset.version for opset in model.opset_import}
    assert domains[DOMAIN] == 1


def test_int_and_string_args_are_encoded_natively_not_via_repr_fallback():
    """CONST nodes (a plain int/float arg) should use the native ``arg_i``/
    ``arg_f`` attribute, not fall back to ``arg_repr`` -- only genuinely
    irregular ``arg`` types (a dataclass, a tuple, ...) should need that
    escape hatch (see the module's own docstring on why one exists at all).
    """
    ast = _conv_kernel_ast()
    model = uop_to_onnx_model(ast)

    const_nodes = [n for n in model.graph.node if n.op_type == "CONST"]
    assert const_nodes, "expected at least one CONST uop in a real Conv2D kernel"
    for node in const_nodes:
        attr_names = {a.name for a in node.attribute}
        assert "arg_repr" not in attr_names
        assert ("arg_i" in attr_names) or ("arg_f" in attr_names)
