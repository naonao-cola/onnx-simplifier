"""Tests for ``onnxsim.webgpu_custom_kernel_runtime.split_around_node`` --
the graph-surgery half of running an ONNX model through ``onnxruntime-web``
with one node (carrying a custom WebGPU program) excised and dispatched
separately. See that module's own docstring for why the node must be
physically removed rather than just left in place and expected to fail
gracefully.

Models are built via the ONNX text format parser (see CLAUDE.md's testing
guidance); correctness is checked by recomposing ``pre`` -> the excised node
(run standalone, standing in for a real WebGPU dispatch, which needs a
browser -- see ``scripts/convertmodel/test/webgpu_custom_kernel_runtime.test.mjs``
for that half) -> ``post`` and comparing against
``onnx.reference.ReferenceEvaluator`` running the original, unsplit graph.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

from onnxsim.vitisai_target import split_model
from onnxsim.webgpu_custom_kernel_runtime import split_around_node


def _model(body, initializer=(), opset=17, ir_version=10):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["": {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _session_inputs(model):
    return [i.name for i in model.graph.input]


def _named(model, output_name, node_name):
    for node in model.graph.node:
        if output_name in node.output:
            node.name = node_name
            return model
    raise AssertionError(f"no node producing {output_name!r}")


def _run_standalone(node, feeds, output_shapes_known_from=None):
    """Runs a single detached ``NodeProto`` via ``ReferenceEvaluator``,
    standing in for a real WebGPU dispatch of that node's own generated
    kernel -- this test only checks the graph surgery, not codegen (see
    ``tests/test_webgpu_tinygrad_codegen.py`` for that).
    """
    node_model = onnx.ModelProto()
    node_model.CopyFrom(output_shapes_known_from)
    del node_model.graph.node[:]
    del node_model.graph.input[:]
    del node_model.graph.output[:]
    # Keep the original model's initializers (e.g. the node's own weight)
    # -- only its nodes/inputs/outputs need replacing.
    node_model.graph.node.append(node)
    for name in feeds:
        node_model.graph.input.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )
    for name in node.output:
        node_model.graph.output.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )
    return ReferenceEvaluator(node_model).run(None, feeds)


def _conv3d_model():
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(
        rng.standard_normal((2, 2, 3, 3, 3)).astype(np.float32), "w"
    )
    model = _model(
        """
        g (float[1,2,4,4,4] x) => (float[?] y)
        {
          pre = Relu(x)
          conv_out = Conv<kernel_shape = [3, 3, 3]>(pre, w)
          y = Relu(conv_out)
        }
        """,
        initializer=[w],
    )
    _named(model, "pre", "pre_relu")
    _named(model, "conv_out", "conv_node")
    _named(model, "y", "post_relu")
    onnx.checker.check_model(model)
    return model


def test_split_excises_exactly_the_named_node():
    model = _conv3d_model()
    result = split_around_node(model, "conv_node")

    assert result.pre is not None
    assert [n.name for n in result.pre.graph.node] == ["pre_relu"]
    assert [n.name for n in result.post.graph.node] == ["post_relu"]
    assert result.node.name == "conv_node"
    assert list(result.node.input) == ["pre", "w"]
    assert list(result.node.output) == ["conv_out"]


def test_split_pre_and_post_are_checker_valid():
    model = _conv3d_model()
    result = split_around_node(model, "conv_node")
    onnx.checker.check_model(result.pre)
    onnx.checker.check_model(result.post)


def test_split_recomposition_matches_reference():
    model = _conv3d_model()
    result = split_around_node(model, "conv_node")

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 4, 4, 4)).astype(np.float32)

    (y_ref,) = ReferenceEvaluator(model).run(None, {"x": x})

    pre_output_name = result.pre.graph.output[0].name
    (pre_value,) = ReferenceEvaluator(result.pre).run(None, {"x": x})

    (mid_value,) = _run_standalone(
        result.node, {pre_output_name: pre_value}, output_shapes_known_from=model
    )

    post_input_name = result.post.graph.input[0].name
    (y_composed,) = ReferenceEvaluator(result.post).run(
        None, {post_input_name: mid_value}
    )

    np.testing.assert_allclose(y_composed, y_ref, rtol=1e-5)


def test_split_when_node_consumes_only_graph_inputs_has_no_pre():
    rng = np.random.default_rng(2)
    w = numpy_helper.from_array(
        rng.standard_normal((2, 2, 3, 3, 3)).astype(np.float32), "w"
    )
    model = _model(
        """
        g (float[1,2,4,4,4] x) => (float[?] y)
        {
          conv_out = Conv<kernel_shape = [3, 3, 3]>(x, w)
          y = Relu(conv_out)
        }
        """,
        initializer=[w],
    )
    _named(model, "conv_out", "conv_node")
    _named(model, "y", "post_relu")
    onnx.checker.check_model(model)

    result = split_around_node(model, "conv_node")

    assert result.pre is None
    assert list(result.node.input) == ["x", "w"]
    onnx.checker.check_model(result.post)


def test_split_carries_side_input_from_original_graph_input():
    # post_scale here needs both conv_out (through the excised node) *and*
    # the original graph input `scale` directly, never touched by pre or
    # the excised node -- exercises split_model's own side-input
    # passthrough (see its docstring), not anything split_around_node adds
    # on top.
    rng = np.random.default_rng(3)
    w = numpy_helper.from_array(
        rng.standard_normal((2, 2, 3, 3, 3)).astype(np.float32), "w"
    )
    model = _model(
        """
        g (float[1,2,4,4,4] x, float scale) => (float[?] y)
        {
          pre = Relu(x)
          conv_out = Conv<kernel_shape = [3, 3, 3]>(pre, w)
          y = Mul(conv_out, scale)
        }
        """,
        initializer=[w],
    )
    _named(model, "pre", "pre_relu")
    _named(model, "conv_out", "conv_node")
    _named(model, "y", "post_scale")
    onnx.checker.check_model(model)

    result = split_around_node(model, "conv_node")
    onnx.checker.check_model(result.post)
    # The side input must have joined post's own inputs, not been dropped.
    assert "scale" in [i.name for i in result.post.graph.input]

    rng2 = np.random.default_rng(4)
    x = rng2.standard_normal((1, 2, 4, 4, 4)).astype(np.float32)
    scale = np.float32(2.5)
    feeds = {"x": x, "scale": scale}
    (y_ref,) = ReferenceEvaluator(model).run(None, feeds)

    (pre_value,) = ReferenceEvaluator(result.pre).run(None, {"x": x})
    (mid_value,) = _run_standalone(
        result.node,
        {result.pre.graph.output[0].name: pre_value},
        output_shapes_known_from=model,
    )
    post_feeds = {
        name: (mid_value if name != "scale" else scale)
        for name in _session_inputs(result.post)
    }
    (y_composed,) = ReferenceEvaluator(result.post).run(None, post_feeds)
    np.testing.assert_allclose(y_composed, y_ref, rtol=1e-5)


def test_split_unknown_node_raises():
    model = _conv3d_model()
    with pytest.raises(ValueError, match="no node named"):
        split_around_node(model, "does_not_exist")


def test_split_accepts_file_path(tmp_path):
    model = _conv3d_model()
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))

    result = split_around_node(str(path), "conv_node")
    assert result.node.name == "conv_node"


def test_split_does_not_mutate_input_model():
    model = _conv3d_model()
    before = model.SerializeToString()
    split_around_node(model, "conv_node")
    assert model.SerializeToString() == before


def test_split_model_still_rejects_unproduced_boundary():
    # split_around_node builds directly on split_model -- confirm the
    # dependency is the real thing, not a hand-duplicated stand-in.
    model = _model(
        """
        g (float[2,2] x) => (float[2,2] y)
        {
          y = Relu(x)
        }
        """
    )
    with pytest.raises(ValueError, match="not produced by the graph"):
        split_model(model, ["nope"])
