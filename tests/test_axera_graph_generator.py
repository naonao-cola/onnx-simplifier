import os
import sys

import onnx
import pytest

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import graph_generator  # noqa: E402


def _model(nodes, inputs, outputs, initializers=()):
    graph = onnx.helper.make_graph(nodes, "source", inputs, outputs, initializers)
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])


def _full_model():
    f = onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [8], list(range(8)))
    nodes = [
        onnx.helper.make_node("Gather", ["x", "indices"], ["g"], axis=3),
        onnx.helper.make_node("Reshape", ["g", "shape"], ["r"]),
        onnx.helper.make_node("MatMul", ["r", "w"], ["m"]),
        onnx.helper.make_node("Transpose", ["m"], ["t"], perm=[0, 1, 3, 2]),
        onnx.helper.make_node("Add", ["t", "b"], ["y"]),
    ]
    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [4], [1, 1, 8, 4])
    inputs = [
        onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 4, 16]),
        onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [4, 6]),
        onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 1, 6, 8]),
    ]
    return _model(nodes, inputs, [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 6, 8])], [f, shape])


def test_schedule_fuses_measured_chain():
    plan = graph_generator.schedule_graph(_full_model())
    assert plan.chain == "gather_reshape_matmul_transpose_add"
    assert plan.segments[0].inputs == ("x", "w", "b")


def test_generate_uses_one_fused_template(tmp_path):
    source = tmp_path / "source.onnx"
    output = tmp_path / "generated.axmodel"
    onnx.save(_full_model(), source)
    plan = graph_generator.generate(str(source), str(output), indices=[15, 0, 7, 7, 3, 12, 1, 14])
    assert plan.chain == "gather_reshape_matmul_transpose_add"
    generated = onnx.load(str(output), load_external_data=False)
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert [item.name for item in generated.graph.input] == ["x", "w", "b"]


def test_schedule_refuses_unmeasured_transpose():
    model = _full_model()
    model.graph.node[3].attribute[0].ints[:] = [0, 2, 1, 3]
    with pytest.raises(ValueError, match="Transpose perm"):
        graph_generator.schedule_graph(model)
