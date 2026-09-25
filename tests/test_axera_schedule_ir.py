import os
import sys

import onnx

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import schedule_ir  # noqa: E402


def _full_model():
    indices = onnx.helper.make_tensor(
        "indices", onnx.TensorProto.INT64, [8], list(range(8))
    )
    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [4], [1, 1, 8, 4])
    nodes = [
        onnx.helper.make_node("Gather", ["x", "indices"], ["g"], axis=3),
        onnx.helper.make_node("Reshape", ["g", "shape"], ["r"]),
        onnx.helper.make_node("MatMul", ["r", "w"], ["m"]),
        onnx.helper.make_node("Transpose", ["m"], ["t"], perm=[0, 1, 3, 2]),
        onnx.helper.make_node("Add", ["t", "b"], ["y"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 4, 16]),
        onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [4, 6]),
        onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 1, 6, 8]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "source",
        inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 6, 8])],
        [indices, shape],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )


def test_schedule_ir_records_measured_fused_kernel():
    model = _full_model()
    ir = schedule_ir.build(model)
    assert [item.name for item in ir.inputs] == ["x", "w", "b"]
    assert [item.name for item in ir.outputs] == ["y"]
    assert ir.kernels == (
        schedule_ir.KernelSpec(
            "kernel_0",
            "gather_reshape_matmul_transpose_add",
            ("x", "w", "b"),
            "y",
            "gather_reshape_matmul_transpose_add",
        ),
    )
    assert ir.dependencies == ()
    assert ir.inputs[0].nbytes == 1 * 1 * 4 * 16 * 4
    assert ir.outputs[0].nbytes == 1 * 1 * 6 * 8 * 4
    assert ir.memory_size == 256 + 128 + 192 + 192
    assert [allocation.name for allocation in ir.allocations] == ["b", "w", "x", "y"]


def test_schedule_ir_writes_deterministic_json(tmp_path):
    model = _full_model()
    source = tmp_path / "source.onnx"
    output = tmp_path / "schedule.json"
    onnx.save(model, str(source))
    schedule_ir.write(str(source), str(output))
    assert '"chain": "gather_reshape_matmul_transpose_add"' in output.read_text()


def test_fused_internal_values_do_not_consume_schedule_arena():
    model = onnx.shape_inference.infer_shapes(_full_model())
    ir = schedule_ir.build(model)
    assert {item.name for item in ir.allocations} == {"x", "w", "b", "y"}
    assert ir.memory_size == 256 + 128 + 192 + 192
