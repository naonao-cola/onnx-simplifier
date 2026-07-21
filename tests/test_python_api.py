import io
from typing import Any, Callable, Dict, Optional
import os
import tempfile

import numpy as np
import torch
import onnx
import onnxsim
import torchvision as tv
import pytest

from onnxsim.test_utils import export_simplify_and_check_by_python_api


def str_is_logical_positive(x: str) -> bool:
    return x.lower() in ["1", "on", "true"]


def skip_in_ci():
    return pytest.mark.skipif(
        str_is_logical_positive(os.getenv("CI", "")), reason="memory limited"
    )


def test_just_reshape():
    class JustReshape(torch.nn.Module):
        def __init__(self):
            super(JustReshape, self).__init__()

        def forward(self, x):
            return x.view((x.shape[0], x.shape[1], x.shape[3] * x.shape[2]))

    net = JustReshape()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net, dummy_input, export_kwargs={"do_constant_folding": False}
    )
    assert len(sim_model.graph.node) == 1


def test_a_model_not_need_simplification():
    class ModelNotNeedSimplification(torch.nn.Module):
        def __init__(self):
            super(ModelNotNeedSimplification, self).__init__()

        def forward(self, x):
            return x + 1

    net = ModelNotNeedSimplification()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(net, dummy_input)
    assert len(sim_model.graph.node) == 1


def test_exprimental_simplify_subgraph():
    class WithSubGraph(torch.nn.Module):
        def __init__(self):
            super(WithSubGraph, self).__init__()

        def forward(self, x):
            if x.sum() > 1.0:
                # NOTE: even onnxsim cannot simplify it,
                # a canonical pass in onnx-optimizer is needed for it.
                # so this test only tests that include_subgraph doesn't
                # result in invalid model in this case
                return 3 + x + 3
            else:
                return x + 4

    net = torch.jit.script(WithSubGraph())
    dummy_input = torch.randn(2)
    sim_model = export_simplify_and_check_by_python_api(
        net, dummy_input, simplify_kwargs={"include_subgraph": True}
    )
    assert len(sim_model.graph.node) == 3
    assert len(sim_model.graph.node[2].attribute[0].g.node) == 2
    assert len(sim_model.graph.node[2].attribute[1].g.node) == 1


def test_dynamic_batch_size():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            return x + 2

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "dynamic_axes": {"input": {0: "batch_size"}},
        },
        simplify_kwargs={"test_input_shapes": {"input": [2, 3, 4, 5]}},
    )
    assert len(sim_model.graph.node) == 1


def test_dynamic_axes_preserve_dynamic_dimension():
    # Regression test for GitHub issue #299. When a dimension of the input is
    # dynamic, the shape computation that reads that dimension at runtime must
    # NOT be constant-folded away, otherwise the simplified model bakes in the
    # dummy batch size and breaks for every other input size.
    #
    # onnxsim only folds a node when *all* of its inputs are constants
    # (initializers or the outputs of already-folded nodes). A graph input is
    # not a constant, so a "Shape" op reading a dynamic input is never folded.
    # This test locks that in behaviourally: the simplified model must still
    # run correctly at a batch size different from the one used at export time.
    class DynamicReshape(torch.nn.Module):
        def __init__(self):
            super(DynamicReshape, self).__init__()

        def forward(self, x):
            # Keep the dynamic batch dim, merge the two static trailing dims.
            return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

    net = DynamicReshape()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
        },
        simplify_kwargs={"test_input_shapes": {"input": [2, 3, 4, 5]}},
    )

    # The simplified model must still expose the batch dimension as dynamic
    # rather than hardcoding the dummy value of 2.
    in_dim0 = sim_model.graph.input[0].type.tensor_type.shape.dim[0]
    out_dim0 = sim_model.graph.output[0].type.tensor_type.shape.dim[0]
    assert in_dim0.dim_value == 0 and in_dim0.dim_param == "batch"
    assert out_dim0.dim_value == 0 and out_dim0.dim_param == "batch"

    # And it must actually run for a batch size other than the export dummy of
    # 2. If the shape computation had been folded to a constant, this would
    # raise or produce the wrong output shape.
    for batch_size in (1, 2, 7):
        x = np.random.rand(batch_size, 3, 4, 5).astype(np.float32)
        outputs = onnxsim.backend.run_model(sim_model, {"input": x})
        (result,) = outputs.values()
        assert result.shape == (batch_size, 3, 20)
        np.testing.assert_allclose(
            result, x.reshape(batch_size, 3, 20), rtol=1e-5, atol=1e-6
        )


# NOTE: `include_subgraph` makes this test fail
@skip_in_ci()
def test_torchvision_fasterrcnn_fpn():
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# maskrcnn is only supported in opset 11 and higher
@skip_in_ci()
def test_torchvision_maskrcnn_fpn_opset11():
    model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# keypointrcnn is only supported in opset 11 and higher
@skip_in_ci()
def test_torchvision_keypointrcnn_fpn():
    model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# shufflenet and mnasnet causes segfault in CI (perhaps because of memory limit)
# but works locally
@skip_in_ci()
def test_torchvision_shufflenet_v2():
    model = tv.models.shufflenet_v2_x1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@skip_in_ci()
def test_torchvision_mnasnet():
    model = tv.models.mnasnet1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@skip_in_ci()
def test_torchvision_deeplabv3():
    model = tv.models.segmentation.deeplabv3_resnet50(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


def test_unused_output():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            x1 = x + 2
            x1 = x1 - 2
            x1 = x1 * 2
            x1 = x1 / 2
            y1 = x1
            x2 = x + 2
            x2 = x2 - 2
            x2 = x2 * 2
            x2 = x2 / 2
            y2 = x2
            x3 = x + 2
            x3 = x3 - 2
            x3 = x3 * 2
            x3 = x3 / 2
            y3 = x3
            return y1, y2, y3

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "output_names": ["output0", "output1", "output2"],
        },
        simplify_kwargs={"unused_output": ["output1", "output2"]},
    )
    assert len(sim_model.graph.node) == 4


def test_remove_unused_initializer():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.w = torch.nn.Parameter(torch.ones(5, 4))

        def forward(self, x):
            return x + torch.transpose(self.w, 0, 1)

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        is_model_valid=lambda model: any(
            node.op_type == "Transpose" for node in model.graph.node
        ),
        export_kwargs={"do_constant_folding": False},
    )
    assert len(sim_model.graph.node) == 1
    assert len(sim_model.graph.initializer) == 1


@skip_in_ci()
def test_model_larger_than_2gb():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            # a parameter is 500MB
            self.w1 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w2 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w3 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w4 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w5 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))

        def forward(self, x):
            return x + (self.w1 + self.w2 + self.w3 + self.w4 + self.w5)

    net = SimpleModel()
    dummy_input = torch.randn(125 * 1024 * 1024)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        is_model_valid=lambda model: sum(
            node.op_type == "Add" for node in model.graph.node
        )
        == 5,
        export_kwargs={"do_constant_folding": False},
    )
    assert len(sim_model.graph.node) == 1
    assert sim_model.graph.node[0].op_type == "Add"


def test_unset_optional_input():
    fmap = []
    nodes = [] 
    initializers = []

    fmap.append(onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, shape=(1,3,4,4)))

    X = np.random.rand(1,3,2,2).astype(np.float32)
    initializers.append(onnx.helper.make_tensor('X', onnx.TensorProto.FLOAT, X.shape, X.copy().tobytes(), raw=True))
    sizes = np.asarray([1,3,4,4]).astype(np.int64)
    initializers.append(onnx.helper.make_tensor('sizes', onnx.TensorProto.INT64, sizes.shape, sizes.copy().tobytes(), raw=True))

    nodes.append(onnx.helper.make_node(
      'Resize',
      inputs=['X', '', '', 'sizes'],
      outputs=['y'],
      mode='linear'))

    graph_def = onnx.helper.make_graph(
      nodes,
      'test_unset_optional_input',
      [],
      [fmap[-1]],
      value_info=fmap,
      initializer=initializers
      )

    opset_imports = [onnx.helper.make_opsetid("", 14)]
    
    model = onnx.helper.make_model(graph_def, opset_imports=opset_imports, ir_version=10)
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    assert len(model.graph.node) == 1
    assert len(model.graph.initializer) == 2
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1


def test_fold_deterministic_op():
    # An op that the operator schema marks as deterministic and whose inputs are
    # all constants should be constant-folded away.
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    initializers = [
        onnx.helper.make_tensor('a', onnx.TensorProto.FLOAT, a.shape, a.tobytes(), raw=True),
        onnx.helper.make_tensor('b', onnx.TensorProto.FLOAT, b.shape, b.tobytes(), raw=True),
    ]
    node = onnx.helper.make_node('Add', inputs=['a', 'b'], outputs=['y'])
    out = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, (2, 3))
    graph_def = onnx.helper.make_graph(
        [node], 'test_fold_deterministic_op', [], [out], initializer=initializers)
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10)

    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    # The Add node is folded into a single constant initializer.
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1


def test_do_not_fold_random_op():
    # RandomUniform is non-deterministic according to the operator schema
    # determinism attribute, so it must not be constant-folded even though it
    # has no non-constant inputs.
    node = onnx.helper.make_node(
        'RandomUniform', inputs=[], outputs=['y'],
        shape=[2, 3], dtype=onnx.TensorProto.FLOAT)
    out = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, (2, 3))
    graph_def = onnx.helper.make_graph(
        [node], 'test_do_not_fold_random_op', [], [out])
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10)

    sim_model, _ = onnxsim.simplify(model, check_n=0)
    assert len(sim_model.graph.node) == 1
    assert sim_model.graph.node[0].op_type == 'RandomUniform'
    assert len(sim_model.graph.initializer) == 0


def test_do_not_fold_random_like_op():
    # RandomNormalLike is non-deterministic; it must not be folded even when its
    # input is a constant.
    x = np.zeros((2, 3), dtype=np.float32)
    initializers = [
        onnx.helper.make_tensor('x', onnx.TensorProto.FLOAT, x.shape, x.tobytes(), raw=True),
    ]
    node = onnx.helper.make_node('RandomNormalLike', inputs=['x'], outputs=['y'])
    out = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, (2, 3))
    graph_def = onnx.helper.make_graph(
        [node], 'test_do_not_fold_random_like_op', [], [out], initializer=initializers)
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10)

    sim_model, _ = onnxsim.simplify(model, check_n=0)
    assert any(n.op_type == 'RandomNormalLike' for n in sim_model.graph.node)


def test_overwrite_input_shape_ignores_non_positive():
    # A non-positive value in overwrite_input_shapes must not be written to the
    # graph as a literal (e.g. 0) dimension; the original dimension should be
    # kept instead so the simplified model stays runnable (GitHub issue #237).
    x = onnx.helper.make_tensor_value_info(
        'input', onnx.TensorProto.FLOAT, ['N', 3, 'H', 'W'])
    y = onnx.helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, ['N', 3, 'H', 'W'])
    node = onnx.helper.make_node('Relu', ['input'], ['output'])
    graph_def = onnx.helper.make_graph(
        [node], 'test_overwrite_input_shape_ignores_non_positive', [x], [y])
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)])

    sim_model, _ = onnxsim.simplify(
        model, overwrite_input_shapes={'input': [1, 3, 0, 0]})
    dims = sim_model.graph.input[0].type.tensor_type.shape.dim
    # The positive value is applied, the non-positive ones are left untouched
    # (the original dynamic dim params are kept, never set to 0).
    assert dims[0].dim_value == 1
    assert dims[2].dim_param == 'H'
    assert dims[3].dim_param == 'W'


def test_preserve_doc_strings():
    # onnxsim must not drop the doc_string fields of the model / graph / inputs
    # / outputs while simplifying (GitHub issue #428).
    x = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 4])
    x.doc_string = "input documentation"
    y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 4])
    y.doc_string = "output documentation"
    node = onnx.helper.make_node('Relu', ['X'], ['Y'])
    graph_def = onnx.helper.make_graph(
        [node], 'test_preserve_doc_strings', [x], [y])
    graph_def.doc_string = "graph documentation"
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.doc_string = "model documentation"

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert sim_model.doc_string == "model documentation"
    assert sim_model.graph.doc_string == "graph documentation"
    assert sim_model.graph.input[0].doc_string == "input documentation"
    assert sim_model.graph.output[0].doc_string == "output documentation"


def _make_scalar_initializer(name: str, value, dtype) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _quant_params():
    return [
        _make_scalar_initializer("s", 0.01, np.float32),
        _make_scalar_initializer("zp", 128, np.uint8),
    ]


def _build_contrib_model(nodes, inputs, outputs, initializer):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer=initializer)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 13),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 9
    return model


def _value_info_shape(model: onnx.ModelProto, name: str):
    for vi in model.graph.value_info:
        if vi.name == name:
            tensor_type = vi.type.tensor_type
            if not tensor_type.HasField("shape"):
                return None
            return [d.dim_value for d in tensor_type.shape.dim]
    return None


def test_qlinear_add_shape_inference():
    # QLinearAdd is an ONNX Runtime "com.microsoft" contrib op. Without a schema
    # registered for it, ONNX shape inference stops and the intermediate tensor
    # never gets a shape (GitHub issue #245).
    nodes = [
        onnx.helper.make_node(
            "QLinearAdd",
            ["A", "s", "zp", "B", "s", "zp", "s", "zp"],
            ["C"],
            domain="com.microsoft",
        ),
        onnx.helper.make_node("DequantizeLinear", ["C", "s", "zp"], ["out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
        onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "out", onnx.TensorProto.FLOAT, [1, 3, 16, 16]
        )
    ]
    model = _build_contrib_model(nodes, inputs, outputs, _quant_params())
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 3, 16, 16]


def test_qlinear_concat_shape_inference():
    nodes = [
        onnx.helper.make_node(
            "QLinearConcat",
            ["s", "zp", "A", "s", "zp", "B", "s", "zp"],
            ["C"],
            domain="com.microsoft",
            axis=1,
        ),
        onnx.helper.make_node("DequantizeLinear", ["C", "s", "zp"], ["out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
        onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 5, 16, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "out", onnx.TensorProto.FLOAT, [1, 8, 16, 16]
        )
    ]
    model = _build_contrib_model(nodes, inputs, outputs, _quant_params())
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 8, 16, 16]


def test_unknown_contrib_op_is_tolerated():
    # Registering schemas for the supported quantized ops must not make the
    # checker reject other, unregistered "com.microsoft" contrib operators.
    nodes = [
        onnx.helper.make_node(
            "QLinearAdd",
            ["A", "s", "zp", "B", "s", "zp", "s", "zp"],
            ["C"],
            domain="com.microsoft",
        ),
        onnx.helper.make_node(
            "SomeUnknownContribOp", ["C"], ["D"], domain="com.microsoft"
        ),
        onnx.helper.make_node("DequantizeLinear", ["C", "s", "zp"], ["out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
        onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "out", onnx.TensorProto.FLOAT, [1, 3, 16, 16]
        ),
        onnx.helper.make_tensor_value_info("D", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
    ]
    model = _build_contrib_model(nodes, inputs, outputs, _quant_params())
    sim_model, check_ok = onnxsim.simplify(model, skip_constant_folding=True)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 3, 16, 16]


def test_perform_optimization_false():
    def _create_dummy_model():
        class MockModel(torch.nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = MockModel()
        dummy_input = torch.randn(1, 10)
        onnx_file = "dummy_model.onnx"
        torch.onnx.export(model, dummy_input, onnx_file, dynamo=False)
        return onnx_file

    onnx_model_path = _create_dummy_model()
    onnx_model = onnx.load(onnx_model_path)
    simple_model, _ = onnxsim.simplify(onnx_model, perform_optimization=False, skip_shape_inference=True)
    assert simple_model is not None

