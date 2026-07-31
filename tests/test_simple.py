import os
import tempfile
from typing import Optional

import onnx
import onnxruntime
import torch
from onnx import TensorProto, helper, numpy_helper

import onnxsim
from onnxsim.test_utils import export_simplify_and_check_by_python_api


def test_onnx_simplifier():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    export_simplify_and_check_by_python_api(MockModel(), torch.randn(1, 10))


def test_mg():
    class MG(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, b):
            x = x.float()
            b = b.float()
            sh = x.shape
            x = x.view(sh[0], sh[1], -1)
            b = b.squeeze(-1)
            b = b.squeeze(-1)
            a = torch.matmul(b, x)
            preds = a.view(1, 100, sh[2], sh[3])
            return preds

    x = torch.randn([1, 256, 160, 184])
    b = torch.randn([100, 256, 1, 1])
    opt = export_simplify_and_check_by_python_api(
        MG(), (x, b), export_kwargs={"dynamo": True}
    )
    sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(
        out_names,
        {opt.graph.input[0].name: x.numpy(), opt.graph.input[1].name: b.numpy()},
    )
    assert outs[0].shape == MG()(x, b).shape


def test_transformer():
    model = torch.nn.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
    )
    model.to("cpu").to(torch.float32)
    model.eval()

    inputs = (
        torch.rand((100, 2, 256), dtype=torch.float32),
        torch.rand((15, 2, 256), dtype=torch.float32),
    )
    export_simplify_and_check_by_python_api(
        model, inputs, export_kwargs={"dynamo": True}
    )


def test_upsample():
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1_1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(16)
            self.maxpool = torch.nn.MaxPool2d(2, stride=2)
            self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.conv3_1 = torch.nn.Conv2d(32, 16, 3, padding=1)
            self.conv3_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.conv3_3 = torch.nn.Conv2d(16, 3, 3, padding=1)
            self.bn3 = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x)))))
            x2 = self.maxpool(x1)
            xup = F.interpolate(
                x2, scale_factor=2, mode="bilinear", align_corners=False
            )
            x3 = self.bn2(self.conv2_2(F.relu(self.conv2_1(xup))))
            x4 = F.relu(self.conv3_3(self.conv3_2(F.relu(self.conv3_1(x3)))))
            x5 = self.bn3(x4)
            return F.softsign(x5)

    inp = torch.rand(1, 3, 96, 96)
    net = Net()
    opt = export_simplify_and_check_by_python_api(
        net, (inp,), export_kwargs={"opset_version": 9}
    )

    u_out = None
    for n in opt.graph.node:
        if n.op_type == "Upsample":
            u_out = n.output[0]
    assert u_out is not None
    u_info = None
    for v in opt.graph.value_info:
        if v.name == u_out:
            u_info = v
    assert u_info is not None
    assert [i.dim_value for i in v.type.tensor_type.shape.dim] == [1, 3, 96, 96]


def test_concat_squeese():
    # test for https://github.com/onnxsim/onnxsim/issues/46
    class Model(torch.nn.Module):
        def forward(self, x):
            # return torch.cat((torch.mean(x, 1, keepdim=True), torch.mean(x, 1, keepdim=True)), dim=1)
            return torch.cat(
                (torch.mean(x, 1).unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
            )

    export_simplify_and_check_by_python_api(
        Model(), (torch.rand(20, 20),), export_kwargs={"opset_version": 9}
    )


def test_trilinear():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, input_tensor):
            return torch.nn.functional.interpolate(
                input_tensor, scale_factor=[4, 4, 4], mode="trilinear"
            )

    x = torch.rand(1, 8, 20, 120, 120)
    opt = export_simplify_and_check_by_python_api(
        Model(),
        (x,),
        export_kwargs={
            "opset_version": 11,
            "export_params": True,
        },
    )
    sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(out_names, {opt.graph.input[0].name: x.numpy()})
    assert outs[0].shape == (1, 8, 80, 480, 480)


def test_optional_type():
    class MyModule(torch.nn.Module):
        def forward(self, x: Optional[torch.Tensor]):
            return x is not None

    module = MyModule()
    inputs = (torch.tensor(1),)
    module = torch.jit.script(module, example_inputs=[inputs, (None,)])
    export_simplify_and_check_by_python_api(
        module,
        inputs,
        simplify_kwargs={"input_data": {"i": inputs[0].numpy()}},
        export_kwargs={"opset_version": 15, "input_names": ["i"]},
    )


def test_ext():
    class MyMod(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.param = torch.nn.Parameter(
                torch.rand(1024, 1024, 1024, dtype=torch.float16)
            )

        def forward(self, x):
            return self.param * x

    module = MyMod()
    inputs = (torch.tensor(1),)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_fn = os.path.join(tmpdirname, "tmp.onnx")
        torch.onnx.export(module, inputs, model_fn, dynamo=False, input_names=["x"])
        sim_model, check_ok = onnxsim.simplify(model_fn, check_n=0)
        module = None
        assert check_ok


def _constant_value(model, name):
    # Return the constant value produced for `name`, whether it is delivered as
    # a graph initializer or as a Constant node's value attribute, else None.
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer)
    for node in model.graph.node:
        if node.op_type == "Constant" and name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None


def test_partial_shape_evaluation_gather():
    # Partial shape evaluation for https://github.com/onnxsim/onnxsim/issues/139
    # The input's leading dimension is dynamic, but a Gather that reads only the
    # static dimensions of its shape must still be pre-computed into a constant.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 4, 5])
    g = helper.make_tensor_value_info("g", TensorProto.INT64, [3])
    indices = helper.make_tensor("indices", TensorProto.INT64, [3], [1, 2, 3])
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("Gather", ["s", "indices"], ["g"], axis=0),
    ]
    graph = helper.make_graph(nodes, "g", [x], [g], initializer=[indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The whole Shape -> Gather chain collapses to a constant even though the
    # batch dimension is dynamic.
    assert "Shape" not in op_types
    assert "Gather" not in op_types
    value = _constant_value(sim_model, "g")
    assert value is not None
    assert list(value) == [3, 4, 5]


def test_partial_shape_evaluation_keeps_dynamic_gather():
    # A Gather that reads the dynamic dimension must NOT be folded: its value is
    # unknown until runtime.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 4, 5])
    g = helper.make_tensor_value_info("g", TensorProto.INT64, [1])
    indices = helper.make_tensor("indices", TensorProto.INT64, [1], [0])
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("Gather", ["s", "indices"], ["g"], axis=0),
    ]
    graph = helper.make_graph(nodes, "g", [x], [g], initializer=[indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The dynamic dimension cannot be pre-computed, so the ops stay and "g" is
    # not turned into a constant.
    assert "Gather" in op_types
    assert _constant_value(sim_model, "g") is None


def test_partial_shape_evaluation_reshape_single_dynamic_dim():
    # Data propagation for Reshape: the target shape is computed at runtime from
    # the input's shape (Shape -> Gather -> Concat), and it has a single dynamic
    # entry (the batch) plus otherwise-static dimensions. Partial shape
    # evaluation propagates that shape to [batch, 60], and the Reshape pass
    # materializes it as the constant [-1, 60] -- the single -1 lets ONNX infer
    # the dynamic dim from the total element count, which is provably equivalent
    # -- so the whole Shape -> Gather -> Concat scaffolding collapses away.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 60])
    idx0 = helper.make_tensor("idx0", TensorProto.INT64, [1], [0])
    sixty = helper.make_tensor("sixty", TensorProto.INT64, [1], [60])  # 3*4*5
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("Gather", ["s", "idx0"], ["b"], axis=0),
        helper.make_node("Concat", ["b", "sixty"], ["newshape"], axis=0),
        helper.make_node("Reshape", ["x", "newshape"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[idx0, sixty])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The runtime shape computation is gone; only the Reshape (fed by a constant
    # shape) survives.
    assert "Shape" not in op_types
    assert "Gather" not in op_types
    assert "Concat" not in op_types
    reshape = [n for n in sim_model.graph.node if n.op_type == "Reshape"]
    assert len(reshape) == 1
    shape_value = _constant_value(sim_model, reshape[0].input[1])
    assert shape_value is not None
    assert list(shape_value) == [-1, 60]


def test_data_propagation_through_reshape():
    # ONNX has no data-propagation function for Reshape, so onnxsim registers one
    # (contrib_schemas): a shape tensor threaded through an element-preserving
    # Reshape must keep its propagated value so downstream shape arithmetic still
    # folds. Here Shape(x) -> Reshape(., [-1]) -> Gather([1, 2]) reads only the
    # static dims, so it must pre-compute to [3, 4] even though the batch is
    # dynamic -- which requires the value to survive the Reshape.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 4])
    g = helper.make_tensor_value_info("g", TensorProto.INT64, [2])
    flat = helper.make_tensor("flat", TensorProto.INT64, [1], [-1])
    indices = helper.make_tensor("indices", TensorProto.INT64, [2], [1, 2])
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("Reshape", ["s", "flat"], ["s2"]),
        helper.make_node("Gather", ["s2", "indices"], ["g"], axis=0),
    ]
    graph = helper.make_graph(nodes, "g", [x], [g], initializer=[flat, indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The value propagated through the Reshape, so the whole chain pre-computes.
    assert "Gather" not in op_types
    assert "Reshape" not in op_types
    value = _constant_value(sim_model, "g")
    assert value is not None
    assert list(value) == [3, 4]


def test_partial_shape_evaluation_symbolic_arithmetic_reshape():
    # Native symbolic shape evaluation (issue #532). The Reshape target is
    # computed at runtime with *arithmetic over the dynamic batch dim*:
    #   half = ReduceProd(Shape(x)) / 2 = (batch * 768) / 2 = batch * 384
    #   newshape = Concat(half, [2]) = [batch * 384, 2]
    # ONNX data propagation cannot carry a symbol through ReduceProd/Div, so the
    # #526 rewrite alone leaves the whole Shape -> ReduceProd -> Div -> Concat
    # scaffolding standing. The SymExpr evaluator keeps the dim as `384*batch`,
    # sees a single symbolic entry, and materializes the shape as the constant
    # [-1, 2] -- provably equivalent since batch*384*2 == numel(x) for every
    # batch -- so the scaffolding collapses.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 768])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])
    two = helper.make_tensor("two", TensorProto.INT64, [1], [2])
    two2 = helper.make_tensor("two2", TensorProto.INT64, [1], [2])
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("ReduceProd", ["s"], ["total"], keepdims=1),
        helper.make_node("Div", ["total", "two"], ["half"]),
        helper.make_node("Concat", ["half", "two2"], ["newshape"], axis=0),
        helper.make_node("Reshape", ["x", "newshape"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[two, two2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The runtime shape arithmetic is gone; only the Reshape (fed by a constant
    # shape) survives.
    assert "Shape" not in op_types
    assert "ReduceProd" not in op_types
    assert "Div" not in op_types
    assert "Concat" not in op_types
    reshape = [n for n in sim_model.graph.node if n.op_type == "Reshape"]
    assert len(reshape) == 1
    shape_value = _constant_value(sim_model, reshape[0].input[1])
    assert shape_value is not None
    assert list(shape_value) == [-1, 2]


def test_unfoldable_const_node_keeps_topological_order():
    # A const node (all-constant inputs) that fails to fold must keep its
    # original position. Here SequenceEmpty is treated as a const node; its
    # output feeds a non-const consumer (SequenceInsert). If a failed const node
    # were moved to the end of the graph it would land after its consumer and
    # break topological sorting, making the output fail onnx's checker (issues
    # #238, #335, #352).
    #
    # Constant folding is disabled here on purpose: SequenceEmpty produces a
    # sequence value, which the backend returns as an empty list. Folding would
    # coerce that into an empty *tensor* initializer and drop the node, which is
    # semantically wrong for a sequence. Skipping folding keeps the sequence
    # pipeline intact so we exercise the topological ordering of the preserved
    # nodes.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    nodes = [
        helper.make_node("SequenceEmpty", [], ["seq"]),
        helper.make_node("SequenceInsert", ["seq", "x"], ["seq2"]),
        helper.make_node("ConcatFromSequence", ["seq2"], ["y"], axis=0, new_axis=0),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, skip_constant_folding=True)
    assert check_ok
    # Output must remain a valid, topologically sorted graph.
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.index("SequenceEmpty") < op_types.index("SequenceInsert")


def test_folding_does_not_duplicate_initializers():
    # Folding a const op that reads a weight (here a Transpose on an initializer)
    # produces a new initializer for the result but must not leave the original
    # operand initializer dangling in the graph. Otherwise the weight data is
    # duplicated, which can push a large model past onnx's 2GB protobuf limit
    # before the optimizer runs (issue #174).
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3], list(range(12)))
    nodes = [
        helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
        helper.make_node("Add", ["x", "wt"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    # Disable the onnx optimizer so the constant folding logic alone is
    # responsible for cleaning up the dangling initializer.
    sim_model, check_ok = onnxsim.simplify(model, perform_optimization=False)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # Core invariant of the fix, independent of platform: the simplified graph
    # must never carry an initializer that no node consumes. Whether the backend
    # executor is able to fold the Transpose can vary between environments, but
    # the "no dangling initializer" property must hold either way.
    op_types = [n.op_type for n in sim_model.graph.node]
    used = {i for n in sim_model.graph.node for i in n.input}
    for init in sim_model.graph.initializer:
        assert init.name in used, f"unused initializer left behind: {init.name}"

    # When the Transpose was actually folded into an initializer, the folded
    # result must replace the original weight "w" rather than duplicate it.
    if "Transpose" not in op_types:
        assert "w" not in {init.name for init in sim_model.graph.initializer}


def test_batched_constant_folding():
    # Many independent constant sub-expressions are folded in a single backend
    # Session (batched folding) instead of one Session per node. This builds a
    # wide fan of Constant -> Mul chains that are summed into one constant and
    # then added to the runtime input, and checks that every constant node is
    # folded into a single initializer and that the result stays numerically
    # correct.
    import numpy as np

    n = 40
    base = np.arange(8, dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [8])
    base_init = numpy_helper.from_array(base, name="base")

    nodes = []
    prod_names = []
    for i in range(n):
        c_name = f"c{i}"
        p_name = f"p{i}"
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [c_name],
                value=numpy_helper.from_array(
                    np.array(float(i), dtype=np.float32), name=c_name
                ),
            )
        )
        nodes.append(helper.make_node("Mul", ["base", c_name], [p_name]))
        prod_names.append(p_name)
    nodes.append(helper.make_node("Sum", prod_names, ["acc"]))
    nodes.append(helper.make_node("Add", ["x", "acc"], ["y"]))

    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[base_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, perform_optimization=False)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # No initializer may be left dangling regardless of backend availability.
    used = {i for node in sim_model.graph.node for i in node.input}
    for init in sim_model.graph.initializer:
        assert init.name in used, f"unused initializer left behind: {init.name}"

    op_types = [node.op_type for node in sim_model.graph.node]
    # When the backend folded the constants, the whole fan collapses to a single
    # Add reading the runtime input and one folded initializer.
    if "Mul" not in op_types and "Sum" not in op_types and "Constant" not in op_types:
        assert op_types == ["Add"]
        acc = base * (n * (n - 1) / 2)
        x_val = np.random.rand(8).astype(np.float32)
        sess = onnxruntime.InferenceSession(
            sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (out,) = sess.run(None, {"x": x_val})
        np.testing.assert_allclose(out, x_val + acc, rtol=1e-5, atol=1e-5)


def test_fp8_qdq_model():
    # Regression test for GitHub issue #348. NVIDIA ModelOpt emits fp8 QDQ
    # models whose QuantizeLinear/DequantizeLinear zero points use the
    # ``float8_e4m3fn`` element type (17). onnxoptimizer's tensor-value hashing
    # passes (eliminate_common_subexpression / eliminate_duplicate_initializer)
    # cannot hash such tensors and used to abort the whole simplification with
    # "RuntimeError: no supported data type: 17". onnxsim now detects these
    # tensors and transparently skips the offending passes instead of crashing.
    def fp8_zero_point(name: str) -> onnx.TensorProto:
        # A single float8_e4m3fn zero, expressed as its raw byte so the test
        # does not depend on ml_dtypes.
        return helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, [], b"\x00", raw=True)

    weight = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 3], [0.1 * i for i in range(12)]
    )
    w_scale = helper.make_tensor("w_scale", TensorProto.FLOAT, [], [0.05])
    a_scale = helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.1])

    nodes = [
        # activation QDQ (dynamic input, must be preserved)
        helper.make_node("QuantizeLinear", ["X", "a_scale", "a_zp"], ["X_q"]),
        helper.make_node("DequantizeLinear", ["X_q", "a_scale", "a_zp"], ["X_dq"]),
        # weight QDQ (constant inputs) plus a duplicated weight zero point to
        # exercise eliminate_duplicate_initializer as well.
        helper.make_node("QuantizeLinear", ["W", "w_scale", "w_zp"], ["W_q"]),
        helper.make_node("DequantizeLinear", ["W_q", "w_scale", "w_zp2"], ["W_dq"]),
        helper.make_node("MatMul", ["X_dq", "W_dq"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fp8_qdq",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [
            weight,
            w_scale,
            a_scale,
            fp8_zero_point("a_zp"),
            fp8_zero_point("w_zp"),
            fp8_zero_point("w_zp2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    # Must not raise "no supported data type: 17".
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The fp8 QDQ structure must survive simplification.
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.count("QuantizeLinear") == 2
    assert op_types.count("DequantizeLinear") == 2
    assert "MatMul" in op_types


def test_fp8_qdq_modelopt_integration():
    # NVIDIA ModelOpt used to simplify ONNX models inside its quantization
    # preprocessing with exactly ``model_simp, check = onnxsim.simplify(model)``
    # (wrapped in a try/except that fell back to the unsimplified model on
    # error). It dropped onnxsim for onnxslim, in part because onnxsim aborted
    # on ModelOpt's fp8 QDQ output with "no supported data type: 17" (issue
    # #348). This exercises that exact call shape on a richer, more
    # ModelOpt-like graph -- Conv + Gemm with both activation and weight QDQ and
    # duplicated float8 zero points -- to guard that onnxsim simplifies it
    # without crashing and preserves the QDQ structure TensorRT relies on.
    def fp8_zero_point(name: str) -> onnx.TensorProto:
        # A single float8_e4m3fn zero as its raw byte (no ml_dtypes dependency).
        return helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, [], b"\x00", raw=True)

    conv_w = helper.make_tensor(
        "conv_w",
        TensorProto.FLOAT,
        [8, 3, 3, 3],
        [0.01 * (i % 7 - 3) for i in range(8 * 3 * 3 * 3)],
    )
    gemm_w = helper.make_tensor(
        "gemm_w", TensorProto.FLOAT, [8, 8], [0.02 * (i % 5 - 2) for i in range(64)]
    )
    conv_ws = helper.make_tensor("conv_w_scale", TensorProto.FLOAT, [], [0.02])
    gemm_ws = helper.make_tensor("gemm_w_scale", TensorProto.FLOAT, [], [0.03])
    act_s = helper.make_tensor("act_scale", TensorProto.FLOAT, [], [0.1])
    act_s2 = helper.make_tensor("act_scale2", TensorProto.FLOAT, [], [0.1])

    nodes = [
        # activation QDQ on the dynamic input -- must survive simplification.
        helper.make_node("QuantizeLinear", ["X", "act_scale", "a_zp"], ["Xq"]),
        helper.make_node("DequantizeLinear", ["Xq", "act_scale", "a_zp"], ["Xdq"]),
        # weight QDQ (constant). ModelOpt duplicates fp8 zero points across
        # weights, which is what tripped eliminate_duplicate_initializer.
        helper.make_node(
            "QuantizeLinear", ["conv_w", "conv_w_scale", "cw_zp"], ["cwq"]
        ),
        helper.make_node(
            "DequantizeLinear", ["cwq", "conv_w_scale", "cw_zp2"], ["cwdq"]
        ),
        helper.make_node("Conv", ["Xdq", "cwdq"], ["conv_out"], kernel_shape=[3, 3]),
        helper.make_node("GlobalAveragePool", ["conv_out"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        # second activation QDQ + weight QDQ feeding a Gemm.
        helper.make_node("QuantizeLinear", ["flat", "act_scale2", "a_zp2"], ["flatq"]),
        helper.make_node(
            "DequantizeLinear", ["flatq", "act_scale2", "a_zp2"], ["flatdq"]
        ),
        helper.make_node(
            "QuantizeLinear", ["gemm_w", "gemm_w_scale", "gw_zp"], ["gwq"]
        ),
        helper.make_node(
            "DequantizeLinear", ["gwq", "gemm_w_scale", "gw_zp2"], ["gwdq"]
        ),
        helper.make_node("Gemm", ["flatdq", "gwdq"], ["Y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "modelopt_fp8_qdq",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 6, 6])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        [
            conv_w,
            gemm_w,
            conv_ws,
            gemm_ws,
            act_s,
            act_s2,
            fp8_zero_point("a_zp"),
            fp8_zero_point("a_zp2"),
            fp8_zero_point("cw_zp"),
            fp8_zero_point("cw_zp2"),
            fp8_zero_point("gw_zp"),
            fp8_zero_point("gw_zp2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    # ModelOpt's exact former call shape. Must not raise "no supported data
    # type: 17"; check must be True so ModelOpt would keep the simplified model.
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # Every QDQ pair must be preserved -- TensorRT needs the QDQ structure, and
    # onnxsim must not fold or dedupe it away.
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.count("QuantizeLinear") == 4
    assert op_types.count("DequantizeLinear") == 4
    assert "Conv" in op_types
    assert "Gemm" in op_types


def test_if_with_const_cond_is_folded():
    # An `If` whose condition is a constant, with branches that each just
    # produce a constant, used to crash simplify() with a segfault (exit 139):
    # onnxsim's constant folding turns the branch `Constant` nodes into subgraph
    # initializers, and the onnxoptimizer "eliminate_if_with_const_cond" pass
    # then dereferenced a null value while inlining the taken branch (the branch
    # output was now an initializer with no producing node). With the fixed pass
    # the `If` is folded away and its output becomes the taken branch's constant
    # (GitHub issue #452).
    tv = helper.make_tensor("tv", TensorProto.FLOAT, [2], [1.0, 2.0])
    fv = helper.make_tensor("fv", TensorProto.FLOAT, [2], [3.0, 4.0])
    then_b = helper.make_graph(
        [helper.make_node("Constant", [], ["to"], value=tv)],
        "tb",
        [],
        [helper.make_tensor_value_info("to", TensorProto.FLOAT, [2])],
    )
    else_b = helper.make_graph(
        [helper.make_node("Constant", [], ["fo"], value=fv)],
        "fb",
        [],
        [helper.make_tensor_value_info("fo", TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node(
        "If", ["c"], ["Y"], then_branch=then_b, else_branch=else_b
    )
    graph = helper.make_graph(
        [if_node],
        "g",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        [helper.make_tensor("c", TensorProto.BOOL, [], [True])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The constant-condition `If` (condition is True) must be folded away, and
    # the model output must be the "then" branch constant [1.0, 2.0]. Read the
    # folded value straight out of the graph rather than running the model,
    # since helper.make_model stamps the latest ONNX IR version, which the
    # bundled onnxruntime may not load.
    assert all(n.op_type != "If" for n in sim_model.graph.node)

    def _resolve_const(model, name):
        # Follow the output through initializers, Constant nodes, and Identity
        # aliases until a constant tensor is found.
        seen = set()
        while name and name not in seen:
            seen.add(name)
            for init in model.graph.initializer:
                if init.name == name:
                    return numpy_helper.to_array(init)
            alias = None
            for node in model.graph.node:
                if name not in node.output:
                    continue
                if node.op_type == "Constant":
                    (attr,) = [a for a in node.attribute if a.name == "value"]
                    return numpy_helper.to_array(attr.t)
                if node.op_type == "Identity":
                    alias = node.input[0]
                break
            name = alias
        raise AssertionError(f"output {name!r} is not a resolvable constant")

    out = _resolve_const(sim_model, sim_model.graph.output[0].name)
    assert out.tolist() == [1.0, 2.0]


def test_ir3_conv_bn_fuses():
    # IR version 3 models (e.g. the opset-8 ``resnet101-v1-7``) list every
    # initializer as a graph input too, which is required before IR 4. onnxsim
    # used to skip ``remove_initializer_from_input`` for such models, so the
    # weights stayed "runtime inputs" and onnxoptimizer refused to fold them --
    # ``fuse_bn_into_conv`` never fired and the graph came out unchanged
    # (GitHub issue #543). onnxsim now bumps these to IR 4 and drops the
    # initializer inputs, so the Conv+BN pair fuses away.
    import numpy as np

    out_ch, in_ch = 4, 3
    W = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32)
    scale = np.abs(np.random.randn(out_ch).astype(np.float32)) + 1
    bias = np.random.randn(out_ch).astype(np.float32)
    mean = np.random.randn(out_ch).astype(np.float32)
    var = np.abs(np.random.randn(out_ch).astype(np.float32)) + 1
    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(var, "var"),
    ]
    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"])
    bn = helper.make_node(
        "BatchNormalization", ["conv_out", "scale", "bias", "mean", "var"], ["Y"]
    )
    graph = helper.make_graph(
        [conv, bn],
        "g",
        # Initializers are *also* listed as graph inputs, as IR<4 requires.
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_ch, 8, 8])]
        + [
            helper.make_tensor_value_info(t.name, TensorProto.FLOAT, t.dims)
            for t in inits
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_ch, 6, 6])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 8)])
    model.ir_version = 3
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The BatchNormalization is folded into the Conv weights and disappears.
    assert all(n.op_type != "BatchNormalization" for n in sim_model.graph.node)
    assert any(n.op_type == "Conv" for n in sim_model.graph.node)


def _ir3_matmul_model(opset: int) -> onnx.ModelProto:
    # A minimal IR-3 model whose weight ``W`` is *also* listed as a graph input,
    # as IR<4 required. ``opset`` selects the ai.onnx version under test.
    import numpy as np

    W = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        "g",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        [W],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 3
    return model


def test_remove_initializer_from_input_skips_ancient_opset():
    # The opset-3 onnx-caffe2 ``resnet50-caffe2-v1-3`` regressed once IR<4
    # folding was enabled: bumping such a model to IR 4 lets onnxoptimizer's
    # value-baking fusions fire, and ``fuse_bn_into_conv`` emits ``Cast`` nodes
    # using the modern INT ``to`` attribute -- illegal under opset 3, where
    # ``to`` is a STRING -- so onnx/onnxruntime abort with "Mismatched attribute
    # type in 'Cast_0 : to'. Expected: 'STRING', actual: 'INT'". Opsets below
    # ``_MIN_OPSET_FOR_INITIALIZER_FOLD`` must therefore be left untouched: the
    # initializer stays a graph input (so the fusions stay disabled) and the IR
    # version is not bumped.
    from onnxsim.onnx_simplifier import remove_initializer_from_input

    model = _ir3_matmul_model(opset=3)
    out = remove_initializer_from_input(model)
    assert out.ir_version == 3
    assert "W" in [i.name for i in out.graph.input]


def test_remove_initializer_from_input_folds_modern_opset():
    # An opset >= 6 IR-3 model (e.g. the opset-8 ``resnet101-v1-7``) is safe to
    # fold: the initializer-input is dropped and the model bumped to IR 4 so the
    # freed constant becomes foldable and value-baking fusions can fire.
    from onnxsim.onnx_simplifier import remove_initializer_from_input

    model = _ir3_matmul_model(opset=8)
    out = remove_initializer_from_input(model)
    assert out.ir_version == 4
    assert "W" not in [i.name for i in out.graph.input]
