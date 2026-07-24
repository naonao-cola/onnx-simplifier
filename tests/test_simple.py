import tempfile
import torch
import onnx
import onnxruntime
import onnxsim
import os

from onnx import helper, numpy_helper, TensorProto
from typing import Optional
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
    opt = export_simplify_and_check_by_python_api(MG(), (x, b), export_kwargs={"dynamo": True})
    sess = onnxruntime.InferenceSession(opt.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(out_names, { opt.graph.input[0].name: x.numpy(), opt.graph.input[1].name: b.numpy() })
    assert outs[0].shape == MG()(x, b).shape


def test_transformer():
    model = torch.nn.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1)
    model.to('cpu').to(torch.float32)
    model.eval()

    inputs = (
        torch.rand((100, 2, 256), dtype=torch.float32),
        torch.rand((15, 2, 256), dtype=torch.float32),
    )
    export_simplify_and_check_by_python_api(model, inputs, export_kwargs={"dynamo": True})


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
            xup = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3 = self.bn2(self.conv2_2(F.relu(self.conv2_1(xup))))
            x4 = F.relu(self.conv3_3(self.conv3_2(F.relu(self.conv3_1(x3)))))
            x5 = self.bn3(x4)
            return F.softsign(x5)


    inp = torch.rand(1, 3, 96, 96)
    net = Net()
    opt = export_simplify_and_check_by_python_api(net, (inp,), export_kwargs={"opset_version": 9})

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
            return torch.cat((torch.mean(x, 1).unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    export_simplify_and_check_by_python_api(Model(), (torch.rand(20, 20),), export_kwargs={"opset_version": 9})


def test_trilinear():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, input_tensor):
            return torch.nn.functional.interpolate(input_tensor, scale_factor=[4, 4, 4], mode='trilinear')

    x = torch.rand(1, 8, 20, 120, 120)
    opt = export_simplify_and_check_by_python_api(
        Model(),
        (x,),
        export_kwargs={
            "opset_version": 11,
            "export_params": True,
        })
    sess = onnxruntime.InferenceSession(opt.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(out_names, { opt.graph.input[0].name: x.numpy() })
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
        export_kwargs={"opset_version": 15, "input_names": ["i"]})


def test_ext():
    class MyMod(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.param = torch.nn.Parameter(torch.rand(1024, 1024, 1024, dtype=torch.float16))

        def forward(self, x):
            return self.param * x

    module = MyMod()
    inputs = (torch.tensor(1),)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_fn = os.path.join(tmpdirname, "tmp.onnx")
        torch.onnx.export(module, inputs, model_fn, dynamo=False, input_names=["x"])
        model = onnx.load(model_fn)
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
        "conv_w", TensorProto.FLOAT, [8, 3, 3, 3],
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
        helper.make_node("QuantizeLinear", ["conv_w", "conv_w_scale", "cw_zp"], ["cwq"]),
        helper.make_node("DequantizeLinear", ["cwq", "conv_w_scale", "cw_zp2"], ["cwdq"]),
        helper.make_node("Conv", ["Xdq", "cwdq"], ["conv_out"], kernel_shape=[3, 3]),
        helper.make_node("GlobalAveragePool", ["conv_out"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        # second activation QDQ + weight QDQ feeding a Gemm.
        helper.make_node("QuantizeLinear", ["flat", "act_scale2", "a_zp2"], ["flatq"]),
        helper.make_node("DequantizeLinear", ["flatq", "act_scale2", "a_zp2"], ["flatdq"]),
        helper.make_node("QuantizeLinear", ["gemm_w", "gemm_w_scale", "gw_zp"], ["gwq"]),
        helper.make_node("DequantizeLinear", ["gwq", "gemm_w_scale", "gw_zp2"], ["gwdq"]),
        helper.make_node("Gemm", ["flatdq", "gwdq"], ["Y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "modelopt_fp8_qdq",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 6, 6])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        [
            conv_w, gemm_w, conv_ws, gemm_ws, act_s, act_s2,
            fp8_zero_point("a_zp"), fp8_zero_point("a_zp2"),
            fp8_zero_point("cw_zp"), fp8_zero_point("cw_zp2"),
            fp8_zero_point("gw_zp"), fp8_zero_point("gw_zp2"),
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
