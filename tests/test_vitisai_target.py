"""Tests for the Vitis AI NPU target legalizer (onnxsim/vitisai_target.py).

The EP aborts on ``Conv`` nodes that rely on default attributes, so the
legalizer materializes them explicitly; the checker flags the remaining
known-bad shapes (default-attr ``Conv``, ``LSTM``, ``If``, bf16-typed
tensors).
"""

import numpy as np
import onnx
from onnx import parser

from onnxsim.vitisai_target import check_vitisai_support, legalize_for_vitisai


def _model(body, initializer=(), opset=18, ir_version=10):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["": {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def _conv_weight():
    return onnx.numpy_helper.from_array(
        np.full((8, 4, 3, 3), 0.1, dtype=np.float32), "W"
    )


def test_legalize_materializes_conv_defaults():
    model = _model(
        """agraph (float[1,4,8,8] x) => (float[1,8,6,6] y)
        {
          y = Conv(x, W)
        }""",
        initializer=[_conv_weight()],
    )
    out = legalize_for_vitisai(model)
    # The input model is never mutated.
    assert {a.name for a in model.graph.node[0].attribute} == set()
    attrs = {a.name: a for a in out.graph.node[0].attribute}
    assert set(attrs) == {"strides", "pads", "dilations", "kernel_shape", "group"}
    assert list(attrs["strides"].ints) == [1, 1]
    assert list(attrs["pads"].ints) == [0, 0, 0, 0]
    assert list(attrs["dilations"].ints) == [1, 1]
    assert list(attrs["kernel_shape"].ints) == [3, 3]
    assert attrs["group"].i == 1
    onnx.checker.check_model(out)


def test_legalize_resolves_weight_through_dq_chain():
    # The int8 shape every quantizer emits: the Conv weight arrives via
    # DequantizeLinear, and the kernel shape must come from the quantized
    # initializer behind it.
    model = _model(
        """agraph (float[1,4,8,8] x) => (float[1,8,6,6] y)
        {
          Wd = DequantizeLinear(Wq, Ws, Wz)
          y = Conv(x, Wd)
        }""",
        initializer=[
            onnx.numpy_helper.from_array(
                np.full((8, 4, 3, 3), 10, dtype=np.int8), "Wq"
            ),
            onnx.numpy_helper.from_array(np.array(0.01, dtype=np.float32), "Ws"),
            onnx.numpy_helper.from_array(np.array(0, dtype=np.int8), "Wz"),
        ],
    )
    out = legalize_for_vitisai(model)
    conv = next(n for n in out.graph.node if n.op_type == "Conv")
    attrs = {a.name: a for a in conv.attribute}
    assert set(attrs) == {"strides", "pads", "dilations", "kernel_shape", "group"}
    assert list(attrs["kernel_shape"].ints) == [3, 3]
    onnx.checker.check_model(out)


def test_legalize_leaves_explicit_conv_alone():
    model = _model(
        """agraph (float[1,4,8,8] x) => (float[1,8,6,6] y)
        {
          y = Conv(x, W) <strides = [1, 1], pads = [0, 0, 0, 0],
            dilations = [1, 1], kernel_shape = [3, 3], group = 1>
        }""",
        initializer=[_conv_weight()],
    )
    out = legalize_for_vitisai(model)
    assert {a.name for a in out.graph.node[0].attribute} == {
        "strides",
        "pads",
        "dilations",
        "kernel_shape",
        "group",
    }
    assert check_vitisai_support(out) == []


def test_legalize_skips_dynamic_weight():
    # A Conv whose weight is a graph input (not an initializer) has no static
    # dims to materialize from -- left alone, still valid.
    model = _model(
        """agraph (float[1,4,8,8] x, float[8,4,3,3] W) => (float[1,8,6,6] y)
        {
          y = Conv(x, W)
        }"""
    )
    out = legalize_for_vitisai(model)
    assert {a.name for a in out.graph.node[0].attribute} == set()
    onnx.checker.check_model(out)


def test_check_flags_default_attr_conv():
    model = _model(
        """agraph (float[1,4,8,8] x) => (float[1,8,6,6] y)
        {
          y = Conv(x, W)
        }""",
        initializer=[_conv_weight()],
    )
    messages = check_vitisai_support(model)
    assert len(messages) == 1
    assert "Conv" in messages[0]


def test_check_flags_lstm():
    model = _model(
        """agraph (float[3,1,4] x) => (float[3,1,1,2] y, float[1,1,2] yh, float[1,1,2] yc)
        <float[1,8,4] w = {0.1}, float[1,8,2] r = {0.1}>
        {
          y, yh, yc = LSTM(x, w, r) <hidden_size = 2>
        }"""
    )
    messages = check_vitisai_support(model)
    assert len(messages) == 1
    assert "LSTM" in messages[0]


def test_check_flags_if():
    # Bisected on a ResNeXt-FPN detector: the graph without its If node
    # compiles on the EP, adding just the If aborts session creation in
    # the MLIR lowering -- while minimal static If/Loop probes compile
    # fine, so every If is flagged (the checker cannot tell the fatal
    # dynamic-shape lowering from a safe one).
    model = _model(
        """agraph (bool c, float[2,2] x) => (float[2,2] y)
        {
          y = If (c) <then_branch = g1 () => (float[2,2] t) { t = Identity (x) },
            else_branch = g2 () => (float[2,2] e) { e = Relu (x) }>
        }"""
    )
    messages = check_vitisai_support(model)
    assert len(messages) == 1
    assert "If" in messages[0]


def test_check_flags_bf16():
    model = _model(
        """agraph (bfloat16[2,2] x) => (bfloat16[2,2] y)
        {
          y = Identity(x)
        }"""
    )
    messages = check_vitisai_support(model)
    # Both the bf16 input and the bf16 output are flagged.
    assert len(messages) == 2
    assert all("bf16" in message for message in messages)


def test_check_accepts_file_path(tmp_path):
    path = str(tmp_path / "model.onnx")
    onnx.save(
        _model(
            """agraph (float[1,4,8,8] x) => (float[1,8,6,6] y)
            {
              y = Conv(x, W) <strides = [1, 1], pads = [0, 0, 0, 0],
                dilations = [1, 1], kernel_shape = [3, 3], group = 1>
            }""",
            initializer=[_conv_weight()],
        ),
        path,
    )
    assert check_vitisai_support(path) == []
    out = legalize_for_vitisai(path)
    onnx.checker.check_model(out)
