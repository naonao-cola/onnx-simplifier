"""A handful of minimal PyTorch -> ONNX -> onnxsim.simplify() smoke tests.

Unlike tests/test_torch_export_integration.py (which cross-checks the legacy
and dynamo exporter backends against each other) or tests/test_pruning.py
(which uses these exports only as one-off spot-checks before hand-building
onnx.parser equivalents), this file is just the simplest possible worked
examples of the export-then-simplify pattern: one module per test, one thing
asserted per test.
"""

import os
import tempfile

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

import onnxsim
from onnxsim.test_utils import export_simplify_and_check_by_python_api

onnxruntime = pytest.importorskip("onnxruntime")


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class _ConvBnRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class _ReshapeFlatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def test_simplify_accepts_a_plain_linear_export():
    # The most basic case: export succeeds and onnxsim accepts the result.
    sim_model = export_simplify_and_check_by_python_api(_Linear(), torch.randn(2, 10))
    assert sim_model.graph.output[0].name


def test_conv_bn_relu_fuses_batchnorm_into_conv():
    # A classic onnxsim simplification: BatchNormalization folds into the
    # preceding Conv's weights, so it disappears from the simplified graph
    # while the numeric result is unchanged. do_constant_folding=False is
    # needed so the raw export actually keeps a BatchNormalization node --
    # with it left at its default (True), the legacy exporter's own
    # constant-folding pass already fuses BN into Conv itself, before
    # onnxsim ever sees the graph.
    module = _ConvBnRelu().eval()
    x = torch.randn(1, 3, 8, 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_fn = os.path.join(tmpdir, "model.onnx")
        torch.onnx.export(
            module,
            (x,),
            model_fn,
            dynamo=False,
            do_constant_folding=False,
            input_names=["x"],
            output_names=["y"],
        )
        raw_model = onnx.load(model_fn)

    sim_model, check_ok = onnxsim.simplify(raw_model, check_n=3)
    assert check_ok
    onnx.checker.check_model(sim_model)

    raw_op_types = [n.op_type for n in raw_model.graph.node]
    sim_op_types = [n.op_type for n in sim_model.graph.node]
    assert "BatchNormalization" in raw_op_types
    assert "BatchNormalization" not in sim_op_types
    assert "Conv" in sim_op_types

    expected = module(x).detach().numpy()
    sess = onnxruntime.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = sess.run(["y"], {"x": x.numpy()})
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_static_reshape_constant_folds_into_initializer():
    # x.reshape(x.shape[0], -1) on a statically-shaped input: the exporter
    # already resolves the target shape to a literal at trace time (a single
    # Constant node feeding Reshape), and onnxsim folds that Constant node
    # away entirely into an initializer, leaving only the Reshape node.
    module = _ReshapeFlatten().eval()
    x = torch.randn(2, 3, 4)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_fn = os.path.join(tmpdir, "model.onnx")
        torch.onnx.export(
            module, (x,), model_fn, dynamo=False, input_names=["x"], output_names=["y"]
        )
        raw_model = onnx.load(model_fn)

    sim_model, check_ok = onnxsim.simplify(raw_model, check_n=3)
    assert check_ok
    onnx.checker.check_model(sim_model)

    assert "Constant" in [n.op_type for n in raw_model.graph.node]
    assert [n.op_type for n in sim_model.graph.node] == ["Reshape"]

    expected = module(x).detach().numpy()
    (actual,) = ReferenceEvaluator(sim_model).run(None, {"x": x.numpy()})
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
