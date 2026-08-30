"""Integration tests exercising both of torch.onnx.export's backends -- the
legacy TorchScript-based exporter (``dynamo=False``) and the newer
torch.export-based one (``dynamo=True``, the default since PyTorch 2.9) --
against onnxsim.simplify(). Most torch-based tests elsewhere in this suite
are regression tests pinned to one specific historical bug and default to
whichever exporter onnxsim.test_utils picks; this file instead treats
"does onnxsim work equally well on both exporters' output" as the thing
under test.
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime
import torch

import onnxsim
from onnxsim.test_utils import export_simplify_and_check_by_python_api


def _export_and_simplify(module, inputs, **export_kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_fn = os.path.join(tmpdir, "model.onnx")
        torch.onnx.export(module, inputs, model_fn, **export_kwargs)
        model = onnx.load(model_fn)
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    onnx.checker.check_model(sim_model)
    return sim_model


def _run(model, feeds):
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    return dict(zip(out_names, sess.run(out_names, feeds)))


class _ConvBnRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


def test_dynamo_export_basic_module():
    # The new torch.export-based exporter (dynamo=True, the default since
    # PyTorch 2.9) on the simplest possible module -- most other torch-based
    # tests in this suite default to the legacy exporter via
    # onnxsim.test_utils's own dynamo=False default.
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    export_simplify_and_check_by_python_api(
        Linear(), torch.randn(2, 10), export_kwargs={"dynamo": True}
    )


def test_dynamo_and_legacy_export_produce_equivalent_simplified_models():
    # The same module exported through both backends must simplify to
    # models that are numerically equivalent, even though the raw exported
    # graphs (and possibly the simplified op sequences) can differ -- the
    # legacy exporter traces through TorchScript, while torch.export
    # captures via Dynamo and applies its own set of ATen decompositions.
    module = _ConvBnRelu().eval()
    x = torch.randn(2, 3, 8, 8)

    legacy = _export_and_simplify(
        module, (x,), dynamo=False, input_names=["x"], output_names=["y"]
    )
    dynamo = _export_and_simplify(
        module, (x,), dynamo=True, input_names=["x"], output_names=["y"]
    )

    feeds = {"x": x.numpy()}
    legacy_out = _run(legacy, feeds)["y"]
    dynamo_out = _run(dynamo, feeds)["y"]
    np.testing.assert_allclose(legacy_out, dynamo_out, rtol=1e-4, atol=1e-5)


def test_dynamo_export_conv_bn_relu_fuses():
    # fuse_bn_into_conv should fire the same way regardless of which
    # exporter produced the graph. BatchNormalization must disappear and
    # the fused Conv must still be numerically correct against the
    # unfused PyTorch module.
    module = _ConvBnRelu().eval()
    x = torch.randn(2, 3, 8, 8)
    sim_model = _export_and_simplify(
        module, (x,), dynamo=True, input_names=["x"], output_names=["y"]
    )

    op_types = [n.op_type for n in sim_model.graph.node]
    assert "BatchNormalization" not in op_types
    assert "Conv" in op_types

    expected = module(x).detach().numpy()
    actual = _run(sim_model, {"x": x.numpy()})["y"]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_dynamo_export_preserves_dynamic_batch():
    # The dynamo=True counterpart of test_dynamic_axes_preserve_dynamic_dimension
    # (tests/test_python_api.py, regression test for issue #299): a Shape
    # computation reading a dynamic dimension must not be constant-folded
    # away. The new exporter spells dynamic shapes with torch.export.Dim
    # instead of the legacy exporter's dynamic_axes dict, and its Dynamo
    # capture can shape the exported graph differently, so this needs its
    # own coverage rather than assuming the legacy-exporter test generalizes.
    class DynamicReshape(torch.nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

    module = DynamicReshape().eval()
    x = torch.randn(2, 3, 4, 5)
    batch = torch.export.Dim("batch")
    sim_model = _export_and_simplify(
        module,
        (x,),
        dynamo=True,
        input_names=["x"],
        output_names=["y"],
        dynamic_shapes={"x": {0: batch}},
    )

    in_dim0 = sim_model.graph.input[0].type.tensor_type.shape.dim[0]
    out_dim0 = sim_model.graph.output[0].type.tensor_type.shape.dim[0]
    assert in_dim0.dim_value == 0 and in_dim0.dim_param
    assert out_dim0.dim_value == 0 and out_dim0.dim_param

    for batch_size in (1, 2, 7):
        x_np = np.random.rand(batch_size, 3, 4, 5).astype(np.float32)
        (result,) = _run(sim_model, {"x": x_np}).values()
        assert result.shape == (batch_size, 3, 20)
        np.testing.assert_allclose(
            result, x_np.reshape(batch_size, 3, 20), rtol=1e-5, atol=1e-6
        )


def test_dynamo_export_with_control_flow():
    # torch.export (and therefore dynamo=True) captures control flow via
    # torch.cond rather than tracing through a single branch the way the
    # legacy TorchScript exporter's `torch.jit.trace` effectively would --
    # this exercises the resulting ONNX `If` op through onnxsim.
    class Cond(torch.nn.Module):
        def forward(self, x):
            return torch.cond(
                x.sum() > 0,
                lambda t: t * 2,
                lambda t: t * -2,
                (x,),
            )

    module = Cond().eval()
    positive = torch.ones(3, 4)
    negative = -torch.ones(3, 4)

    sim_model = _export_and_simplify(
        module, (positive,), dynamo=True, input_names=["x"], output_names=["y"]
    )
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "If" in op_types

    np.testing.assert_allclose(
        _run(sim_model, {"x": positive.numpy()})["y"],
        module(positive).detach().numpy(),
    )
    np.testing.assert_allclose(
        _run(sim_model, {"x": negative.numpy()})["y"],
        module(negative).detach().numpy(),
    )
