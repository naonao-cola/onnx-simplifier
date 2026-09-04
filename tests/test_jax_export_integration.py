"""Integration tests exporting ONNX models from JAX -- and from Flax NNX
modules built on top of it -- via the third-party `jax2onnx
<https://github.com/enpasos/jax2onnx>`_ converter, then feeding that ONNX
graph through onnxsim.simplify().

JAX has no official/first-party ONNX exporter (see
https://github.com/jax-ml/jax/issues/7629); jax2onnx traces JAX code through
the same jaxpr machinery XLA compiles and lowers it directly to ONNX. That
makes it a genuinely independent onnx-producing frontend from the
torch.onnx.export-based fixtures most of this suite's other export
integration tests (test_torch_export_integration.py,
test_mmdeploy_integration.py) are built on, with its own lowering quirks
worth exercising end-to-end:

- Flax's channels-last (NHWC) convolution convention is lowered to ONNX's
  NCHW-only Conv wrapped in explicit, non-identity NHWC<->NCHW Transposes --
  a fusion (BatchNormalization into Conv) has to fire straight through them.
- A reshape depending on a symbolic batch dimension lowers to a
  Shape/Gather/Concat/Reshape chain that must keep working (not get folded
  into a wrong static shape) for every batch size after simplification.
- jax.lax.cond lowers to an ONNX `If`.

jax2onnx is not part of onnxsim's test requirements (it drags in jax, flax,
onnxruntime, and a handful of other heavy packages as its own dependencies),
so the whole module is skipped when it is not installed. To run locally::

    pip install jax2onnx
    pip install --force-reinstall --no-deps .   # the onnxsim under test
    pytest tests/test_jax_export_integration.py -v
"""

import numpy as np
import onnx
import pytest

import onnxsim

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
jax2onnx = pytest.importorskip("jax2onnx")
onnxruntime = pytest.importorskip("onnxruntime")

to_onnx = jax2onnx.to_onnx


def _simplify(model):
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


def test_jax_export_basic_function():
    # A plain jax.numpy function, with no Flax module in the picture at all.
    def f(x):
        return jnp.tanh(x @ x.T) + 1.0

    x = np.random.RandomState(0).randn(4, 4).astype(np.float32)
    model = to_onnx(f, [(4, 4)])
    sim_model = _simplify(model)

    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    expected = np.asarray(f(jnp.asarray(x)))
    actual = _run(sim_model, {input_name: x})[output_name]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class _ConvBnRelu(nnx.Module):
    def __init__(self, *, rngs):
        self.conv = nnx.Conv(3, 8, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn = nnx.BatchNorm(8, use_running_average=True, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.bn(self.conv(x)))


def test_jax_export_conv_bn_relu_fuses():
    # fuse_bn_into_conv must fire straight through jax2onnx's NHWC<->NCHW
    # layout Transposes: Flax convs are channels-last, so jax2onnx wraps the
    # ONNX (NCHW-only) Conv in a Transpose on the way in and one on the way
    # out. Those two Transposes use non-identity permutations and must
    # survive simplification untouched; only the BatchNormalization should
    # disappear, folded into the Conv.
    module = _ConvBnRelu(rngs=nnx.Rngs(0))
    x = np.random.RandomState(1).randn(2, 8, 8, 3).astype(np.float32)  # NHWC

    model = to_onnx(module, [(2, 8, 8, 3)])
    orig_op_types = [n.op_type for n in model.graph.node]
    assert "BatchNormalization" in orig_op_types
    assert orig_op_types.count("Transpose") == 2

    sim_model = _simplify(model)
    sim_op_types = [n.op_type for n in sim_model.graph.node]
    assert "BatchNormalization" not in sim_op_types
    assert "Conv" in sim_op_types
    assert sim_op_types.count("Transpose") == 2

    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    expected = np.asarray(module(jnp.asarray(x)))
    orig_out = _run(model, {input_name: x})[output_name]
    sim_out = _run(sim_model, {input_name: x})[output_name]
    np.testing.assert_allclose(orig_out, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sim_out, expected, rtol=1e-4, atol=1e-5)


def test_jax_export_preserves_dynamic_batch_dimension():
    # jax2onnx's counterpart of test_dynamo_export_preserves_dynamic_batch
    # (tests/test_torch_export_integration.py): a symbolic batch dimension
    # ("B") makes `x.reshape(x.shape[0], -1)` lower to a genuine
    # Shape/Gather/.../Reshape chain reading the runtime batch size, rather
    # than a literal target-shape constant. onnxsim must not fold that chain
    # into a shape baked in for one particular batch size.
    def f(x):
        return x.reshape(x.shape[0], -1)

    model = to_onnx(f, [("B", 3, 4)])
    sim_model = _simplify(model)

    in_dim0 = sim_model.graph.input[0].type.tensor_type.shape.dim[0]
    out_dim0 = sim_model.graph.output[0].type.tensor_type.shape.dim[0]
    assert in_dim0.dim_value == 0 and in_dim0.dim_param
    assert out_dim0.dim_value == 0 and out_dim0.dim_param

    input_name = sim_model.graph.input[0].name
    output_name = sim_model.graph.output[0].name
    for batch_size in (1, 2, 5):
        x = np.random.RandomState(batch_size).randn(batch_size, 3, 4).astype(np.float32)
        expected = np.asarray(f(jnp.asarray(x)))
        actual = _run(sim_model, {input_name: x})[output_name]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_jax_export_control_flow():
    # jax.lax.cond lowers to an ONNX `If`, jax2onnx's equivalent of the
    # torch.cond-based test_dynamo_export_with_control_flow.
    def f(x):
        return jax.lax.cond(jnp.sum(x) > 0, lambda t: t * 2.0, lambda t: t * -2.0, x)

    model = to_onnx(f, [(3, 4)])
    sim_model = _simplify(model)
    assert "If" in [n.op_type for n in sim_model.graph.node]

    input_name = sim_model.graph.input[0].name
    output_name = sim_model.graph.output[0].name
    positive = np.ones((3, 4), dtype=np.float32)
    negative = -np.ones((3, 4), dtype=np.float32)
    for x in (positive, negative):
        expected = np.asarray(f(jnp.asarray(x)))
        actual = _run(sim_model, {input_name: x})[output_name]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
