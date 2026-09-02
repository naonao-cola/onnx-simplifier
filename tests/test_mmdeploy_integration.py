"""OpenMMLab mmdeploy integration test.

`mmdeploy <https://github.com/open-mmlab/mmdeploy>`_ is OpenMMLab's model
deployment toolbox: it converts PyTorch models from the OpenMMLab codebases
(mmdetection, mmpose, mmsegmentation, ...) to ONNX and then feeds that ONNX
graph to one of several backend runtimes for inference. Its ONNX Runtime
backend wrapper, ``mmdeploy.backend.onnxruntime.wrapper.ORTWrapper``, is the
actual production code path a deployed mmdeploy pipeline runs a model
through -- it is not a bespoke test harness, so it is a genuine, independent
consumer of onnxsim's output.

This module feeds an onnxsim-simplified graph into ``ORTWrapper`` and checks
that simplification is transparent to it:

1. Whatever ``ORTWrapper`` could load and run before simplification, it must
   still load and run after -- onnxsim must not produce a graph mmdeploy's
   own runtime wrapper rejects.
2. The result must be unchanged: ``ORTWrapper`` run on the original graph and
   on the simplified graph must agree numerically.

This intentionally goes through ``ORTWrapper`` rather than mmdeploy's
``mmdeploy.apis.onnx.export.export`` PyTorch-to-ONNX exporter: as of
mmdeploy 1.3.1 (the latest PyPI release), ``export`` unconditionally wraps
the call in mmdeploy's ``RewriterContext`` function-patching machinery,
which raises on exit against current PyTorch's dynamo-based
``torch.onnx.export`` internals -- a preexisting mmdeploy/PyTorch version
incompatibility, unrelated to onnxsim, that would make these tests flaky for
reasons having nothing to do with what they are meant to guard. Building the
input graph with ``onnx.parser`` instead (the same convention every other
migrated test in this repo uses) exercises the same downstream consumer
(``ORTWrapper``) without depending on that broken path.

``mmdeploy`` (and its ``torch`` / ``onnxruntime`` dependencies) is heavy and
not part of onnxsim's test requirements, so the whole module is skipped when
it is not installed. The dedicated ``backend-integration`` CI workflow
installs it and runs these tests; the regular build-and-test matrix skips
them.

Note for whoever wires up the CI job: the mmdeploy 1.3.1 wheel on PyPI still
imports ``mmcv.Config``, which no longer exists as of mmcv 2.0 -- it was
only ever an mmcv 2.0.0rc-era backward-compat alias for
``mmengine.Config``, and mmdeploy's own ``main`` branch has since moved to
importing ``mmengine.Config`` directly. So pip-installing mmdeploy's PyPI
release next to any mmcv>=2.0 final release fails at import time before this
module's ``pytest.importorskip`` even gets a chance to run. Install mmdeploy
from its GitHub ``main`` branch instead (``pip install
git+https://github.com/open-mmlab/mmdeploy.git``) to pick up that fix.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

torch = pytest.importorskip("torch", reason="mmdeploy needs torch, which is not installed")
ort_wrapper_mod = pytest.importorskip(
    "mmdeploy.backend.onnxruntime.wrapper",
    reason="mmdeploy is not installed",
)
ORTWrapper = ort_wrapper_mod.ORTWrapper

import onnxsim  # noqa: E402  (imported after the mmdeploy availability check)

_OPSET = 17
_IR_VERSION = 9


def _model(body, initializer=(), opset=_OPSET, ir_version=_IR_VERSION):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def _rand(*shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def _conv_bn_relu_with_redundant_transpose():
    """Transpose(identity) -> Conv -> BatchNormalization -> Relu.

    onnxsim removes the identity ``Transpose`` outright and folds the
    ``BatchNormalization`` into the ``Conv``, so the simplified graph is
    ``Conv -> Relu`` only.
    """
    w = onnx.numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    scale = onnx.numpy_helper.from_array(_rand(8, seed=2), "scale")
    shift = onnx.numpy_helper.from_array(_rand(8, seed=3), "shift")
    mean = onnx.numpy_helper.from_array(_rand(8, seed=4), "mean")
    var = onnx.numpy_helper.from_array(np.abs(_rand(8, seed=5)) + 1.0, "var")
    return _model(
        """
        conv_bn_relu (float[1,3,8,8] input) => (float[1,8,8,8] output)
        {
          t = Transpose<perm = [0, 1, 2, 3]>(input)
          c = Conv<pads = [1, 1, 1, 1]>(t, w)
          bn = BatchNormalization(c, scale, shift, mean, var)
          output = Relu(bn)
        }
        """,
        initializer=[w, scale, shift, mean, var],
    )


def _foldable_shape_reshape():
    """Shape -> Gather -> Concat -> Reshape, fully determined by constants.

    onnxsim collapses the whole chain into the ``Reshape``'s literal target
    shape.
    """
    w = onnx.numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    return _model(
        """
        foldable_shape_reshape (float[1,3,8,8] input) => (float[1,8,64] output)
        <int64[1] idx = {0}, int64[1] ch = {8}, int64[1] m1 = {-1}>
        {
          c = Conv<pads = [1, 1, 1, 1]>(input, w)
          r = Relu(c)
          shp = Shape(r)
          n = Gather<axis = 0>(shp, idx)
          newshape = Concat<axis = 0>(n, ch, m1)
          output = Reshape(r, newshape)
        }
        """,
        initializer=[w],
    )


def _random_feeds(model, seed=0):
    rng = np.random.RandomState(seed)
    initializer_names = {init.name for init in model.graph.initializer}
    feeds = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        feeds[inp.name] = torch.from_numpy(
            ((rng.rand(*shape) - 0.5) * 2.0).astype(np.float32)
        )
    return feeds


def _run_with_ort_wrapper(model, feeds, tmp_path, name):
    path = str(tmp_path / f"{name}.onnx")
    onnx.save(model, path)
    output_names = [o.name for o in model.graph.output]
    wrapper = ORTWrapper(path, "cpu", output_names=output_names)
    return wrapper(feeds)


def test_mmdeploy_ortwrapper_matches_after_simplify(tmp_path):
    """mmdeploy's own ONNXRuntime backend wrapper must agree pre/post simplify."""
    model = _conv_bn_relu_with_redundant_transpose()
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok

    orig_op_types = [n.op_type for n in model.graph.node]
    sim_op_types = [n.op_type for n in sim_model.graph.node]
    assert "Transpose" in orig_op_types and "Transpose" not in sim_op_types
    assert "BatchNormalization" in orig_op_types
    assert "BatchNormalization" not in sim_op_types

    feeds = _random_feeds(model, seed=0)
    orig_out = _run_with_ort_wrapper(model, feeds, tmp_path, "orig")
    sim_out = _run_with_ort_wrapper(sim_model, feeds, tmp_path, "sim")

    for name in orig_out:
        torch.testing.assert_close(orig_out[name], sim_out[name], rtol=0, atol=0)


def test_mmdeploy_ortwrapper_shape_reshape_chain(tmp_path):
    """The Shape/Gather/Concat/Reshape fold must still run cleanly on ORTWrapper."""
    model = _foldable_shape_reshape()
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(sim_model.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=1)
    orig_out = _run_with_ort_wrapper(model, feeds, tmp_path, "orig_reshape")
    sim_out = _run_with_ort_wrapper(sim_model, feeds, tmp_path, "sim_reshape")

    for name in orig_out:
        torch.testing.assert_close(
            orig_out[name], sim_out[name], rtol=1e-5, atol=1e-6
        )


def test_mmdeploy_ortwrapper_output_matches_onnx_reference(tmp_path):
    """The mmdeploy-wrapped ORT result for a simplified model must match onnx's
    own reference evaluator, so onnxsim's output isn't merely "not crashing"
    under mmdeploy's wrapper but numerically correct."""
    model = _conv_bn_relu_with_redundant_transpose()
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=2)
    wrapped_out = _run_with_ort_wrapper(sim_model, feeds, tmp_path, "sim_ref")

    from onnx.reference import ReferenceEvaluator

    np_feeds = {name: tensor.numpy() for name, tensor in feeds.items()}
    evaluator = ReferenceEvaluator(sim_model)
    output_names = [o.name for o in sim_model.graph.output]
    reference_out = evaluator.run(output_names, np_feeds)

    for name, ref in zip(output_names, reference_out):
        np.testing.assert_allclose(
            wrapped_out[name].numpy(), ref, rtol=1e-4, atol=1e-5
        )
