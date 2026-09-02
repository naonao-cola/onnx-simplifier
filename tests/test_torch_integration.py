"""PyTorch (``onnx2torch``) integration test.

``docs/dlpack-executor.md`` frames onnxsim's DLPack executor boundary as an
"embeddability" seam: onnxsim should be droppable into another ONNX-based
compiler or runtime stack with the host only implementing one executor
callback. ``tests/test_nncase_integration.py``, ``tests/test_tvm_integration.py``,
and ``tests/test_tinygrad_integration.py`` are the regression tests for that
claim against nncase's, Apache TVM's, and tinygrad's own ONNX frontends. This
module is the analogous test for PyTorch, via
`onnx2torch <https://github.com/ENOT-AutoDL/onnx2torch>`_'s ``convert``, which
imports an ONNX ``ModelProto`` directly into an ``fx.GraphModule`` -- no
separate compile step, and it runs on plain CPU PyTorch, so, like tinygrad,
this is a genuine independent ONNX backend to cross-check onnxsim's
simplifications against.

Every op used by the models below (``Conv``, ``BatchNormalization``, ``Relu``,
``Transpose``, ``Shape``, ``Gather``, ``Concat``, ``Reshape``) is implemented
by onnx2torch's node converter registry.

This module feeds onnxsim's simplified output into ``onnx2torch`` and checks
two things:

1. Simplification must not make a graph *harder* for onnx2torch to convert
   and run: whatever it accepted before simplification, it must still accept
   after.
2. Simplification must not change what onnx2torch computes: the original and
   simplified graphs must agree numerically.

A third, more interesting case is also covered: the classic
``Shape -> Gather -> Concat -> Reshape`` chain that ships fully-constant
dimensions. onnxsim constant-folds the whole chain into a literal ``Reshape``
target shape, which onnx2torch ingests fine either way here, but is the same
regression guard the nncase/TVM/tinygrad modules carry for a
fixed-shape-preferring backend.

``torch`` and ``onnx2torch`` are not part of onnxsim's test requirements, so
the whole module is skipped when either is not installed. The dedicated
``backend-integration`` CI workflow installs them and runs these tests; the
regular build-and-test matrix skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

torch = pytest.importorskip("torch", reason="torch is not installed")
onnx2torch = pytest.importorskip("onnx2torch", reason="onnx2torch is not installed")

import onnxsim  # noqa: E402  (imported after the torch/onnx2torch availability check)

_OPSET = 17
_IR_VERSION = 8


def _model(
    body, initializer=(), opset=_OPSET, ir_version=_IR_VERSION
) -> onnx.ModelProto:
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


def _rand(*shape, seed=0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def _conv_bn_relu() -> onnx.ModelProto:
    """Conv -> BatchNormalization -> Relu: onnxsim folds BN into the Conv."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    scale = numpy_helper.from_array(_rand(8, seed=2), "scale")
    shift = numpy_helper.from_array(_rand(8, seed=3), "shift")
    mean = numpy_helper.from_array(_rand(8, seed=4), "mean")
    var = numpy_helper.from_array(np.abs(_rand(8, seed=5)) + 1.0, "var")
    return _model(
        """
        conv_bn_relu (float[1,3,8,8] x) => (float[1,8,8,8] y)
        {
          c = Conv<pads = [1, 1, 1, 1]>(x, w)
          bn = BatchNormalization(c, scale, shift, mean, var)
          y = Relu(bn)
        }
        """,
        initializer=[w, scale, shift, mean, var],
    )


def _redundant_transpose() -> onnx.ModelProto:
    """An identity Transpose (perm=[0,1,2,3]) that onnxsim removes outright."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    return _model(
        """
        redundant_transpose (float[1,3,8,8] x) => (float[1,8,8,8] y)
        {
          t = Transpose<perm = [0, 1, 2, 3]>(x)
          c = Conv<pads = [1, 1, 1, 1]>(t, w)
          y = Relu(c)
        }
        """,
        initializer=[w],
    )


def _foldable_shape_reshape() -> onnx.ModelProto:
    """Shape -> Gather -> Concat -> Reshape, fully determined by constants.

    onnxsim folds the whole chain into the ``Reshape``'s literal target
    shape, which onnx2torch ingests either way -- the same regression guard
    the nncase/TVM/tinygrad equivalents carry for backends that do prefer a
    literal target.
    """
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    return _model(
        """
        foldable_shape_reshape (float[1,3,8,8] x) => (float[1,8,64] y)
        <int64[1] idx = {0}, int64[1] ch = {8}, int64[1] m1 = {-1}>
        {
          c = Conv<pads = [1, 1, 1, 1]>(x, w)
          r = Relu(c)
          shp = Shape(r)
          n = Gather<axis = 0>(shp, idx)
          newshape = Concat<axis = 0>(n, ch, m1)
          y = Reshape(r, newshape)
        }
        """,
        initializer=[w],
    )


_MODELS = {
    "conv_bn_relu": _conv_bn_relu,
    "redundant_transpose": _redundant_transpose,
    "foldable_shape_reshape": _foldable_shape_reshape,
}


def _random_feeds(model: onnx.ModelProto, seed: int = 0):
    """Deterministic random float32 input for every non-initializer graph input."""
    rng = np.random.RandomState(seed)
    initializer_names = {init.name for init in model.graph.initializer}
    feeds = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        feeds[inp.name] = (rng.rand(*shape).astype(np.float32) - 0.5) * 2.0
    return feeds


def _run_with_torch(model: onnx.ModelProto, feeds):
    """Import ``model`` into onnx2torch and evaluate it.

    ``onnx2torch.convert`` builds an ``fx.GraphModule`` whose forward takes
    the graph's non-initializer inputs positionally (in ``graph.input``
    order) and returns a single tensor, or a tuple in ``graph.output`` order
    when there is more than one output.
    """
    torch_model = onnx2torch.convert(model)
    initializer_names = {init.name for init in model.graph.initializer}
    args = [
        torch.from_numpy(feeds[inp.name])
        for inp in model.graph.input
        if inp.name not in initializer_names
    ]
    with torch.no_grad():
        outputs = torch_model(*args)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    return [o.detach().numpy() for o in outputs]


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_simplified_model_runs_on_torch(name):
    """The simplified graph must always convert and run on onnx2torch."""
    model = _MODELS[name]()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=0)
    simplified_out = _run_with_torch(simplified, feeds)

    # Whatever onnx2torch did with the original graph, it must still accept
    # the simplified one -- and if it also accepted the original, the two
    # must agree numerically (simplification must be semantics-preserving on
    # onnx2torch).
    try:
        original_out = _run_with_torch(model, feeds)
    except Exception:
        return
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)


def test_simplify_is_bit_exact_on_torch():
    """Removing a redundant (identity) Transpose must not change the torch result.

    onnxsim removes the ``Transpose`` node entirely, so the simplified graph
    is exactly the same arithmetic the original ran -- onnx2torch's output
    must be bit-identical.
    """
    model = _redundant_transpose()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=1)
    original_out = _run_with_torch(model, feeds)
    simplified_out = _run_with_torch(simplified, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_array_equal(orig, simp)


def test_torch_output_matches_onnx_reference():
    """The onnx2torch result for a simplified model must match onnx's reference evaluator."""
    model = _conv_bn_relu()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=2)
    torch_out = _run_with_torch(simplified, feeds)

    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(simplified)
    reference_out = evaluator.run(None, feeds)
    for ref, cand in zip(reference_out, torch_out):
        np.testing.assert_allclose(ref, cand, rtol=1e-3, atol=1e-4)


def test_simplify_constant_folds_shape_reshape_chain():
    """Regression guard for the constant-fold-Shapes-into-Reshape case.

    onnxsim must collapse the ``Shape -> Gather -> Concat -> Reshape`` chain
    into a literal ``Reshape`` target, and the simplified graph must still
    compute the same result on onnx2torch as the original.
    """
    model = _foldable_shape_reshape()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    # The whole Shape/Gather/Concat chain collapses into the Reshape's target.
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=3)
    simplified_out = _run_with_torch(simplified, feeds)
    original_out = _run_with_torch(model, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)
