"""Apache TVM integration test.

``docs/dlpack-executor.md`` names TVM explicitly as one of the "another
ONNX-based compiler or runtime stack" targets the DLPack executor boundary was
designed to make onnxsim embeddable into. This module is the regression test
for that claim on the *Python* side: it feeds onnxsim's simplified output into
Apache TVM's Relax ONNX importer + compiler (``tvm.relax.frontend.onnx``,
TVM's current, non-legacy ONNX ingestion path) and checks two things:

1. Simplification must not make a graph *harder* to compile: whatever TVM
   accepted before simplification, it must still accept after.
2. Simplification must not change what TVM computes.

A third, more interesting case is also covered: the classic
``Shape -> Gather -> Concat -> Reshape`` chain that ships fully-constant
dimensions is rejected outright by TVM's ONNX frontend (it insists the
``Reshape`` target be a compile-time ``Shape``, not a runtime tensor), but
compiles fine once onnxsim constant-folds the chain to a literal shape. So for
TVM, onnxsim is not just "does no harm" -- it is sometimes the difference
between a graph TVM can ingest and one it can't.

``apache-tvm`` is heavy (~100MB wheel) and not part of onnxsim's test
requirements, so the whole module is skipped when it is not installed. The
dedicated ``backend-integration`` CI workflow installs it and runs these
tests; the regular build-and-test matrix skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

tvm = pytest.importorskip("tvm", reason="apache-tvm is not installed")
onnx_frontend = pytest.importorskip(
    "tvm.relax.frontend.onnx", reason="apache-tvm is not installed"
)
from tvm import relax  # noqa: E402  (only valid once the importorskip above passed)

import onnxsim  # noqa: E402  (imported after the tvm availability check)

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
    """Conv -> Mul(scale) -> Add(shift) -> Relu: onnxsim fuses scale/shift into Conv."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    scale = numpy_helper.from_array(_rand(1, 8, 1, 1, seed=2), "scale")
    shift = numpy_helper.from_array(_rand(1, 8, 1, 1, seed=3), "shift")
    return _model(
        """
        conv_bn_relu (float[1,3,8,8] x) => (float[1,8,8,8] y)
        {
          c = Conv<pads = [1, 1, 1, 1]>(x, w)
          m = Mul(c, scale)
          a = Add(m, shift)
          y = Relu(a)
        }
        """,
        [w, scale, shift],
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
        [w],
    )


def _foldable_shape_reshape() -> onnx.ModelProto:
    """Shape -> Gather -> Concat -> Reshape, fully determined by constants.

    TVM's Relax ONNX frontend requires a ``Reshape``'s target shape to be a
    compile-time ``Shape`` value, not a runtime tensor, so it rejects this
    graph as-is (see module docstring). onnxsim folds the whole chain into a
    literal target shape, which TVM then imports fine.
    """
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    idx = numpy_helper.from_array(np.array([0], np.int64), "idx")
    minus1 = numpy_helper.from_array(np.array([-1], np.int64), "m1")
    ch = numpy_helper.from_array(np.array([8], np.int64), "ch")
    return _model(
        """
        foldable_shape_reshape (float[1,3,8,8] x) => (float[1,8,64] y)
        {
          c = Conv<pads = [1, 1, 1, 1]>(x, w)
          r = Relu(c)
          shp = Shape(r)
          n = Gather<axis = 0>(shp, idx)
          newshape = Concat<axis = 0>(n, ch, m1)
          y = Reshape(r, newshape)
        }
        """,
        [w, idx, minus1, ch],
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


def _compile_and_run_with_tvm(model: onnx.ModelProto, feeds):
    """Import ``model`` with the Relax ONNX frontend, build it, and run it on CPU."""
    shape_dict = {name: list(arr.shape) for name, arr in feeds.items()}
    mod = onnx_frontend.from_onnx(model, shape_dict=shape_dict)
    mod = relax.transform.LegalizeOps()(mod)
    executable = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(executable, tvm.cpu())
    # Graph inputs are single-input in every model here; feeds preserves the
    # same graph.input order the frontend used to build the function's params.
    args = [tvm.runtime.tensor(value) for value in feeds.values()]
    result = vm["main"](*args)
    if isinstance(result, tvm.runtime.Tensor):
        return [result.numpy()]
    return [r.numpy() for r in result]


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_simplified_model_compiles_and_runs_on_tvm(name):
    """The simplified graph must always import, compile, and run on TVM."""
    model = _MODELS[name]()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=0)
    simplified_out = _compile_and_run_with_tvm(simplified, feeds)

    # Whatever TVM did with the original graph, it must still accept the
    # simplified one -- and if it also accepted the original, the two must
    # agree numerically (simplification must be semantics-preserving on TVM).
    try:
        original_out = _compile_and_run_with_tvm(model, feeds)
    except Exception:
        return
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)


def test_simplify_is_bit_exact_on_tvm():
    """Removing a redundant (identity) Transpose must not change the TVM result."""
    model = _redundant_transpose()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=1)
    original_out = _compile_and_run_with_tvm(model, feeds)
    simplified_out = _compile_and_run_with_tvm(simplified, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=0, atol=0)


def test_tvm_output_matches_onnx_reference():
    """The TVM result for a simplified model must match onnx's own reference evaluator."""
    model = _conv_bn_relu()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=2)
    tvm_out = _compile_and_run_with_tvm(simplified, feeds)

    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(simplified)
    reference_out = evaluator.run(None, feeds)
    for ref, cand in zip(reference_out, tvm_out):
        np.testing.assert_allclose(ref, cand, rtol=1e-3, atol=1e-4)


def test_simplify_is_required_for_tvm_import():
    """Regression guard for the constant-fold-enables-import case.

    If a future onnxsim change stops folding this chain, or a future TVM
    release starts accepting it unfolded, this test's assumptions should be
    revisited -- it deliberately pins today's behavior on both sides.
    """
    model = _foldable_shape_reshape()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    # The whole Shape/Gather/Concat chain collapses into the Reshape's target.
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=3)
    with pytest.raises(Exception):
        _compile_and_run_with_tvm(model, feeds)

    # The simplified graph must import and run cleanly.
    _compile_and_run_with_tvm(simplified, feeds)
