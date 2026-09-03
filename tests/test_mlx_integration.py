"""MLX integration test.

``docs/dlpack-executor.md`` frames onnxsim's DLPack executor boundary as an
"embeddability" seam: onnxsim should be droppable into another ONNX-based
compiler or runtime stack with the host only implementing one executor
callback. ``tests/test_tinygrad_integration.py`` and
``tests/test_torch_integration.py`` are the regression tests for that claim
against tinygrad's and PyTorch's (via ``onnx2torch``) own ONNX-consuming
paths. This module is the analogous test for
`MLX <https://github.com/ml-explore/mlx>`_, Apple's NumPy-like array
framework.

Unlike tinygrad and onnx2torch, MLX has no ONNX importer of its own to hand a
whole ``ModelProto`` to (the only PyPI package named for this, ``mlx-onnx``,
exports MLX callables *to* ONNX -- the opposite direction). So, mirroring how
``tests/test_halide_integration.py`` handles the same situation for Halide,
this module ships a small, deliberately minimal ONNX-subset-to-MLX lowering
(``_run_with_mlx`` below) covering just the ops the test models use: ``Conv``,
``BatchNormalization``, ``Relu``, ``Transpose``, ``Shape``, ``Gather``,
``Concat``, and ``Reshape``. This mirrors how a real integrator embedding
onnxsim into an MLX-based backend would write exactly one lowering pass per
op they care about, then hand onnxsim's *simplified* graph to it.

Unlike Halide's lowering, this one does not need to reject the
``Shape -> Gather -> Concat -> Reshape`` chain: MLX arrays are eager (a
``Shape`` node's output is just ``mx.array(x.shape)``, immediately usable by
``Gather``/``Concat``/``Reshape``), so the full op set the three test models
use is supported either way -- the ``foldable_shape_reshape`` test below
checks the constant-fold-through-onnxsim case is still numerically
equivalent, the same way ``test_tinygrad_integration.py`` and
``test_torch_integration.py`` do, rather than the "onnxsim makes this
ingestible at all" case Halide's stricter lowering demonstrates.

MLX's Linux wheel needs a backend extra (``pip install mlx[cpu]`` -- bare
``mlx`` fails to import with ``libmlx.so`` missing, since backend selection
on Linux is via the ``cpu``/``cuda``/``cuda12``/``cuda13`` extras); on the CPU
backend this runs on an ordinary CI runner, no GPU needed. The ``mlx`` PyPI
package is not part of onnxsim's test requirements, so the whole module is
skipped when it is not installed. The dedicated ``backend-integration`` CI
workflow installs it and runs these tests; the regular build-and-test matrix
skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

mx = pytest.importorskip("mlx.core", reason="mlx is not installed")

import onnxsim  # noqa: E402  (imported after the mlx availability check)

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
    shape, which the lowering below ingests either way -- the same
    regression guard the nncase/TVM/tinygrad/torch equivalents carry for
    backends that do prefer a literal target.
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


# ---------------------------------------------------------------------------
# Minimal ONNX-subset -> MLX lowering.
#
# Values are kept as plain ``mx.array``s in ONNX's own layout/axis order
# throughout (NCHW for the 4D activations here) -- MLX's ``reshape``,
# ``transpose``, ``take``, and ``concatenate`` all follow the same axis
# convention as onnx/numpy, so no axis bookkeeping is needed except at the
# ``conv2d`` boundary, which is channels-last (NHWC input, ``(Cout, KH, KW,
# Cin)`` weight) and is transposed into and back out of on each call.
# ---------------------------------------------------------------------------


def _get_attr(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
    return default


def _toposort(graph: onnx.GraphProto):
    """Kahn's algorithm over node I/O names; robust to any valid node order."""
    available = {i.name for i in graph.input} | {i.name for i in graph.initializer}
    remaining = list(graph.node)
    ordered = []
    while remaining:
        progressed = False
        still_remaining = []
        for node in remaining:
            if all(name in available or name == "" for name in node.input):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                still_remaining.append(node)
        if not progressed:
            raise RuntimeError("graph has an unresolvable or cyclic node dependency")
        remaining = still_remaining
    return ordered


def _conv2d(x, w, pads, bias=None):
    """ONNX ``Conv`` (NCHW, symmetric ``pads``) via MLX's channels-last conv2d."""
    pt, pl, pb, pr = pads
    if pt != pb or pl != pr:
        raise NotImplementedError(
            "this MLX test lowering only supports symmetric Conv padding"
        )
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    w_nhwc = mx.transpose(w, (0, 2, 3, 1))  # (Cout,Cin,KH,KW) -> (Cout,KH,KW,Cin)
    y_nhwc = mx.conv2d(x_nhwc, w_nhwc, stride=1, padding=(pt, pl))
    if bias is not None:
        y_nhwc = y_nhwc + mx.reshape(bias, (1, 1, 1, -1))
    return mx.transpose(y_nhwc, (0, 3, 1, 2))


def _batch_norm(x, scale, bias, mean, var, epsilon):
    c = scale.shape[0]
    scale_r = mx.reshape(scale, (1, c, 1, 1))
    bias_r = mx.reshape(bias, (1, c, 1, 1))
    mean_r = mx.reshape(mean, (1, c, 1, 1))
    var_r = mx.reshape(var, (1, c, 1, 1))
    return scale_r * (x - mean_r) * mx.rsqrt(var_r + epsilon) + bias_r


def _run_with_mlx(model: onnx.ModelProto, feeds):
    """Lower ``model`` to MLX ops with the minimal op set above, and run it."""
    values = {
        init.name: mx.array(numpy_helper.to_array(init))
        for init in model.graph.initializer
    }
    for inp in model.graph.input:
        if inp.name in values:
            continue
        values[inp.name] = mx.array(feeds[inp.name])

    for node in _toposort(model.graph):
        op = node.op_type
        ins = [values[n] for n in node.input]
        if op == "Conv":
            bias = ins[2] if len(ins) > 2 else None
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
            out = _conv2d(ins[0], ins[1], pads, bias)
        elif op == "BatchNormalization":
            epsilon = _get_attr(node, "epsilon", 1e-5)
            out = _batch_norm(ins[0], ins[1], ins[2], ins[3], ins[4], epsilon)
        elif op == "Relu":
            out = mx.maximum(ins[0], mx.array(0.0))
        elif op == "Transpose":
            perm = _get_attr(node, "perm", list(reversed(range(ins[0].ndim))))
            out = mx.transpose(ins[0], tuple(perm))
        elif op == "Shape":
            out = mx.array(list(ins[0].shape), dtype=mx.int64)
        elif op == "Gather":
            axis = _get_attr(node, "axis", 0)
            out = mx.take(ins[0], ins[1], axis=axis)
        elif op == "Concat":
            axis = _get_attr(node, "axis", 0)
            out = mx.concatenate(ins, axis=axis)
        elif op == "Reshape":
            target = [int(v) for v in ins[1].tolist()]
            out = mx.reshape(ins[0], target)
        else:
            raise NotImplementedError(f"op '{op}' not supported by the MLX test backend")
        values[node.output[0]] = out

    return [np.asarray(values[out.name]) for out in model.graph.output]


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_simplified_model_runs_on_mlx(name):
    """The simplified graph must always lower and run on MLX."""
    model = _MODELS[name]()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=0)
    simplified_out = _run_with_mlx(simplified, feeds)

    # Whatever the MLX lowering did with the original graph, it must still
    # accept the simplified one -- and if it also accepted the original, the
    # two must agree numerically (simplification must be semantics-preserving
    # on MLX too).
    try:
        original_out = _run_with_mlx(model, feeds)
    except Exception:
        return
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)


def test_simplify_is_bit_exact_on_mlx():
    """Removing a redundant (identity) Transpose must not change the MLX result.

    onnxsim removes the ``Transpose`` node entirely, so the simplified graph
    is exactly the same arithmetic the original ran -- MLX's output must be
    bit-identical.
    """
    model = _redundant_transpose()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=1)
    original_out = _run_with_mlx(model, feeds)
    simplified_out = _run_with_mlx(simplified, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_array_equal(orig, simp)


def test_mlx_output_matches_onnx_reference():
    """The MLX result for a simplified model must match onnx's reference evaluator."""
    model = _conv_bn_relu()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=2)
    mlx_out = _run_with_mlx(simplified, feeds)

    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(simplified)
    reference_out = evaluator.run(None, feeds)
    for ref, cand in zip(reference_out, mlx_out):
        np.testing.assert_allclose(ref, cand, rtol=1e-3, atol=1e-4)


def test_simplify_constant_folds_shape_reshape_chain():
    """Regression guard for the constant-fold-Shapes-into-Reshape case.

    onnxsim must collapse the ``Shape -> Gather -> Concat -> Reshape`` chain
    into a literal ``Reshape`` target, and the simplified graph must still
    compute the same result on MLX as the original.
    """
    model = _foldable_shape_reshape()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    # The whole Shape/Gather/Concat chain collapses into the Reshape's target.
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=3)
    simplified_out = _run_with_mlx(simplified, feeds)
    original_out = _run_with_mlx(model, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)
