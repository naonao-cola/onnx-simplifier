"""nncase integration test.

nncase <https://github.com/kendryte/nncase>`_ is a model compiler for the
Kendryte K230 / K510 application processors (and a generic ``cpu`` target). It
imports an ONNX ``ModelProto`` directly, compiles it to a chosen target, and can
evaluate the compiled module on the host -- so it is a genuine, independent ONNX
backend we can use to double check that onnxsim's simplifications are
semantics-preserving outside of onnxruntime.

nncase's supported ONNX ops are listed in
``https://github.com/kendryte/nncase/blob/master/docs/onnx_ops.md``. Every op
used by the models below is taken from that supported set (Conv,
BatchNormalization, Relu, Transpose, Shape, Gather, Concat, Reshape, ...).

This module feeds onnxsim's simplified output into nncase and checks two things:

1. Simplification must not make a graph *harder* to compile: whatever nncase
   accepted before simplification, it must still accept after.
2. Simplification must not change what nncase computes: the original and
   simplified graphs must agree numerically on nncase's own compiler/evaluator.

A third, more interesting case is also covered: the classic
``Shape -> Gather -> Concat -> Reshape`` chain that ships fully-constant
dimensions. onnxsim constant-folds the whole chain into a literal ``Reshape``
target shape, which nncase ingests fine and which keeps the graph from carrying
a runtime-computed shape that a fixed-shape compiler would rather not see.

nncase is a .NET application (it bundles ``Nncase.Compiler.dll`` and needs the
.NET runtime) and is not part of onnxsim's test requirements, so the whole
module is skipped when it is not installed. The dedicated
``nncase-integration`` CI workflow installs it and runs these tests; the regular
build-and-test matrix skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

nncase = pytest.importorskip("nncase", reason="nncase is not installed")
from nncase import CompileOptions, Compiler, ImportOptions, RuntimeTensor  # noqa: E402

import onnxsim  # noqa: E402  (imported after the nncase availability check)

_OPSET = 17
_IR_VERSION = 8


def _model(nodes, inputs, outputs, initializers, name) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = _IR_VERSION
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
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node(
            "BatchNormalization", ["c", "scale", "shift", "mean", "var"], ["bn"]
        ),
        helper.make_node("Relu", ["bn"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
        [w, scale, shift, mean, var],
        "conv_bn_relu",
    )


def _redundant_transpose() -> onnx.ModelProto:
    """An identity Transpose (perm=[0,1,2,3]) that onnxsim removes outright."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    nodes = [
        helper.make_node("Transpose", ["x"], ["t"], perm=[0, 1, 2, 3]),
        helper.make_node("Conv", ["t", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
        [w],
        "redundant_transpose",
    )


def _foldable_shape_reshape() -> onnx.ModelProto:
    """Shape -> Gather -> Concat -> Reshape, fully determined by constants.

    onnxsim folds the whole chain into the ``Reshape``'s literal target shape,
    which nncase ingests cleanly and keeps a fixed-shape compiler from having to
    materialize a runtime-computed shape.
    """
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    idx = numpy_helper.from_array(np.array([0], np.int64), "idx")
    minus1 = numpy_helper.from_array(np.array([-1], np.int64), "m1")
    ch = numpy_helper.from_array(np.array([8], np.int64), "ch")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Shape", ["r"], ["shp"]),
        helper.make_node("Gather", ["shp", "idx"], ["n"], axis=0),
        helper.make_node("Concat", ["n", "ch", "m1"], ["newshape"], axis=0),
        helper.make_node("Reshape", ["r", "newshape"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 64])],
        [w, idx, minus1, ch],
        "foldable_shape_reshape",
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


def _compile_and_run_with_nncase(model: onnx.ModelProto, feeds):
    """Import ``model`` into nncase, compile for the cpu target, and evaluate it."""
    compile_options = CompileOptions()
    compile_options.target = "cpu"
    # These dumps only write files; turn them off so the test leaves no clutter.
    compile_options.dump_asm = False
    compile_options.dump_ir = False

    compiler = Compiler(compile_options)
    compiler.import_onnx(model.SerializeToString(), ImportOptions())
    compiler.compile()

    evaluator = compiler.create_evaluator(0)
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [
        inp.name for inp in model.graph.input if inp.name not in initializer_names
    ]
    for i, name in enumerate(input_names):
        evaluator.set_input_tensor(i, RuntimeTensor.from_numpy(feeds[name]))
    evaluator.run()

    return [
        evaluator.get_output_tensor(i).to_numpy() for i in range(evaluator.outputs_size)
    ]


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_simplified_model_compiles_and_runs_on_nncase(name):
    """The simplified graph must always import, compile, and run on nncase."""
    model = _MODELS[name]()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=0)
    simplified_out = _compile_and_run_with_nncase(simplified, feeds)

    # Whatever nncase did with the original graph, it must still accept the
    # simplified one -- and if it also accepted the original, the two must
    # agree numerically (simplification must be semantics-preserving on nncase).
    try:
        original_out = _compile_and_run_with_nncase(model, feeds)
    except Exception:
        return
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)


def test_simplify_is_bit_exact_on_nncase():
    """Removing a redundant (identity) Transpose must not change the nncase result.

    onnxsim removes the ``Transpose`` node entirely, so the simplified graph is
    exactly the same arithmetic the original compiled to -- nncase's output must
    be bit-identical.
    """
    model = _redundant_transpose()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=1)
    original_out = _compile_and_run_with_nncase(model, feeds)
    simplified_out = _compile_and_run_with_nncase(simplified, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_array_equal(orig, simp)


def test_nncase_output_matches_onnx_reference():
    """The nncase result for a simplified model must match onnx's reference evaluator."""
    model = _conv_bn_relu()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=2)
    nncase_out = _compile_and_run_with_nncase(simplified, feeds)

    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(simplified)
    reference_out = evaluator.run(None, feeds)
    for ref, cand in zip(reference_out, nncase_out):
        np.testing.assert_allclose(ref, cand, rtol=1e-3, atol=1e-4)


def test_simplify_constant_folds_shape_reshape_chain():
    """Regression guard for the constant-fold-Shapes-into-Reshape case.

    onnxsim must collapse the ``Shape -> Gather -> Concat -> Reshape`` chain
    into a literal ``Reshape`` target so a fixed-shape compiler like nncase
    ingests it cleanly.
    """
    model = _foldable_shape_reshape()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    # The whole Shape/Gather/Concat chain collapses into the Reshape's target.
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=3)
    simplified_out = _compile_and_run_with_nncase(simplified, feeds)

    # The simplified graph must match the original nncase computation. Even if a
    # future nncase release ingests the unfolded chain too, the simplified graph
    # must still be numerically equivalent to the original.
    original_out = _compile_and_run_with_nncase(model, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-3, atol=1e-4)
