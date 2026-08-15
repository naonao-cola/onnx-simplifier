"""Halide integration test.

``docs/dlpack-executor.md`` frames onnxsim's DLPack executor boundary as an
"embeddability" seam: onnxsim should be droppable into another ONNX-based
compiler or runtime stack (a different ORT build, IREE, TVM, a hardware
vendor runtime) with the host only implementing one executor callback.
``tests/test_tvm_integration.py`` is the regression test for that claim
against Apache TVM's real ONNX frontend. This module is the analogous test
for `Halide <https://halide-lang.org/>`_, a widely used ahead-of-time array
compiler (image/tensor pipelines compiled to native code, used in production
inference and image-processing stacks).

Halide has no ready-made ONNX frontend to hand a whole ``ModelProto`` to (unlike
TVM's ``tvm.relax.frontend.onnx``), so this module ships a small, deliberately
minimal ONNX-subset-to-Halide lowering (``_compile_and_run_with_halide``
below) covering just the ops the test models use: ``Conv``, ``Mul``, ``Add``,
``Relu``, ``Transpose``, and a constant-shape-only ``Reshape``. This mirrors
how a real integrator embedding onnxsim into a Halide-based backend would
write exactly one lowering pass per op they care about, then hand onnxsim's
*simplified* graph to it -- onnxsim's fusions and constant folding shrink the
op set the backend needs to support.

The ``Reshape`` restriction is deliberate and mirrors a real AOT-compiler
constraint (the same one the TVM Relax ONNX frontend enforces, see
``test_tvm_integration.py``): Halide pipelines are ahead-of-time compiled
with statically scheduled loop bounds, so a ``Reshape`` target must be a
compile-time constant, not a tensor computed at run time from
``Shape``/``Gather``/``Concat``. This lowering pass does not implement
``Shape``/``Gather``/``Concat`` at all, so it rejects a
``Shape -> Gather -> Concat -> Reshape`` chain outright -- exactly the
pattern onnxsim constant-folds into a literal ``Reshape`` target, which the
lowering pass then accepts fine. So, as with TVM, onnxsim simplification is
sometimes the difference between a graph this Halide lowering can ingest and
one it can't.

The ``halide`` PyPI package is not part of onnxsim's test requirements, so
the whole module is skipped when it is not installed. The dedicated
``halide-integration`` CI workflow installs it and runs these tests; the
regular build-and-test matrix skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

hl = pytest.importorskip("halide", reason="halide is not installed")

import onnxsim  # noqa: E402  (imported after the halide availability check)

_OPSET = 17
_IR_VERSION = 8


def _safe_jit_target():
    """The host target, minus AVX-512/AVX-VNNI.

    Halide's default JIT target is ``get_host_target()``, detected from
    ``cpuid`` at process start. On some virtualized cloud CI runners cpuid
    advertises AVX-512/AVX-VNNI support the actual (possibly heterogeneous)
    fleet hardware or hypervisor does not reliably provide, and code Halide
    generates using those instructions then ``SIGILL``s at ``realize()``
    time. AVX-512 is not needed for these tiny test models, so this drops it
    (and the newer AVX-VNNI) unconditionally rather than trusting cpuid.
    """
    target = hl.get_host_target()
    for feature_name in (
        "AVX512",
        "AVX512_Cannonlake",
        "AVX512_SapphireRapids",
        "AVX512_Skylake",
        "AVX512_Zen4",
        "AVX512_Zen5",
        "AVX512_KNL",
        "AVXVNNI",
    ):
        feature = getattr(hl.TargetFeature, feature_name, None)
        if feature is not None and target.has_feature(feature):
            target = target.without_feature(feature)
    return target


_JIT_TARGET = _safe_jit_target()


def _model(nodes, inputs, outputs, initializers, name) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = _IR_VERSION
    onnx.checker.check_model(model)
    return model


def _rand(*shape, seed=0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def _conv_bn_relu() -> onnx.ModelProto:
    """Conv -> Mul(scale) -> Add(shift) -> Relu."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    scale = numpy_helper.from_array(_rand(1, 8, 1, 1, seed=2), "scale")
    shift = numpy_helper.from_array(_rand(1, 8, 1, 1, seed=3), "shift")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Mul", ["c", "scale"], ["m"]),
        helper.make_node("Add", ["m", "shift"], ["a"]),
        helper.make_node("Relu", ["a"], ["y"]),
    ]
    return _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
        [w, scale, shift],
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

    See the module docstring: the Halide lowering below only supports a
    ``Reshape`` whose target shape is already a compile-time constant, so it
    rejects this graph as-is and accepts it once onnxsim has folded the
    whole chain into a literal target shape.
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


# ---------------------------------------------------------------------------
# Minimal ONNX-subset -> Halide lowering.
#
# Tensor convention: an ONNX tensor of shape ``(d0, d1, ..., dn-1)`` is
# represented as a Halide ``Func`` defined over ``n`` ``Var``s in *reversed*
# axis order (``Func(v_{n-1}, ..., v_0)`` where ``v_i`` ranges over ``d_i``).
# This matches how ``hl.Buffer(numpy_array)`` itself lays out a numpy array
# (fastest-varying / last numpy axis becomes Halide's first Var), so numpy
# arrays can be fed in and read back without any axis shuffling.
# ---------------------------------------------------------------------------


def _get_attr(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
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


def _broadcast_shape(a_shape, b_shape):
    n = max(len(a_shape), len(b_shape))
    a = (1,) * (n - len(a_shape)) + tuple(a_shape)
    b = (1,) * (n - len(b_shape)) + tuple(b_shape)
    out = []
    for da, db in zip(a, b):
        if da != db and da != 1 and db != 1:
            raise ValueError(f"shapes {a_shape} and {b_shape} are not broadcastable")
        out.append(max(da, db))
    return tuple(out)


# The Halide Python bindings do not keep a wrapped numpy array (or the
# Buffer built from it) alive on their own -- a Func's definition only
# captures the Buffer's underlying data pointer, not a Python reference to
# it. If nothing else references the array/Buffer, Python's GC is free to
# collect it as soon as the wrapping helper returns, and a later allocation
# can then reuse that freed memory -- silently corrupting an *earlier*
# Func's data out from under it (see the "keepalive" append calls below).
# Every array/Buffer handed to Halide is therefore kept alive here for the
# lifetime of the whole test module.
_KEEPALIVE = []


def _conv2d(x_f, x_shape, w_arr, pads, b_arr=None):
    n, cin, h, w = x_shape
    cout, cin2, kh, kw = w_arr.shape
    assert cin == cin2, "Conv input/weight channel mismatch"
    pt, pl, pb, pr = pads
    oh, ow = h + pt + pb - kh + 1, w + pl + pr - kw + 1
    x_pad = hl.BoundaryConditions.constant_exterior(
        x_f, hl.f32(0), [(0, w), (0, h), (0, cin), (0, n)]
    )
    w_arr = np.ascontiguousarray(w_arr).copy()
    wbuf = hl.Buffer(w_arr)
    _KEEPALIVE.extend([w_arr, wbuf])
    r = hl.RDom([(0, cin), (0, kh), (0, kw)])  # r[0]=cin, r[1]=kh, r[2]=kw
    ow_, oh_, oc_, on_ = hl.Var("ow"), hl.Var("oh"), hl.Var("oc"), hl.Var("on")
    f = hl.Func("conv")
    if b_arr is None:
        f[ow_, oh_, oc_, on_] = hl.f32(0)
    else:
        # onnx's optional Conv "B" input: a length-Cout per-output-channel bias.
        b_arr = np.ascontiguousarray(b_arr).copy()
        bbuf = hl.Buffer(b_arr)
        _KEEPALIVE.extend([b_arr, bbuf])
        f[ow_, oh_, oc_, on_] = bbuf[oc_]
    f[ow_, oh_, oc_, on_] += (
        x_pad[ow_ + r[2] - pl, oh_ + r[1] - pt, r[0], on_] * wbuf[r[2], r[1], r[0], oc_]
    )
    return f, (n, cout, oh, ow)


def _transpose(x_f, x_shape, perm):
    ndim = len(x_shape)
    out_shape = tuple(x_shape[p] for p in perm)
    out_vars = [hl.Var(f"t{i}") for i in range(ndim)]
    in_onnx_idx = [None] * ndim
    for out_axis in range(ndim):
        in_onnx_idx[perm[out_axis]] = out_vars[ndim - 1 - out_axis]
    in_args = list(reversed(in_onnx_idx))
    f = hl.Func("transpose")
    f[tuple(out_vars)] = x_f[tuple(in_args)]
    return f, out_shape


def _binop(op_fn, a_f, a_shape, b_f, b_shape, name):
    out_shape = _broadcast_shape(a_shape, b_shape)
    ndim = len(out_shape)
    out_vars = [hl.Var(f"{name}{i}") for i in range(ndim)]

    def operand_args(op_shape):
        args = []
        for k in range(len(op_shape)):
            onnx_axis = len(op_shape) - 1 - k
            args.append(0 if op_shape[onnx_axis] == 1 else out_vars[k])
        return args

    f = hl.Func(name)
    f[tuple(out_vars)] = op_fn(
        a_f[tuple(operand_args(a_shape))], b_f[tuple(operand_args(b_shape))]
    )
    return f, out_shape


def _relu(x_f, x_shape):
    ndim = len(x_shape)
    vs = [hl.Var(f"r{i}") for i in range(ndim)]
    f = hl.Func("relu")
    f[tuple(vs)] = hl.max(x_f[tuple(vs)], hl.f32(0))
    return f, x_shape


def _array_to_func(arr: np.ndarray):
    ndim = arr.ndim
    arr = np.ascontiguousarray(arr).copy()
    buf = hl.Buffer(arr)
    _KEEPALIVE.extend([arr, buf])
    vs = [hl.Var(f"k{i}") for i in range(ndim)]
    f = hl.Func("const")
    f[tuple(vs)] = buf[tuple(vs)]
    return f, arr.shape


def _realize_to_numpy(f, shape):
    return np.asanyarray(f.realize(list(reversed(shape)), _JIT_TARGET))


def _compile_and_run_with_halide(model: onnx.ModelProto, feeds):
    """Lower ``model`` to Halide ``Func``s with the minimal op set above, run it.

    Raises ``NotImplementedError`` for any op this lowering does not cover
    (in particular ``Shape``/``Gather``/``Concat``, see module docstring).
    """
    consts = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    # name -> either a plain numpy constant, or an (func, shape) pair.
    values = dict(consts)

    for inp in model.graph.input:
        if inp.name in consts:
            continue
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        values[inp.name] = _array_to_func(feeds[inp.name])
        assert values[inp.name][1] == shape

    def func_and_shape(name):
        v = values[name]
        return v if isinstance(v, tuple) else _array_to_func(v)

    def const(name):
        v = values[name]
        if isinstance(v, np.ndarray):
            return v
        raise NotImplementedError(
            f"'{name}' is required as a compile-time constant but is only "
            "known at run time -- this Halide lowering needs the graph "
            "simplified (constant-folded) first"
        )

    for node in _toposort(model.graph):
        op = node.op_type
        if op == "Conv":
            x_f, x_shape = func_and_shape(node.input[0])
            w = const(node.input[1])
            b = const(node.input[2]) if len(node.input) > 2 else None
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
            values[node.output[0]] = _conv2d(x_f, x_shape, w, pads, b)
        elif op in ("Mul", "Add"):
            a_f, a_shape = func_and_shape(node.input[0])
            b_f, b_shape = func_and_shape(node.input[1])
            fn = (lambda a, b: a * b) if op == "Mul" else (lambda a, b: a + b)
            values[node.output[0]] = _binop(fn, a_f, a_shape, b_f, b_shape, op.lower())
        elif op == "Relu":
            x_f, x_shape = func_and_shape(node.input[0])
            values[node.output[0]] = _relu(x_f, x_shape)
        elif op == "Transpose":
            x_f, x_shape = func_and_shape(node.input[0])
            perm = _get_attr(node, "perm", list(reversed(range(len(x_shape)))))
            values[node.output[0]] = _transpose(x_f, x_shape, perm)
        elif op == "Reshape":
            x_f, x_shape = func_and_shape(node.input[0])
            target = const(node.input[1]).tolist()
            values[node.output[0]] = _realize_to_numpy(x_f, x_shape).reshape(target)
        else:
            raise NotImplementedError(
                f"op '{op}' not supported by the Halide test backend"
            )

    outputs = []
    for out in model.graph.output:
        v = values[out.name]
        outputs.append(v if isinstance(v, np.ndarray) else _realize_to_numpy(*v))
    return outputs


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_simplified_model_compiles_and_runs_on_halide(name):
    """The simplified graph must always lower, compile, and run on Halide."""
    model = _MODELS[name]()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=0)
    simplified_out = _compile_and_run_with_halide(simplified, feeds)

    # Whatever the Halide lowering did with the original graph, it must
    # still accept the simplified one -- and if it also accepted the
    # original, the two must agree numerically (simplification must be
    # semantics-preserving through this lowering too).
    try:
        original_out = _compile_and_run_with_halide(model, feeds)
    except NotImplementedError:
        return
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=1e-4, atol=1e-5)


def test_simplify_is_bit_exact_on_halide():
    """Removing a redundant (identity) Transpose must not change the Halide result."""
    model = _redundant_transpose()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=1)
    original_out = _compile_and_run_with_halide(model, feeds)
    simplified_out = _compile_and_run_with_halide(simplified, feeds)
    for orig, simp in zip(original_out, simplified_out):
        np.testing.assert_allclose(orig, simp, rtol=0, atol=0)


def test_halide_output_matches_onnx_reference():
    """The Halide result for a simplified model must match onnx's own reference evaluator."""
    model = _conv_bn_relu()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _random_feeds(model, seed=2)
    halide_out = _compile_and_run_with_halide(simplified, feeds)

    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(simplified)
    reference_out = evaluator.run(None, feeds)
    for ref, cand in zip(reference_out, halide_out):
        np.testing.assert_allclose(ref, cand, rtol=1e-4, atol=1e-5)


def test_simplify_is_required_for_halide_lowering():
    """Regression guard for the constant-fold-enables-lowering case.

    If a future onnxsim change stops folding this chain, or this Halide
    lowering ever grows ``Shape``/``Gather``/``Concat`` support, this test's
    assumptions should be revisited -- it deliberately pins today's behavior
    on both sides, mirroring ``test_simplify_is_required_for_tvm_import``.
    """
    model = _foldable_shape_reshape()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    # The whole Shape/Gather/Concat chain collapses into the Reshape's target.
    assert len(simplified.graph.node) < len(model.graph.node)

    feeds = _random_feeds(model, seed=3)
    with pytest.raises(NotImplementedError):
        _compile_and_run_with_halide(model, feeds)

    # The simplified graph must lower and run cleanly.
    _compile_and_run_with_halide(simplified, feeds)
