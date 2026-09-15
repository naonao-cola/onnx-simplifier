#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generates onnxsim/graph_grad_templates_gen.py/.h from onnxscript.

The design discussed alongside onnxsim.graph_grad: instead of
hand-transcribing a VJP rule's graph construction once in graph_grad.py and a
second time in graph_grad.cpp (the duplication that let a real bug -- a wrong
`dvar` factor in _grad_batch_normalization/GradBatchNormalization -- through
review until a finite-difference test caught it), author the rule *once*, in
onnxscript, and have both languages instantiate the same checked-in
FunctionProto text via ONNX's own function-inlining machinery (onnx.inliner
in Python, onnx::inliner::InlineLocalFunctions in C++ -- both already
vendored, see third_party/onnx/onnx/inliner/), exactly like
generate_moe_function_templates.py above it in this directory produces a
checked-in header for contrib_schemas.cpp's own FunctionBuilder-based
instantiation -- the two scripts share the same shape: onnxscript is a
dev-only tool, never a build or runtime dependency, and its output is
checked in rather than regenerated on every build. "Add" and
"BatchNormalization" were the proof of concept; every other rule below whose
core arithmetic is separable from shape/attribute resolution (every
elementwise/broadcasting op) has since been templated the same way -- see
graph_grad.py's "Templated rules" section for the full list and for which
rules stay hand-written instead.

**Why these functions take no ONNX-level attributes.** graph_grad.py's
existing hand-written rules already resolve every rank/shape-dependent
choice (which axes broadcast, how many spatial dims a per-channel parameter
needs reshaping across) into concrete tensors -- constant axes lists via
`_Backward.int64_const`, reshaped broadcast operands via `Reshape` -- *before*
appending the op-specific arithmetic. Keeping that split (host code resolves
shape/rank into tensor-shaped data, the template consumes only tensors) means
every function below is a plain, rank-generic, attribute-free ONNX dataflow
graph: no `ref_attr_name` forwarding, no runtime `If`/`Loop`, none of the
sharp edges generate_moe_function_templates.py's own docstring documents for
onnxscript's attribute-parameter authoring. Broadcast-undoing on Add's
gradient is deliberately left to the *caller* (`_Backward.reduce_to`) for the
same reason: that logic already exists once, is already tested, and does not
need to be inside the template just because the template's caller was
rewritten.

**Numeric validation.** Each function's compiled FunctionProto is checked
(`onnx.checker.check_function`) and executed here against a finite-difference
reference computed from the exact formulas graph_grad.py's own hand-written
rule docstrings already document (`_grad_add`, `_grad_batch_normalization`) --
not merely compiled and trusted. See tests/test_graph_grad_templates.py for
the further check against torch.autograd on a real BatchNorm.

Run this script whenever the generated templates need to change:
    python3 scripts/codegen/generate_grad_templates.py \
        onnxsim/graph_grad_templates_gen.py \
        onnxsim/graph_grad_templates_gen.h
(the C++ header is optional -- pass just the first argument to regenerate
only the Python side; with no arguments at all, prints the Python module to
stdout instead of writing either file).
"""

import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op
from onnxscript.values import Opset

# A private domain this repo fully controls -- never registered with ONNX,
# never seen by a runtime (onnx.inliner/InlineLocalFunctions expands every
# call site before a step graph is returned; see qat_graph.GraphBuilder.call
# and make_step_graph). Version 1 forever: these functions are checked-in,
# reviewed source, not a public/evolving op set that needs versioning.
GRAD_DOMAIN = Opset("onnxsim.grad", 1)


@script(opset=GRAD_DOMAIN)
def GradAdd(g: FLOAT["..."]):
    """``Add``'s VJP before broadcast-undoing: da = db = g. Trivial by
    design -- this function exists to validate the instantiation mechanism
    itself (call node -> model-local function -> onnx.inliner expansion)
    against the simplest possible math, isolating plumbing bugs from math
    bugs. See _grad_add in graph_grad.py for the broadcast-undoing
    (`_Backward.reduce_to`) this deliberately leaves to the caller."""
    da = op.Identity(g)
    db = op.Identity(g)
    return da, db


@script(opset=GRAD_DOMAIN)
def GradBatchNormalization(
    g: FLOAT["..."],
    x: FLOAT["..."],
    mean_b: FLOAT["..."],
    var_b: FLOAT["..."],
    scale_b: FLOAT["..."],
    eps: FLOAT,
    channel_axes: INT64["..."],
    one: FLOAT,
    neg_half: FLOAT,
):
    """``BatchNormalization``'s five gradients (inference mode), transcribed
    from graph_grad.py's _grad_batch_normalization docstring:

        xc = x - mean         inv = 1 / sqrt(var + eps)
        xhat = xc * inv       dx = g * scale * inv
        dscale = sum(g * xhat)         db    = sum(g)
        dmean  = -sum(dx)              dvar  = -0.5 * sum(dx * xhat * inv)

    `mean_b`/`var_b`/`scale_b` arrive already reshaped to broadcast against
    `x` (``[1, C, 1, ..., 1]``, rank matching x) and `channel_axes` already
    holds "every axis but 1" as an int64 tensor -- both rank-dependent
    choices the caller (_Backward, via int64_const/Reshape) resolves before
    the call, exactly as today's hand-written rule does inline. That is what
    keeps this function itself rank-generic: no attribute here depends on
    x's rank, so the same compiled FunctionProto instantiates for a rank-2
    (just batch and channel) or rank-5 input alike.

    `one`/`neg_half` are the literals `1.0`/`-0.5` as ordinary float32
    tensor inputs, the same way the hand-written GradBatchNormalization
    passes them (`ctx.b().Const(1.0f)`/`ctx.b().Const(-0.5f)`) rather than
    as in-body `Constant`/`CastLike` nodes: this function is checked against
    graph_grad.py's BACKWARD_OPS/graph_grad.cpp's BackwardOps() allowlist
    once inlined, and neither `Constant` nor `CastLike` is a member of it
    (see qat_graph.py's EP_FRIENDLY_OPS note for why the allowlist is
    deliberately narrow) -- so this function takes them as data instead of
    manufacturing them itself.
    """
    xc = op.Sub(x, mean_b)
    inv = op.Div(one, op.Sqrt(op.Add(var_b, eps)))
    xhat = op.Mul(xc, inv)
    gs = op.Mul(g, scale_b)
    dx = op.Mul(gs, inv)
    dscale = op.ReduceSum(op.Mul(g, xhat), channel_axes, keepdims=0)
    dbias = op.ReduceSum(g, channel_axes, keepdims=0)
    dmean = op.Neg(op.ReduceSum(dx, channel_axes, keepdims=0))
    dvar = op.Mul(
        op.ReduceSum(op.Mul(op.Mul(dx, xhat), inv), channel_axes, keepdims=0),
        neg_half,
    )
    return dx, dscale, dbias, dmean, dvar


@script(opset=GRAD_DOMAIN)
def GradNeg(g: FLOAT["..."]):
    """``Neg``'s VJP: d/dx (-x) = -g. Closed-form, like ``GradAdd`` -- kept
    templated anyway so ``graph_grad.py``'s ``_grad_neg_templated`` and
    ``graph_grad.cpp``'s mirror share this one line instead of each
    spelling ``Neg(g)`` out separately."""
    dx = op.Neg(g)
    return dx


@script(opset=GRAD_DOMAIN)
def GradExp(g: FLOAT["..."], y: FLOAT["..."]):
    """``Exp``'s VJP, reusing the forward output ``y = exp(x)`` instead of
    calling ``Exp`` again: dx = g * y."""
    dx = op.Mul(g, y)
    return dx


@script(opset=GRAD_DOMAIN)
def GradSqrt(g: FLOAT["..."], y: FLOAT["..."], half: FLOAT):
    """``Sqrt``'s VJP, reusing the forward output ``y = sqrt(x)``: dx =
    0.5 * g / y. Singular at x = 0, as the derivative genuinely is."""
    dx = op.Div(op.Mul(g, half), y)
    return dx


@script(opset=GRAD_DOMAIN)
def GradLog(g: FLOAT["..."], x: FLOAT["..."]):
    """``Log``'s VJP: dx = g / x. Singular at x = 0, same reasoning as
    ``GradSqrt``."""
    dx = op.Div(g, x)
    return dx


@script(opset=GRAD_DOMAIN)
def GradSigmoid(g: FLOAT["..."], y: FLOAT["..."], one: FLOAT):
    """``Sigmoid``'s VJP, reusing the forward output ``y = sigmoid(x)``:
    dx = g * y * (1 - y)."""
    dy = op.Mul(y, op.Sub(one, y))
    dx = op.Mul(g, dy)
    return dx


@script(opset=GRAD_DOMAIN)
def GradTanh(g: FLOAT["..."], y: FLOAT["..."], one: FLOAT):
    """``Tanh``'s VJP, reusing the forward output ``y = tanh(x)``:
    dx = g * (1 - y^2)."""
    dy = op.Sub(one, op.Mul(y, y))
    dx = op.Mul(g, dy)
    return dx


@script(opset=GRAD_DOMAIN)
def GradErf(g: FLOAT["..."], x: FLOAT["..."], c: FLOAT):
    """``Erf``'s VJP: dx = g * c * exp(-x^2), c = 2/sqrt(pi) supplied by the
    caller (see ``GradBatchNormalization``'s docstring for why a literal
    arrives as a plain input rather than an in-body ``Constant``/
    ``CastLike``: neither is in ``graph_grad.BACKWARD_OPS``). This one is
    here entirely for GELU, which every transformer FFN block-wise QAT
    fine-tunes."""
    dy = op.Mul(c, op.Exp(op.Neg(op.Mul(x, x))))
    dx = op.Mul(g, dy)
    return dx


@script(opset=GRAD_DOMAIN)
def GradRelu(g: FLOAT["..."], x: FLOAT["..."], zero: FLOAT):
    """``Relu``'s VJP: mask = (x > 0) as float, dx = g * mask. The
    subgradient at exactly 0 is taken as 0 (strict ``Greater``), matching
    the straight-through masks ``adaround.py`` already builds and the
    hand-written rule this templates."""
    mask = op.Cast(op.Greater(x, zero), to=1)
    dx = op.Mul(g, mask)
    return dx


@script(opset=GRAD_DOMAIN)
def GradMul(g: FLOAT["..."], a: FLOAT["..."], b: FLOAT["..."]):
    """``Mul``'s VJP before broadcast-undoing: da = g * b, db = g * a."""
    da = op.Mul(g, b)
    db = op.Mul(g, a)
    return da, db


@script(opset=GRAD_DOMAIN)
def GradDiv(g: FLOAT["..."], a: FLOAT["..."], b: FLOAT["..."], y: FLOAT["..."]):
    """``Div``'s VJP before broadcast-undoing: da = g / b, db = -g * y / b,
    reusing the forward quotient ``y = a / b`` rather than recomputing a
    square -- one fewer node and no risk of overflowing b^2, exactly like
    the hand-written rule."""
    da = op.Div(g, b)
    db = op.Neg(op.Div(op.Mul(g, y), b))
    return da, db


_NP_TO_ONNX = {
    np.dtype("float32"): onnx.TensorProto.FLOAT,
    np.dtype("int64"): onnx.TensorProto.INT64,
}


def _run_function(fn: onnx.FunctionProto, feeds: dict, output_shapes: dict) -> list:
    """Evaluates a bare FunctionProto by wrapping it in a minimal model with
    a single call node -- the same call-node shape the real step-graph
    builder will use (onnx.reference.ReferenceEvaluator has no direct "run
    this FunctionProto" entry point). ``output_shapes`` must give every
    output's concrete shape: onnx.checker.check_model rejects a graph output
    with no shape at all, so this validator -- which always knows the shape
    it expects, being the one that chose the inputs -- states it explicitly
    rather than reaching for an "unknown rank" placeholder."""
    call = onnx.helper.make_node(
        fn.name, list(fn.input), list(fn.output), domain=fn.domain
    )
    value_info = [
        onnx.helper.make_tensor_value_info(
            n, _NP_TO_ONNX[feeds[n].dtype], list(feeds[n].shape)
        )
        for n in fn.input
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            n, onnx.TensorProto.FLOAT, list(output_shapes[n])
        )
        for n in fn.output
    ]
    graph = onnx.helper.make_graph([call], "poc", value_info, outputs)
    model = onnx.helper.make_model(
        graph,
        functions=[fn],
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid(fn.domain, 1),
        ],
    )
    onnx.checker.check_model(model)
    return ReferenceEvaluator(model).run(None, feeds)


def _validate_grad_add() -> None:
    fn = GradAdd.to_function_proto()
    onnx.checker.check_function(fn)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, 3)).astype(np.float32)
    da, db = _run_function(fn, {"g": g}, {"da": g.shape, "db": g.shape})
    np.testing.assert_array_equal(da, g)
    np.testing.assert_array_equal(db, g)


def _validate_grad_batch_normalization() -> None:
    fn = GradBatchNormalization.to_function_proto()
    onnx.checker.check_function(fn)

    rng = np.random.default_rng(0)
    n, c, h, w = 2, 3, 4, 4
    x = rng.standard_normal((n, c, h, w)).astype(np.float32)
    mean = rng.standard_normal((c,)).astype(np.float32)
    var = np.abs(rng.standard_normal((c,))).astype(np.float32) + 0.1
    scale = rng.standard_normal((c,)).astype(np.float32)
    bias = rng.standard_normal((c,)).astype(np.float32)
    eps = 1e-5
    g = rng.standard_normal((n, c, h, w)).astype(np.float32)

    def bcast(v: np.ndarray) -> np.ndarray:
        return v.reshape(1, c, 1, 1)

    analytic = _run_function(
        fn,
        {
            "g": g,
            "x": x,
            "mean_b": bcast(mean),
            "var_b": bcast(var),
            "scale_b": bcast(scale),
            "eps": np.array(eps, dtype=np.float32),
            "channel_axes": np.array([0, 2, 3], dtype=np.int64),
            "one": np.array(1.0, dtype=np.float32),
            "neg_half": np.array(-0.5, dtype=np.float32),
        },
        {
            "dx": x.shape,
            "dscale": (c,),
            "dbias": (c,),
            "dmean": (c,),
            "dvar": (c,),
        },
    )

    def forward(x, mean, var, scale, bias):
        xc = x - bcast(mean)
        inv = 1.0 / np.sqrt(bcast(var) + eps)
        return xc * inv * bcast(scale) + bcast(bias)

    def loss(**kw):
        return float(np.sum(forward(**kw) * g))

    base = {"x": x, "mean": mean, "var": var, "scale": scale, "bias": bias}
    step = 1e-3

    def fd_grad(param: str) -> np.ndarray:
        grad = np.zeros_like(base[param], dtype=np.float64)
        flat = grad.reshape(-1)
        for i in range(flat.size):
            plus = {k: v.copy() for k, v in base.items()}
            minus = {k: v.copy() for k, v in base.items()}
            plus[param].reshape(-1)[i] += step
            minus[param].reshape(-1)[i] -= step
            flat[i] = (loss(**plus) - loss(**minus)) / (2 * step)
        return grad

    names = ["x", "scale", "bias", "mean", "var"]
    for name, value in zip(names, analytic):
        fd = fd_grad(name)
        err = np.max(np.abs(value.astype(np.float64) - fd))
        rel = err / (np.max(np.abs(fd)) + 1e-6)
        if rel >= 1e-2:
            raise AssertionError(f"GradBatchNormalization.{name}: relative error {rel}")


def _fd_grad(loss, base: dict, param: str, step: float = 1e-3) -> np.ndarray:
    """Central finite-difference derivative of scalar ``loss(**base)`` with
    respect to every element of ``base[param]``, holding the rest fixed.
    The same closure ``_validate_grad_batch_normalization`` above builds
    inline, generalized so each elementwise validator below need not repeat
    it."""
    grad = np.zeros_like(base[param], dtype=np.float64)
    flat = grad.reshape(-1)
    for i in range(flat.size):
        plus = dict(base)
        minus = dict(base)
        plus[param] = base[param].copy()
        minus[param] = base[param].copy()
        plus[param].reshape(-1)[i] += step
        minus[param].reshape(-1)[i] -= step
        flat[i] = (loss(**plus) - loss(**minus)) / (2 * step)
    return grad


def _assert_close_to_fd(name: str, analytic: np.ndarray, fd: np.ndarray) -> None:
    err = np.max(np.abs(analytic.astype(np.float64) - fd))
    rel = err / (np.max(np.abs(fd)) + 1e-6)
    if rel >= 1e-2:
        raise AssertionError(f"{name}: relative error {rel}")


def _validate_grad_neg() -> None:
    """Closed-form, like ``GradAdd``: dx = -g exactly."""
    fn = GradNeg.to_function_proto()
    onnx.checker.check_function(fn)
    g = np.random.default_rng(0).standard_normal((5,)).astype(np.float32)
    (dx,) = _run_function(fn, {"g": g}, {"dx": g.shape})
    np.testing.assert_array_equal(dx, -g)


def _validate_grad_exp() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5,)).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    y = np.exp(x)
    fn = GradExp.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(fn, {"g": g, "y": y}, {"dx": x.shape})
    fd = _fd_grad(lambda x: float(np.sum(np.exp(x) * g)), {"x": x.astype(np.float64)}, "x")
    _assert_close_to_fd("GradExp", dx, fd)


def _validate_grad_sqrt() -> None:
    rng = np.random.default_rng(0)
    x = (np.abs(rng.standard_normal((5,))) + 0.1).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    y = np.sqrt(x)
    fn = GradSqrt.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(
        fn, {"g": g, "y": y, "half": np.array(0.5, dtype=np.float32)}, {"dx": x.shape}
    )
    fd = _fd_grad(lambda x: float(np.sum(np.sqrt(x) * g)), {"x": x.astype(np.float64)}, "x")
    _assert_close_to_fd("GradSqrt", dx, fd)


def _validate_grad_log() -> None:
    rng = np.random.default_rng(0)
    x = (np.abs(rng.standard_normal((5,))) + 0.1).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    fn = GradLog.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(fn, {"g": g, "x": x}, {"dx": x.shape})
    fd = _fd_grad(lambda x: float(np.sum(np.log(x) * g)), {"x": x.astype(np.float64)}, "x")
    _assert_close_to_fd("GradLog", dx, fd)


def _validate_grad_sigmoid() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5,)).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    y = 1.0 / (1.0 + np.exp(-x))
    fn = GradSigmoid.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(
        fn, {"g": g, "y": y, "one": np.array(1.0, dtype=np.float32)}, {"dx": x.shape}
    )
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    fd = _fd_grad(lambda x: float(np.sum(sigmoid(x) * g)), {"x": x.astype(np.float64)}, "x")
    _assert_close_to_fd("GradSigmoid", dx, fd)


def _validate_grad_tanh() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5,)).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    y = np.tanh(x)
    fn = GradTanh.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(
        fn, {"g": g, "y": y, "one": np.array(1.0, dtype=np.float32)}, {"dx": x.shape}
    )
    fd = _fd_grad(lambda x: float(np.sum(np.tanh(x) * g)), {"x": x.astype(np.float64)}, "x")
    _assert_close_to_fd("GradTanh", dx, fd)


def _validate_grad_erf() -> None:
    import math

    rng = np.random.default_rng(0)
    x = rng.standard_normal((5,)).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    c = 2.0 / np.sqrt(np.pi)
    erf = np.vectorize(math.erf)
    fn = GradErf.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(
        fn, {"g": g, "x": x, "c": np.array(c, dtype=np.float32)}, {"dx": x.shape}
    )
    fd = _fd_grad(
        lambda x: float(np.sum(erf(x) * g)), {"x": x.astype(np.float64)}, "x", step=1e-4
    )
    _assert_close_to_fd("GradErf", dx, fd)


def _validate_grad_relu() -> None:
    rng = np.random.default_rng(0)
    # Kept away from 0: a central difference straddling Relu's kink would
    # not agree with either one-sided subgradient.
    x = (np.sign(rng.standard_normal((5,))) * (np.abs(rng.standard_normal((5,))) + 0.2)).astype(
        np.float32
    )
    g = rng.standard_normal((5,)).astype(np.float32)
    fn = GradRelu.to_function_proto()
    onnx.checker.check_function(fn)
    (dx,) = _run_function(
        fn, {"g": g, "x": x, "zero": np.array(0.0, dtype=np.float32)}, {"dx": x.shape}
    )
    fd = _fd_grad(
        lambda x: float(np.sum(np.maximum(x, 0.0) * g)), {"x": x.astype(np.float64)}, "x"
    )
    _assert_close_to_fd("GradRelu", dx, fd)


def _validate_grad_mul() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5,)).astype(np.float32)
    b = rng.standard_normal((5,)).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    fn = GradMul.to_function_proto()
    onnx.checker.check_function(fn)
    da, db = _run_function(fn, {"g": g, "a": a, "b": b}, {"da": a.shape, "db": b.shape})
    base = {"a": a.astype(np.float64), "b": b.astype(np.float64)}
    loss = lambda a, b: float(np.sum(a * b * g))
    _assert_close_to_fd("GradMul.da", da, _fd_grad(loss, base, "a"))
    _assert_close_to_fd("GradMul.db", db, _fd_grad(loss, base, "b"))


def _validate_grad_div() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5,)).astype(np.float32)
    b = (np.abs(rng.standard_normal((5,))) + 0.5).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    y = a / b
    fn = GradDiv.to_function_proto()
    onnx.checker.check_function(fn)
    da, db = _run_function(
        fn, {"g": g, "a": a, "b": b, "y": y}, {"da": a.shape, "db": b.shape}
    )
    base = {"a": a.astype(np.float64), "b": b.astype(np.float64)}
    loss = lambda a, b: float(np.sum((a / b) * g))
    _assert_close_to_fd("GradDiv.da", da, _fd_grad(loss, base, "a"))
    _assert_close_to_fd("GradDiv.db", db, _fd_grad(loss, base, "b"))


# (python identifier, C++ identifier, onnxscript function). Both languages'
# generated files are produced from this single list, so they cannot drift
# from each other -- the whole point of checking in ONNX function *text*
# rather than hand-porting the graph construction twice.
_ENTRIES = [
    ("GRAD_ADD", "kGradAddTemplate", GradAdd),
    (
        "GRAD_BATCH_NORMALIZATION",
        "kGradBatchNormalizationTemplate",
        GradBatchNormalization,
    ),
    ("GRAD_NEG", "kGradNegTemplate", GradNeg),
    ("GRAD_EXP", "kGradExpTemplate", GradExp),
    ("GRAD_SQRT", "kGradSqrtTemplate", GradSqrt),
    ("GRAD_LOG", "kGradLogTemplate", GradLog),
    ("GRAD_SIGMOID", "kGradSigmoidTemplate", GradSigmoid),
    ("GRAD_TANH", "kGradTanhTemplate", GradTanh),
    ("GRAD_ERF", "kGradErfTemplate", GradErf),
    ("GRAD_RELU", "kGradReluTemplate", GradRelu),
    ("GRAD_MUL", "kGradMulTemplate", GradMul),
    ("GRAD_DIV", "kGradDivTemplate", GradDiv),
]


def _python_module(entries) -> str:
    out = []
    out.append("# SPDX-License-Identifier: Apache-2.0")
    out.append("#")
    out.append(
        "# GENERATED FILE -- do not edit by hand. Produced by\n"
        "#   python3 scripts/codegen/generate_grad_templates.py\n"
        "# from the onnxscript function definitions in that script; see its\n"
        "# module docstring for what this is and why it takes no ONNX-level\n"
        "# attributes."
    )
    out.append('"""ONNX function text for onnxsim.graph_grad\'s templated')
    out.append("gradient rules -- parsed back via onnx.parser.parse_function,")
    out.append('never onnxscript itself, which this module does not import."""')
    out.append("")
    out.append("from __future__ import annotations")
    out.append("")
    for py_ident, _cpp_ident, text in entries:
        out.append(f'{py_ident} = """{text}"""')
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _cpp_header(entries) -> str:
    """The same checked-in ONNX text, as C++ string-literal constants --
    onnxsim/graph_grad.cpp reads these with onnx::OnnxParser::Parse (the
    full onnx.defs.parser text format, not the per-statement-line format
    generate_moe_function_templates.py's own header needs for
    onnx::FunctionBuilder): the checked-in text is identical between the two
    languages, only the wrapper differs."""
    out = []
    out.append("// SPDX-License-Identifier: Apache-2.0")
    out.append("//")
    out.append(
        "// GENERATED FILE -- do not edit by hand. Produced by\n"
        "//   python3 scripts/codegen/generate_grad_templates.py\n"
        "// from the onnxscript function definitions in that script; see its\n"
        "// module docstring for what this is and why it takes no ONNX-level\n"
        "// attributes. graph_grad_templates_gen.py is the same text for the\n"
        "// Python side -- both are produced from the same entries so they\n"
        "// cannot drift from each other."
    )
    out.append("#ifndef ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_")
    out.append("#define ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_")
    out.append("")
    out.append(
        "// No enclosing namespace -- graph_grad.cpp, this header's only"
        " consumer, has\n// none either (it mirrors graph_grad.py's flat"
        " module directly)."
    )
    for _py_ident, cpp_ident, text in entries:
        out.append(f'constexpr const char* {cpp_ident} = R"GRAD_TPL({text})GRAD_TPL";')
        out.append("")
    out.append("#endif  // ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_")
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> None:
    _validate_grad_add()
    _validate_grad_batch_normalization()
    _validate_grad_neg()
    _validate_grad_exp()
    _validate_grad_sqrt()
    _validate_grad_log()
    _validate_grad_sigmoid()
    _validate_grad_tanh()
    _validate_grad_erf()
    _validate_grad_relu()
    _validate_grad_mul()
    _validate_grad_div()

    entries = [
        (py_ident, cpp_ident, onnx.printer.to_text(fn.to_function_proto()))
        for py_ident, cpp_ident, fn in _ENTRIES
    ]

    argv = sys.argv[1:]
    if not argv:
        sys.stdout.write(_python_module(entries))
        return
    with open(argv[0], "w") as f:
        f.write(_python_module(entries))
    if len(argv) > 1:
        with open(argv[1], "w") as f:
            f.write(_cpp_header(entries))


if __name__ == "__main__":
    main()
