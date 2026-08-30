#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generates onnxsim/contrib_schemas_moe_templates.gen.h from onnxscript.

contrib_schemas.cpp's BuildMoEFunctionBody decomposes ONNX Runtime's
`com.microsoft.MoE` into standard ONNX ops by unrolling one expert block per
expert at function-build time (see that file for why: an ONNX function body
is a fixed node list, so "run this block once per expert" can only be
realized by literally emitting that many copies). The two pieces of that
body that don't depend on which specific node is being decomposed -- the
routing head (softmax -> top-k -> optional renormalize -> dense gate) and a
single expert's FFN block (Gather-by-index -> Squeeze -> Gemm -> activation
-> Gemm -> slice-the-gate-column -> weight -> accumulate) -- are authored
here as real onnxscript functions instead of hand-typed ONNX text in C++.
onnxscript type-checks each fragment against the exact opset (18) the
generated function body targets, which is what would have caught bugs like
"ReduceSum's axes moved from an attribute to an input at opset 13" or a
Gemm/Squeeze signature mismatch automatically, instead of relying on a
human noticing.

`k` (top-k count), `hidden_size`, and the per-expert index (and its +1) are
only known once BuildMoEFunctionBody sees the actual MoE node being
decomposed -- they can't be literal Python ints when this script runs. Each
is instead written into the onnxscript source as one of the sentinel
integers below, which this script then replaces with a `{{TOKEN}}` text
placeholder in the generated header; contrib_schemas.cpp does a plain
string substitution of those tokens with the real values at function-build
time. That substitution is the one piece of "string handling" that
genuinely can't move into this script: which *values* to plug in isn't
known until a real model is being simplified, only the surrounding text is
fixed ahead of time.

Run this script whenever the generated templates need to change:
    python3 scripts/codegen/generate_moe_function_templates.py \
        onnxsim/contrib_schemas_moe_templates.gen.h
(or with no argument, to print the generated header to stdout instead of
writing it). The output is checked into the repository rather than
generated as part of every C++ build: onnxscript is not otherwise a build
dependency of onnxsim's C++ extension, and this repo builds across many CI
matrices (Windows/macOS/ARM cross-compiles, WASM/Pyodide, s390x under
qemu) that would all need to gain it just to compile a header that only
changes when this script itself changes.
`cmake --build . --target regenerate_moe_templates` re-runs this script in
place for local iteration when onnxscript is installed; ordinary builds
never invoke it.
"""

import linecache
import re
import sys

import onnx
from onnx import parser as onnx_parser
from onnxscript import FLOAT, INT64, script
from onnxscript import opset18 as op

# Sentinel integers substituted for the values BuildMoEFunctionBody only
# knows at function-build time. Chosen to be distinctive enough that a
# plain string replace can't collide with any other literal (-1, 0, 1, ...)
# the generated bodies use.
_HIDDEN_SIZE_SENTINEL = 823001
_K_SENTINEL = 823002
_E_SENTINEL = 823003
_E1_SENTINEL = 823004

_PLACEHOLDERS = {
    _HIDDEN_SIZE_SENTINEL: "{{HIDDEN_SIZE}}",
    _K_SENTINEL: "{{K}}",
    _E_SENTINEL: "{{E}}",
    _E1_SENTINEL: "{{E1}}",
}

_NODE_INDEX_RE = re.compile(r"^\s*\[n\d+\]\s*")
# onnx.printer.to_text prints a whole-number `value_float` without a decimal
# point (e.g. "value_float: float = 0"), which onnx's own text-format parser
# then round-trips as an INT-typed value -- failing the checker with "type
# field and data field mismatch" despite the explicit `: float` annotation.
# The hand-written text this replaces always spelled these as "0.0"/"1.0"/
# "2.0" for the same reason; restore that here for every whole-number float.
_WHOLE_NUMBER_FLOAT_RE = re.compile(r"(value_float: float = -?\d+)(?!\.)\b")


def _extract_body(func_text: str) -> str:
    """Strips onnx.printer.to_text's function header/footer and [nN] tags,
    leaving the plain 'name = Op<attrs>(inputs)' statement lines
    FunctionBuilder::Add() expects."""
    start = func_text.index("{")
    end = func_text.rindex("}")
    lines = []
    for line in func_text[start + 1 : end].splitlines():
        line = _NODE_INDEX_RE.sub("", line)
        line = _WHOLE_NUMBER_FLOAT_RE.sub(r"\1.0", line)
        if line.strip():
            lines.append(line)
    return "\n".join(lines) + "\n"


def _to_template(onnx_fn) -> str:
    text = _extract_body(onnx.printer.to_text(onnx_fn.to_function_proto()))
    for sentinel, token in _PLACEHOLDERS.items():
        text = text.replace(str(sentinel), token)
    return text


def _uniquify_expert_locals(onnx_fn, text: str) -> str:
    """Suffixes every value this expert-block template defines locally with
    `{{E}}`, and rewrites its `AccPrev`/`AccNext` formal parameters to
    `Acc{{E}}`/`Acc{{E1}}`.

    A single expert-block template is spliced into the function body once
    per expert (BuildMoEFunctionBody's whole reason for existing: an ONNX
    function body is a fixed node list, so "run this per expert" means
    literally emitting one copy per expert) -- so unlike the routing head,
    which appears exactly once, every name this template introduces (W1Idx,
    H1, A1, Out, ...) would otherwise collide with the same names from every
    *other* expert's copy in the same body. The original hand-written C++
    this replaces suffixed every such name with the expert index by hand for
    exactly this reason; this reconstructs the same scheme mechanically from
    the compiled FunctionProto's own node outputs, so it can't drift out of
    sync with whatever names the onnxscript source above happens to use.
    """
    proto = onnx_fn.to_function_proto()
    local_names = {out for node in proto.node for out in node.output}
    # AccPrev (a formal input, chained from the previous expert's AccNext)
    # and AccNext (this block's own return value) get the special Acc{{E}} /
    # Acc{{E1}} chain-naming scheme below instead of a generic per-expert
    # suffix, matching the "Acc0, Acc1, ..., AccN" naming the unrolling loop
    # in BuildMoEFunctionBody drives.
    local_names.discard("AccNext")
    for name in sorted(local_names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", f"{name}{{{{E}}}}", text)
    text = re.sub(r"\bAccPrev\b", "Acc{{E}}", text)
    text = re.sub(r"\bAccNext\b", "Acc{{E1}}", text)
    return text


def _check_no_stray_sentinels(name: str, text: str) -> None:
    for sentinel in _PLACEHOLDERS:
        if str(sentinel) in text:
            raise AssertionError(
                f"{name}: sentinel {sentinel} survived placeholder substitution "
                "-- it must have been split across two tokens (e.g. concatenated "
                "with another literal); pick a different sentinel value."
            )


# --- Routing head: softmax -> top-k -> optional renormalize -> dense gate.
# Matches BuildMoEFunctionBody's prologue exactly (see contrib_schemas.cpp):
# router_probs is raw routing *logits* despite its name, so Softmax always
# runs; `normalize` only controls whether the selected top-k weights are
# renormalized to sum to 1, which is a structural difference (extra
# ReduceSum/Div nodes), not just a different literal -- hence two separate
# onnxscript functions rather than one parameterized by a bool.


@script()
def RoutingHeadNormalize(
    input: FLOAT["..."],
    router_probs: FLOAT["N", "E"],  # noqa: F821
):
    FlatShape = op.Constant(value_ints=[-1, _HIDDEN_SIZE_SENTINEL])
    FlatInput = op.Reshape(input, FlatShape)
    InputShape = op.Shape(input)
    Probs = op.Softmax(router_probs, axis=-1)
    TopKConst = op.Constant(value_ints=[_K_SENTINEL])
    TopVals, TopIdx = op.TopK(Probs, TopKConst, axis=-1, largest=1)
    ReduceAxes = op.Constant(value_ints=[-1])
    Denom = op.ReduceSum(TopVals, ReduceAxes, keepdims=1)
    TopValsNorm = op.Div(TopVals, Denom)
    ZeroT = op.Constant(value_float=0.0)
    ZeroCast = op.CastLike(ZeroT, input)
    GateZeros = op.Mul(Probs, ZeroCast)
    Gates = op.ScatterElements(GateZeros, TopIdx, TopValsNorm, axis=-1)
    GAxis = op.Constant(value_ints=[1])
    SqueezeAxis = op.Constant(value_ints=[0])
    Acc0 = op.Mul(FlatInput, ZeroCast)
    return FlatInput, InputShape, Gates, GAxis, SqueezeAxis, Acc0


@script()
def RoutingHeadNoNormalize(
    input: FLOAT["..."],
    router_probs: FLOAT["N", "E"],  # noqa: F821
):
    FlatShape = op.Constant(value_ints=[-1, _HIDDEN_SIZE_SENTINEL])
    FlatInput = op.Reshape(input, FlatShape)
    InputShape = op.Shape(input)
    Probs = op.Softmax(router_probs, axis=-1)
    TopKConst = op.Constant(value_ints=[_K_SENTINEL])
    TopVals, TopIdx = op.TopK(Probs, TopKConst, axis=-1, largest=1)
    TopValsNorm = op.Identity(TopVals)
    ZeroT = op.Constant(value_float=0.0)
    ZeroCast = op.CastLike(ZeroT, input)
    GateZeros = op.Mul(Probs, ZeroCast)
    Gates = op.ScatterElements(GateZeros, TopIdx, TopValsNorm, axis=-1)
    GAxis = op.Constant(value_ints=[1])
    SqueezeAxis = op.Constant(value_ints=[0])
    Acc0 = op.Mul(FlatInput, ZeroCast)
    return FlatInput, InputShape, Gates, GAxis, SqueezeAxis, Acc0


# --- Expert block: Gather-by-index -> Squeeze -> Gemm -> activation -> Gemm
# -> slice-the-gate-column -> weight -> accumulate. `expert` is baked in as
# a literal index at function-*build* time -- the C++ loop that emits one
# of these blocks per expert is what actually unrolls the op; nothing here
# is a runtime/ONNX-level loop. 16 variants: 4 activations x with/without
# fc1 bias x with/without fc2 bias, since bias presence changes which nodes
# appear (an extra Gather+Squeeze feeding Gemm's optional C input), not
# just a literal value.


def _activation_body(activation: str) -> str:
    """Python source (as `@script()` will trace it) computing A1 from H1.

    onnxscript's `@script()` decorator translates the AST of the function
    it's applied to directly into ONNX ops; it can't call out to a plain
    Python helper to inline a sub-sequence of ops (only to another
    `@script()` function, as a real nested function *call* in the graph,
    which isn't what's wanted here -- the current C++ output inlines
    everything into one flat per-expert block). So each activation's ops are
    spliced into the surrounding function's source text below instead of
    being factored into a Python function.
    """
    if activation == "relu":
        return "  A1 = op.Relu(H1)\n"
    if activation == "identity":
        return "  A1 = op.Identity(H1)\n"
    if activation == "silu":
        return "  Sig = op.Sigmoid(H1)\n  A1 = op.Mul(H1, Sig)\n"
    if activation == "gelu":
        # Exact, erf-based -- the same decomposition ONNX's own Gelu op uses
        # for its default (non-"tanh") approximate mode.
        return (
            "  Half = op.Constant(value_float=0.5)\n"
            "  HalfCast = op.CastLike(Half, H1)\n"
            "  One = op.Constant(value_float=1.0)\n"
            "  OneCast = op.CastLike(One, H1)\n"
            "  Two = op.Constant(value_float=2.0)\n"
            "  TwoCast = op.CastLike(Two, H1)\n"
            "  SqrtTwo = op.Sqrt(TwoCast)\n"
            "  XSqrt = op.Div(H1, SqrtTwo)\n"
            "  ErfXSqrt = op.Erf(XSqrt)\n"
            "  Phi = op.Sum(OneCast, ErfXSqrt)\n"
            "  MultX = op.Mul(HalfCast, H1)\n"
            "  A1 = op.Mul(MultX, Phi)\n"
        )
    raise ValueError(activation)


def _make_expert_block(activation: str, has_fc1_bias: bool, has_fc2_bias: bool):
    """Builds one (activation, fc1_bias, fc2_bias) expert-block variant.

    Assembles the `@script()`-decorated function's Python *source text* for
    this combination (signature and fc1/fc2-bias handling vary structurally
    with bias presence, same as BuildMoEFunctionBody's own C++ branches do
    today) and compiles it with `exec`, rather than writing out 16 near-
    identical function defs longhand -- the generated ONNX text is still
    fully concrete per variant either way.
    """
    params = [
        'FlatInput: FLOAT["N", "H"]',
        'Gates: FLOAT["N", "E"]',
        'fc1_experts_weights: FLOAT["E", "I", "H"]',
    ]
    if has_fc1_bias:
        params.append('fc1_experts_bias: FLOAT["E", "I"]')
    params.append('fc2_experts_weights: FLOAT["E", "H", "I"]')
    if has_fc2_bias:
        params.append('fc2_experts_bias: FLOAT["E", "H"]')
    params += [
        'SqueezeAxis: INT64["1"]',
        'GAxis: INT64["1"]',
        'AccPrev: FLOAT["N", "H"]',
    ]

    lines = []
    lines.append("W1Idx = op.Constant(value_ints=[_E_SENTINEL])")
    lines.append("W1_3d = op.Gather(fc1_experts_weights, W1Idx, axis=0)")
    lines.append("W1 = op.Squeeze(W1_3d, SqueezeAxis)")
    lines.append("W2Idx = op.Constant(value_ints=[_E_SENTINEL])")
    lines.append("W2_3d = op.Gather(fc2_experts_weights, W2Idx, axis=0)")
    lines.append("W2 = op.Squeeze(W2_3d, SqueezeAxis)")
    if has_fc1_bias:
        lines.append("B1_2d = op.Gather(fc1_experts_bias, W1Idx, axis=0)")
        lines.append("B1 = op.Squeeze(B1_2d, SqueezeAxis)")
        lines.append("H1 = op.Gemm(FlatInput, W1, B1, transB=1)")
    else:
        lines.append("H1 = op.Gemm(FlatInput, W1, transB=1)")
    body = "  " + "\n  ".join(lines) + "\n" + _activation_body(activation)
    tail = []
    if has_fc2_bias:
        tail.append("B2_2d = op.Gather(fc2_experts_bias, W1Idx, axis=0)")
        tail.append("B2 = op.Squeeze(B2_2d, SqueezeAxis)")
        tail.append("Out = op.Gemm(A1, W2, B2, transB=1)")
    else:
        tail.append("Out = op.Gemm(A1, W2, transB=1)")
    tail.append("GEnd = op.Constant(value_ints=[_E1_SENTINEL])")
    tail.append("GateCol = op.Slice(Gates, W1Idx, GEnd, GAxis)")
    tail.append("Weighted = op.Mul(Out, GateCol)")
    tail.append("AccNext = op.Add(AccPrev, Weighted)")
    body += "  " + "\n  ".join(tail) + "\n  return AccNext\n"

    source = (
        f'@script()\ndef ExpertBlock({", ".join(params)}) -> FLOAT["N", "H"]:\n{body}'
    )
    namespace = {
        "script": script,
        "FLOAT": FLOAT,
        "INT64": INT64,
        "op": op,
        "_E_SENTINEL": _E_SENTINEL,
        "_E1_SENTINEL": _E1_SENTINEL,
        # onnxscript's `@script()` decorator also calls `inspect.getmodule()`
        # on the function it wraps to find the globals dict to compile
        # against; that resolves the exec'd function back to *this* running
        # script's own module (found via `sys.modules[__name__]`) rather than
        # `None`, which is what it would otherwise get without a `__name__`
        # binding matching a real entry in `sys.modules`.
        "__name__": __name__,
    }
    # `inspect.getsource()`, called by the same decorator, needs the source
    # registered in `linecache` -- plain `exec()` of a dynamically compiled
    # string doesn't do that on its own, so register it explicitly (the
    # standard workaround for decorators that inspect their own source).
    filename = f"<expert_block_{activation}_fc1{has_fc1_bias}_fc2{has_fc2_bias}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace["ExpertBlock"]


_ACTIVATIONS = ["relu", "identity", "silu", "gelu"]


def _cpp_ident(activation: str, has_fc1_bias: bool, has_fc2_bias: bool) -> str:
    cap = activation[0].upper() + activation[1:]
    b1 = "Bias" if has_fc1_bias else "NoBias"
    b2 = "Bias" if has_fc2_bias else "NoBias"
    return f"kMoEExpertBlock{cap}Fc1{b1}Fc2{b2}"


def _validate_expert_block_template(
    name: str, template: str, has_fc1_bias: bool, has_fc2_bias: bool
) -> None:
    """Round-trips one expert-block template through onnx's own text-format
    parser and checker with concrete placeholder values, the same parser
    FunctionBuilder::Add() drives from C++ -- catches a malformed fragment
    here instead of at onnxsim's next C++ build/test cycle."""
    text = template.replace("{{E}}", "0").replace("{{E1}}", "1")
    full_text = f"""
<
  ir_version: 10,
  opset_import: ["" : 18]
>
agraph (float[4,6] FlatInput, float[4,3] Gates,
        float[3,8,6] fc1_experts_weights,
        {"float[3,8] fc1_experts_bias," if has_fc1_bias else ""}
        float[3,6,8] fc2_experts_weights,
        {"float[3,6] fc2_experts_bias," if has_fc2_bias else ""}
        int64[1] SqueezeAxis, int64[1] GAxis, float[4,6] Acc0)
    => (float[4,6] Acc1)
{{
{text}
}}
"""
    model = onnx_parser.parse_model(full_text)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, check_type=True)


def _validate_expert_block_chain(
    name: str, template: str, has_fc1_bias: bool, has_fc2_bias: bool
) -> None:
    """Splices two copies of `template` back to back (expert 0 then expert
    1, chained AccPrev -> AccNext -> AccPrev) and checks the result is still
    a single valid function body -- this is the regression check for the bug
    this generator's `_uniquify_expert_locals` step exists to prevent: every
    value a template defines locally (W1Idx, H1, A1, Out, ...) must be
    unique per expert-copy, or two copies in the same function body collide
    on the same name and the checker rejects it (or worse, silently
    shadows). A single-copy check alone can't see this, since a name only
    collides against a *different* copy of the same template.
    """
    block0 = template.replace("{{E}}", "0").replace("{{E1}}", "1")
    block1 = template.replace("{{E}}", "1").replace("{{E1}}", "2")
    fc1_bias_input = ", float[3,8] fc1_experts_bias" if has_fc1_bias else ""
    fc2_bias_input = ", float[3,6] fc2_experts_bias" if has_fc2_bias else ""
    full_text = f"""
<
  ir_version: 10,
  opset_import: ["" : 18]
>
agraph (float[4,6] FlatInput, float[4,3] Gates,
        float[3,8,6] fc1_experts_weights{fc1_bias_input},
        float[3,6,8] fc2_experts_weights{fc2_bias_input},
        int64[1] SqueezeAxis, int64[1] GAxis, float[4,6] Acc0)
    => (float[4,6] Acc2)
{{
{block0}
{block1}
}}
"""
    model = onnx_parser.parse_model(full_text)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, check_type=True)


def _validate_routing_head_template(name: str, template: str) -> None:
    # router_probs is always 2D (num_tokens, num_experts), matching the real
    # MoE schema, even when `input` is the 3D (batch, seq, hidden) form --
    # num_tokens is the flattened batch*seq count (12 = 4*3 here, matching
    # `input`'s leading dims once FlatInput reshapes it to (12, hidden_size)).
    text = template.replace("{{HIDDEN_SIZE}}", "6").replace("{{K}}", "2")
    full_text = f"""
<
  ir_version: 10,
  opset_import: ["" : 18]
>
agraph (float[4,3,6] input, float[12,5] router_probs)
    => (float FlatInput, int64 InputShape, float Gates,
        int64 GAxis, int64 SqueezeAxis, float Acc0)
{{
{text}
}}
"""
    model = onnx_parser.parse_model(full_text)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, check_type=True)


def main() -> None:
    entries = []

    routing_normalize = _to_template(RoutingHeadNormalize)
    routing_no_normalize = _to_template(RoutingHeadNoNormalize)
    _check_no_stray_sentinels("kMoERoutingHeadNormalize", routing_normalize)
    _check_no_stray_sentinels("kMoERoutingHeadNoNormalize", routing_no_normalize)
    _validate_routing_head_template("kMoERoutingHeadNormalize", routing_normalize)
    _validate_routing_head_template("kMoERoutingHeadNoNormalize", routing_no_normalize)
    entries.append(("kMoERoutingHeadNormalize", routing_normalize))
    entries.append(("kMoERoutingHeadNoNormalize", routing_no_normalize))

    for activation in _ACTIVATIONS:
        for has_fc1_bias in (False, True):
            for has_fc2_bias in (False, True):
                fn = _make_expert_block(activation, has_fc1_bias, has_fc2_bias)
                ident = _cpp_ident(activation, has_fc1_bias, has_fc2_bias)
                template = _uniquify_expert_locals(fn, _to_template(fn))
                _check_no_stray_sentinels(ident, template)
                _validate_expert_block_template(
                    ident, template, has_fc1_bias, has_fc2_bias
                )
                _validate_expert_block_chain(
                    ident, template, has_fc1_bias, has_fc2_bias
                )
                entries.append((ident, template))

    out = []
    out.append("// SPDX-License-Identifier: Apache-2.0")
    out.append("//")
    out.append(
        "// GENERATED FILE -- do not edit by hand. Produced by\n"
        "//   python3 scripts/codegen/generate_moe_function_templates.py\n"
        "// from the onnxscript function definitions in that script; see its\n"
        "// module docstring for why this is checked in rather than generated\n"
        "// on every build."
    )
    out.append("#ifndef ONNXSIM_CONTRIB_SCHEMAS_MOE_TEMPLATES_GEN_H_")
    out.append("#define ONNXSIM_CONTRIB_SCHEMAS_MOE_TEMPLATES_GEN_H_")
    out.append("")
    out.append("namespace onnxsim {")
    out.append("")
    for ident, template in entries:
        out.append(f'constexpr const char* {ident} = R"MOE_TPL(')
        out.append(template.rstrip("\n"))
        out.append(')MOE_TPL";')
        out.append("")
    out.append("}  // namespace onnxsim")
    out.append("")
    out.append("#endif  // ONNXSIM_CONTRIB_SCHEMAS_MOE_TEMPLATES_GEN_H_")
    text = "\n".join(out) + "\n"

    argv = sys.argv[1:]
    if argv:
        # An explicit output path, so the CMake `regenerate_moe_templates`
        # dev target can invoke this directly without relying on a shell
        # for `>` redirection (not portable to every CMake generator).
        with open(argv[0], "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
