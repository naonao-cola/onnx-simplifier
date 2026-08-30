#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generates onnxsim/contrib_schemas_moe_templates.gen.h from onnxscript.

contrib_schemas.cpp attaches ONNX Runtime's `com.microsoft.MoE` schema to one
of the fixed FunctionProtos generated here, chosen purely by which optional
inputs (fc1_experts_bias, fc2_experts_bias) the calling node actually has --
that is genuinely unavoidable context-dependence: ONNX's function-inlining
requires a function call's input count to match its formal parameter list
exactly, so a function referencing an optional input directly cannot be
reused for a call site that omits it (confirmed empirically -- there is no
Optional-wrapping/If trick that fixes this, since the *node list itself*,
not just a runtime value, would need to differ). See BuildMoEFunctionBody's
own comment in contrib_schemas.cpp for the full picture of what remains
context-dependent and why.

Beyond that unavoidable bias-presence dispatch, every one of the 24 fixed
functions built here (4 activations x fc1_bias x fc2_bias, plus 8 more for
activation="silu" x fc1_bias x fc2_bias x fc3_bias -- see below) is fully
generic at the ONNX level, using mechanisms a per-instance C++ codegen pass
used to paper over:

  - num_experts, unknown until a real node is inspected, is now a genuine
    ONNX `Loop` trip count (`Shape(fc1_experts_weights)[0]`) instead of a
    literal unrolled N times per node.
  - k, normalize_routing_weights, and use_sparse_mixer are forwarded from
    the calling node's own attributes via `ref_attr_name` (see
    `_bind_attribute` below) instead of being read in C++ and baked in as
    literal text. onnxscript has no direct syntax for `ref_attr_name`, so
    each is authored as a `Constant` holding a *sentinel* int (never a
    value the real math would produce) and rebound to `ref_attr_name` by
    editing the compiled FunctionProto afterwards.
  - use_sparse_mixer's choice of routing algorithm, and
    normalize_routing_weights' renormalization, are both genuine ONNX `If`
    branches on that forwarded value (onnxscript compiles a Python `if` on
    a traced graph value -- as opposed to a Python-level bool -- straight
    to an ONNX `If` node with real then/else subgraphs), not a per-instance
    choice of which template text to emit.

activation_type stays a build-time (per-function) choice rather than a
runtime branch: it is a *string* attribute, and ONNX's string-typed `Equal`
needs opset 19 (this targets opset 18, matching every op already in use);
more importantly it selects genuinely different math/ops, the same kind of
structural difference bias-presence forces above. See BuildMoEFunctionBody's
comment for why swiglu itself is not one of the 4 generated here yet.

use_sparse_mixer's own routing math (in `_routing_lines`) is transcribed
from onnxruntime's actual CUDA kernel
(onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu, `sparse_mixer_top2`) --
there is no CPU kernel implementing it at all, so unlike everything else
generated here it cannot be numerically checked against a running ONNX
Runtime session. It was instead cross-checked against an independent numpy
transliteration of the same CUDA source (see the PR description for the
comparison) before being transcribed a second time, here, into ONNX ops.

The 8 additional activation="silu" x fc1_bias x fc2_bias x fc3_bias
functions implement fc3 -- a real, shipping convention: onnxruntime-genai's
model builder (onnxruntime-genai/src/python/py/models/builders/phi.py,
`Phi3MoELongRoPEModel`) exports Phi-3.5-MoE-style checkpoints this way, and
it is the standard Mixtral gated-MLP shape (separate gate/up/down
projections) generally. `fc2(silu(fc1(x)) * fc3(x))` is transcribed
straight from onnxruntime's own CUDA kernel comment
(onnxruntime/contrib_ops/cuda/moe/moe.cc: "Mixtral case: SiLU activation
with separate FC3 ... Kernel supports SwiGLU which is Linear * SiLU(Gate)
... map Mixtral to SwiGLU by packing weights as [FC3, FC1] (Linear, Gate)"),
not guessed from the schema doc comments alone. Like use_sparse_mixer, this
has no CPU kernel to check against -- onnxruntime's CPU MoE kernel
(onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc) rejects fc3 unconditionally
("FC3 is not implemented for CPU MoE"), for any activation -- so this is
disclosed as the same kind of validation gap as use_sparse_mixer rather than
claimed as ORT-session-verified. It is also why fc3 is only ever generated
for activation="silu": the CUDA kernel's own weight-repacking is gated on
`activation_type_ == ActivationType::Silu` specifically, so relu/identity/
gelu + fc3 has no ORT-defined behavior at all (real ORT would silently
misinterpret such a node's weights, not just skip it) and stays declined in
BuildMoEFunctionBody rather than guessing a formula for it.

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
from onnxscript import FLOAT, script
from onnxscript import opset18 as op

# Sentinel ints standing in for the values that only become known through
# `ref_attr_name` forwarding once a real calling node exists -- see
# `_bind_attribute`. Chosen far outside any literal this code otherwise uses
# (axes, a leading -1 reshape dim, small loop indices, ...).
_K_SENTINEL = 823002
_NORMALIZE_SENTINEL = 823010
_SPARSE_MIXER_SENTINEL = 823011

_NODE_INDEX_RE = re.compile(r"^\s*\[n\d+\]\s*")
_WHOLE_NUMBER_FLOAT_RE = re.compile(r"(value_float: float = -?\d+)(?!\.)\b")


def _bind_attribute(proto, sentinel: int, attr_name: str) -> None:
    """Rewrites the `Constant` node holding `sentinel` (as `value_int`) into
    one that instead forwards the function's own `attr_name` attribute via
    `ref_attr_name`, and declares that attribute on the function itself.

    onnxscript's Python surface has no way to spell `ref_attr_name` directly
    (attribute-valued function parameters can only be forwarded into an op's
    *own* attribute of the same kind, not turned into a tensor via `Constant`
    -- confirmed empirically), so this post-processes the compiled proto
    instead: verified end-to-end (a `Constant`+`Cast`+`If` built this way,
    called through real ONNX Runtime with the bound attribute set to 0 and 1,
    actually branches both ways) before being relied on here.
    """
    found = 0
    for node in proto.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value_int" and attr.i == sentinel:
                attr.ClearField("i")
                attr.ref_attr_name = attr_name
                found += 1
    if found != 1:
        raise AssertionError(
            f"expected exactly one Constant(value_int={sentinel}) standing in "
            f"for '{attr_name}', found {found}"
        )
    proto.attribute.append(attr_name)


def _extract_body(func_text: str) -> str:
    """Strips onnx.printer.to_text's function header/footer and [nN] tags,
    leaving the plain 'name = Op<attrs>(inputs)' statement lines
    FunctionBuilder::Add() expects. Nested graphs (Loop/If bodies) keep their
    own internal [nN] tags too, so this must strip line-by-line rather than
    only at top level."""
    start = func_text.index("{")
    end = func_text.rindex("}")
    lines = []
    for line in func_text[start + 1 : end].splitlines():
        line = _NODE_INDEX_RE.sub("", line)
        line = _WHOLE_NUMBER_FLOAT_RE.sub(r"\1.0", line)
        if line.strip():
            lines.append(line)
    return "\n".join(lines) + "\n"


def _cpp_ident(
    activation: str,
    has_fc1_bias: bool,
    has_fc2_bias: bool,
    has_fc3_bias: bool | None = None,
) -> str:
    cap = activation[0].upper() + activation[1:]
    b1 = "Bias" if has_fc1_bias else "NoBias"
    b2 = "Bias" if has_fc2_bias else "NoBias"
    if has_fc3_bias is None:
        return f"kMoEFunction{cap}Fc1{b1}Fc2{b2}"
    b3 = "Bias" if has_fc3_bias else "NoBias"
    return f"kMoEFunction{cap}Fc1{b1}Fc2{b2}Fc3{b3}"


# --- Per-activation FC1-output -> FC2-input transform. relu/identity/silu
# match ApplyActivation in onnxruntime/contrib_ops/cpu/moe/moe_utils.cc
# exactly (no approximation involved for any of the three). gelu matches
# that same file's *tanh approximation* -- confirmed empirically against a
# real onnxruntime session (isolating the activation with a 1x1 identity-
# weight MoE node): the exact erf-based formula this decomposition used
# before differs from onnxruntime's actual output by up to ~4e-4 absolute
# (~5% relative near the inflection points), which is *not* how
# onnxruntime's own CPU kernel computes gelu here despite `Gelu`
# (ONNX's own standard op) defaulting to the exact form.
def _activation_lines(activation: str) -> str:
    if activation == "relu":
        return "a1 = op.Relu(h1)\n"
    if activation == "identity":
        return "a1 = op.Identity(h1)\n"
    if activation == "silu":
        return "sig = op.Sigmoid(h1)\na1 = op.Mul(h1, sig)\n"
    if activation == "gelu":
        return (
            "c0 = op.Constant(value_float=0.5)\n"
            "c0_cast = op.CastLike(c0, h1)\n"
            "c1 = op.Constant(value_float=1.0)\n"
            "c1_cast = op.CastLike(c1, h1)\n"
            "c_coeff = op.Constant(value_float=0.7978845608)\n"
            "c_coeff_cast = op.CastLike(c_coeff, h1)\n"
            "c_cube = op.Constant(value_float=0.044715)\n"
            "c_cube_cast = op.CastLike(c_cube, h1)\n"
            "h1_cubed = op.Mul(op.Mul(h1, h1), h1)\n"
            "inner = op.Add(h1, op.Mul(c_cube_cast, h1_cubed))\n"
            "tanh_arg = op.Mul(c_coeff_cast, inner)\n"
            "tanh_out = op.Tanh(tanh_arg)\n"
            "one_plus_tanh = op.Add(c1_cast, tanh_out)\n"
            "half_h1 = op.Mul(c0_cast, h1)\n"
            "a1 = op.Mul(half_h1, one_plus_tanh)\n"
        )
    raise ValueError(activation)


# --- Routing: dense (mostly-zero) per-token gate row, shape (num_tokens,
# num_experts). Two algorithms, chosen by a genuine runtime `If` on the
# forwarded `use_sparse_mixer` attribute -- not a per-instance choice.
#
# Plain path: Softmax over the raw routing logits (`router_probs` despite
# the name -- confirmed against onnxruntime's own CPU MoE kernel: its
# output only matches a plain softmax-over-router_probs reference), top-k
# selection, optional renormalization (also a runtime `If`, on
# `normalize_routing_weights`) of the selected weights.
#
# Sparse-mixer path: transcribed from `sparse_mixer_top2` in
# onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu (CUDA-only in real ORT;
# no CPU kernel exists to check this against). For each of the top-2
# logits (k_idx in {0,1}): factor = max(|logits|, val_k); an expert is
# excluded from that pick's softmax denominator if its logit trails val_k
# by more than 2*jitter_eps*factor (jitter_eps = 0.01, hardcoded in the
# kernel, not a MoE attribute); the k_idx=1 pick additionally always
# excludes the k_idx=0 winner. Because the winning expert's own logit
# equals val_k, its numerator term is always exp(0) = 1, so each selected
# expert's routing weight reduces to plain 1/denominator.
def _routing_lines() -> str:
    return (
        "  probs = op.Softmax(router_probs, axis=-1)\n"
        "  k_i = op.Constant(value_int=_K_SENTINEL)\n"
        "  k_1d = op.Unsqueeze(k_i, op.Constant(value_ints=[0]))\n"
        "  top_vals, top_idx = op.TopK(probs, k_1d, axis=-1, largest=1)\n"
        "  reduce_axes = op.Constant(value_ints=[-1])\n"
        "  zero_f = op.Constant(value_float=0.0)\n"
        "  zero_cast = op.CastLike(zero_f, input)\n"
        "  normalize_i = op.Constant(value_int=_NORMALIZE_SENTINEL)\n"
        "  normalize_flag = op.Cast(normalize_i, to=9)\n"
        "  if normalize_flag:\n"
        "    denom = op.ReduceSum(top_vals, reduce_axes, keepdims=1)\n"
        "    top_vals_norm = op.Div(top_vals, denom)\n"
        "  else:\n"
        "    top_vals_norm = op.Identity(top_vals)\n"
        "  gate_zeros = op.Mul(probs, zero_cast)\n"
        "  softmax_gates = op.ScatterElements(gate_zeros, top_idx, top_vals_norm, axis=-1)\n"
        "\n"
        "  two_i = op.Constant(value_ints=[2])\n"
        "  sm_top_vals, sm_top_idx = op.TopK(router_probs, two_i, axis=-1, largest=1)\n"
        "  sm_val0 = op.Slice(sm_top_vals, op.Constant(value_ints=[0]), op.Constant(value_ints=[1]), op.Constant(value_ints=[-1]))\n"
        "  sm_val1 = op.Slice(sm_top_vals, op.Constant(value_ints=[1]), op.Constant(value_ints=[2]), op.Constant(value_ints=[-1]))\n"
        "  sm_idx0 = op.Slice(sm_top_idx, op.Constant(value_ints=[0]), op.Constant(value_ints=[1]), op.Constant(value_ints=[-1]))\n"
        "  sm_idx1 = op.Slice(sm_top_idx, op.Constant(value_ints=[1]), op.Constant(value_ints=[2]), op.Constant(value_ints=[-1]))\n"
        "  sm_jitter = op.Constant(value_float=0.02)\n"
        "  sm_jitter_cast = op.CastLike(sm_jitter, router_probs)\n"
        "  sm_factor0 = op.Max(op.Abs(router_probs), sm_val0)\n"
        "  sm_mask0 = op.Greater(op.Sub(sm_val0, router_probs), op.Mul(sm_jitter_cast, sm_factor0))\n"
        "  sm_zero = op.CastLike(op.Constant(value_float=0.0), router_probs)\n"
        "  sm_exp0 = op.Exp(op.Sub(router_probs, sm_val0))\n"
        "  sm_denom0 = op.ReduceSum(op.Where(sm_mask0, sm_zero, sm_exp0), reduce_axes, keepdims=1)\n"
        "  sm_one = op.CastLike(op.Constant(value_float=1.0), router_probs)\n"
        "  sm_w0 = op.Div(sm_one, sm_denom0)\n"
        "  sm_factor1 = op.Max(op.Abs(router_probs), sm_val1)\n"
        "  sm_mask1_raw = op.Greater(op.Sub(sm_val1, router_probs), op.Mul(sm_jitter_cast, sm_factor1))\n"
        "  sm_arange = op.Range(op.Constant(value_int=0), op.Squeeze(op.Shape(router_probs, start=-1)), op.Constant(value_int=1))\n"
        "  sm_is_idx0 = op.Equal(op.Cast(sm_arange, to=7), op.Cast(sm_idx0, to=7))\n"
        "  sm_mask1 = op.Or(sm_mask1_raw, sm_is_idx0)\n"
        "  sm_exp1 = op.Exp(op.Sub(router_probs, sm_val1))\n"
        "  sm_denom1 = op.ReduceSum(op.Where(sm_mask1, sm_zero, sm_exp1), reduce_axes, keepdims=1)\n"
        "  sm_w1 = op.Div(sm_one, sm_denom1)\n"
        "  sm_zero_row = op.Mul(router_probs, sm_zero)\n"
        "  sm_w_pair = op.Concat(sm_w0, sm_w1, axis=-1)\n"
        "  sm_idx_pair = op.Concat(sm_idx0, sm_idx1, axis=-1)\n"
        "  sparse_mixer_gates = op.ScatterElements(sm_zero_row, sm_idx_pair, sm_w_pair, axis=-1)\n"
        "\n"
        "  sparse_mixer_i = op.Constant(value_int=_SPARSE_MIXER_SENTINEL)\n"
        "  sparse_mixer_flag = op.Cast(sparse_mixer_i, to=9)\n"
        "  if sparse_mixer_flag:\n"
        "    gates = op.Identity(sparse_mixer_gates)\n"
        "  else:\n"
        "    gates = op.Identity(softmax_gates)\n"
    )


def _make_moe_function(
    activation: str,
    has_fc1_bias: bool,
    has_fc2_bias: bool,
    has_fc3_bias: bool | None = None,
):
    """Assembles the `@script()` source for one (activation, fc1_bias,
    fc2_bias[, fc3_bias]) fixed function -- arbitrary num_experts (a real
    ONNX `Loop`), arbitrary k/normalize_routing_weights/use_sparse_mixer
    (forwarded attributes and runtime `If`s), a single fixed activation (see
    module docstring for why that alone stays a build-time choice) -- via
    `exec`, the same technique contrib_schemas_moe_test.cpp's generator used
    before for the 16 (activation, bias) expert-block fragments, now
    producing a complete function per combination instead of a spliced-in
    fragment.

    `has_fc3_bias=None` (the default) means fc3 is absent entirely -- the
    plain per-expert FFN, `fc2(activation(fc1(x)))`. `has_fc3_bias` a bool
    means fc3 is present as a second, ungated per-expert Gemm combined with
    fc1's activated output via elementwise Mul before fc2:
    `fc2(activation(fc1(x)) * fc3(x))`. Only ever called with `activation ==
    "silu"` for that case -- see BuildMoEFunctionBody's comment in
    contrib_schemas.cpp for why: this is ONNX Runtime's own "Mixtral case"
    (onnxruntime/contrib_ops/cuda/moe/moe.cc, `kernel_activation_type =
    ActivationType::Swiglu` remap + the "[FC3, FC1]" weight-packing comment
    right above it) -- silu(fc1(x)) is the kernel's "SiLU(Gate)" and fc3(x)
    its "Linear", with no analogous defined behavior for the other three
    activations.
    """
    has_fc3 = has_fc3_bias is not None
    # Always declare the full, fixed-position formal-parameter signature
    # (matching com.microsoft.MoE's own schema exactly: input, router_probs,
    # fc1_experts_weights, fc1_experts_bias, fc2_experts_weights,
    # fc2_experts_bias[, fc3_experts_weights, fc3_experts_bias]) regardless of
    # has_fc1_bias/has_fc2_bias/has_fc3_bias, even though a given variant's
    # body below only ever *references* a bias input when its corresponding
    # flag is set.
    #
    # fc2_experts_weights/fc2_experts_bias sit *after* fc1_experts_bias in
    # that fixed layout, so a real calling node with fc1_experts_bias absent
    # but fc2_experts_bias present cannot simply supply fewer inputs (ONNX
    # only allows *trailing* optional inputs to be omitted that way) -- it
    # must pass an explicit "" placeholder at fc1_experts_bias's position,
    # meaning its actual input count still reaches through whichever of
    # fc2_experts_bias/fc3_experts_weights/fc3_experts_bias is the last one
    # actually present. A function declaring fewer formal parameters than
    # that (e.g. compacting fc1_experts_bias out of the signature when
    # has_fc1_bias is False) then fails ONNX's own "actuals <= formals"
    # inlining rule for exactly that calling pattern -- caught empirically by
    # this generator's own validation below.
    params = [
        'input: FLOAT["..."]',
        'router_probs: FLOAT["N", "E"]',
        'fc1_experts_weights: FLOAT["E", "I", "H"]',
        'fc1_experts_bias: FLOAT["E", "I"]',
        'fc2_experts_weights: FLOAT["E", "H", "I"]',
        'fc2_experts_bias: FLOAT["E", "H"]',
    ]
    if has_fc3:
        params.append('fc3_experts_weights: FLOAT["E", "I", "H"]')
        params.append('fc3_experts_bias: FLOAT["E", "I"]')

    def indent(text: str, prefix: str) -> str:
        return "".join(prefix + line + "\n" for line in text.splitlines())

    loop_lines = []
    loop_lines.append("w1_3d = op.Gather(fc1_experts_weights, e_idx, axis=0)")
    loop_lines.append("w1 = op.Squeeze(w1_3d, op.Constant(value_ints=[0]))")
    if has_fc1_bias:
        loop_lines.append("b1_2d = op.Gather(fc1_experts_bias, e_idx, axis=0)")
        loop_lines.append("b1 = op.Squeeze(b1_2d, op.Constant(value_ints=[0]))")
        loop_lines.append("h1 = op.Gemm(flat_input, w1, b1, transB=1)")
    else:
        loop_lines.append("h1 = op.Gemm(flat_input, w1, transB=1)")
    loop_lines.append(_activation_lines(activation).rstrip("\n"))
    if has_fc3:
        loop_lines.append("w3_3d = op.Gather(fc3_experts_weights, e_idx, axis=0)")
        loop_lines.append("w3 = op.Squeeze(w3_3d, op.Constant(value_ints=[0]))")
        if has_fc3_bias:
            loop_lines.append("b3_2d = op.Gather(fc3_experts_bias, e_idx, axis=0)")
            loop_lines.append("b3 = op.Squeeze(b3_2d, op.Constant(value_ints=[0]))")
            loop_lines.append("h3 = op.Gemm(flat_input, w3, b3, transB=1)")
        else:
            loop_lines.append("h3 = op.Gemm(flat_input, w3, transB=1)")
        loop_lines.append("a1 = op.Mul(a1, h3)")
    loop_lines.append("w2_3d = op.Gather(fc2_experts_weights, e_idx, axis=0)")
    loop_lines.append("w2 = op.Squeeze(w2_3d, op.Constant(value_ints=[0]))")
    if has_fc2_bias:
        loop_lines.append("b2_2d = op.Gather(fc2_experts_bias, e_idx, axis=0)")
        loop_lines.append("b2 = op.Squeeze(b2_2d, op.Constant(value_ints=[0]))")
        loop_lines.append("out = op.Gemm(a1, w2, b2, transB=1)")
    else:
        loop_lines.append("out = op.Gemm(a1, w2, transB=1)")
    loop_lines.append("gate_col = op.Gather(gates, e_idx, axis=1)")
    loop_lines.append("weighted = op.Mul(out, gate_col)")
    loop_lines.append("acc_final = op.Add(acc_final, weighted)")
    loop_body = indent("\n".join(loop_lines), "    ")

    routing = _routing_lines()

    source = (
        "@script()\n"
        f'def MoEFunction({", ".join(params)}) -> FLOAT["..."]:\n'
        "  hidden_shape = op.Shape(fc1_experts_weights, start=2)\n"
        "  neg_one = op.Constant(value_ints=[-1])\n"
        "  flat_shape = op.Concat(neg_one, hidden_shape, axis=0)\n"
        "  flat_input = op.Reshape(input, flat_shape)\n"
        "  input_shape = op.Shape(input)\n"
        "  num_experts_1d = op.Shape(fc1_experts_weights, start=0, end=1)\n"
        "  num_experts = op.Squeeze(num_experts_1d)\n"
        f"{routing}"
        "  acc_final = op.Mul(flat_input, zero_cast)\n"
        "  for e in range(num_experts):\n"
        "    e_idx = op.Reshape(e, op.Constant(value_ints=[1]))\n"
        f"{loop_body}"
        "  flat_output = op.Identity(acc_final)\n"
        "  output = op.Reshape(flat_output, input_shape)\n"
        "  return output\n"
    )
    namespace = {
        "script": script,
        "FLOAT": FLOAT,
        "op": op,
        "_K_SENTINEL": _K_SENTINEL,
        "_NORMALIZE_SENTINEL": _NORMALIZE_SENTINEL,
        "_SPARSE_MIXER_SENTINEL": _SPARSE_MIXER_SENTINEL,
        "__name__": __name__,
    }
    fc3_tag = "None" if has_fc3_bias is None else str(has_fc3_bias)
    filename = (
        f"<moe_function_{activation}_fc1{has_fc1_bias}_fc2{has_fc2_bias}_fc3{fc3_tag}>"
    )
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    try:
        exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    except Exception:
        print(source, file=sys.stderr)
        raise
    return namespace["MoEFunction"]


_ACTIVATIONS = ["relu", "identity", "silu", "gelu"]


def main() -> None:
    entries = []
    for activation in _ACTIVATIONS:
        for has_fc1_bias in (False, True):
            for has_fc2_bias in (False, True):
                fn = _make_moe_function(activation, has_fc1_bias, has_fc2_bias)
                proto = fn.to_function_proto()
                _bind_attribute(proto, _K_SENTINEL, "k")
                _bind_attribute(proto, _NORMALIZE_SENTINEL, "normalize_routing_weights")
                _bind_attribute(proto, _SPARSE_MIXER_SENTINEL, "use_sparse_mixer")
                text = _extract_body(onnx.printer.to_text(proto))
                ident = _cpp_ident(activation, has_fc1_bias, has_fc2_bias)
                entries.append((ident, text))

    # fc3 (Mixtral-style gated MLP, fc2(silu(fc1(x)) * fc3(x))) is only ever
    # generated for activation="silu" -- see _make_moe_function's and
    # BuildMoEFunctionBody's own comments for why relu/identity/gelu + fc3
    # stays declined instead (no defined ORT semantics for those).
    for has_fc1_bias in (False, True):
        for has_fc2_bias in (False, True):
            for has_fc3_bias in (False, True):
                fn = _make_moe_function(
                    "silu", has_fc1_bias, has_fc2_bias, has_fc3_bias
                )
                proto = fn.to_function_proto()
                _bind_attribute(proto, _K_SENTINEL, "k")
                _bind_attribute(proto, _NORMALIZE_SENTINEL, "normalize_routing_weights")
                _bind_attribute(proto, _SPARSE_MIXER_SENTINEL, "use_sparse_mixer")
                text = _extract_body(onnx.printer.to_text(proto))
                ident = _cpp_ident("silu", has_fc1_bias, has_fc2_bias, has_fc3_bias)
                entries.append((ident, text))

    out = []
    out.append("// SPDX-License-Identifier: Apache-2.0")
    out.append("//")
    out.append(
        "// GENERATED FILE -- do not edit by hand. Produced by\n"
        "//   python3 scripts/codegen/generate_moe_function_templates.py\n"
        "// from the onnxscript function definitions in that script; see its\n"
        "// module docstring for what's generic (Loop over num_experts,\n"
        "// attribute-forwarded k/normalize_routing_weights/use_sparse_mixer)\n"
        "// vs. still fixed per variant (activation_type, bias presence) and\n"
        "// why."
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
        with open(argv[0], "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
