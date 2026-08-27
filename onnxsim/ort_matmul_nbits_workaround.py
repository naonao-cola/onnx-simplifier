"""Workaround for an ONNX Runtime graph-optimizer bug (reproduced against
onnxruntime 1.29.0; not yet confirmed fixed in later releases -- check
before assuming this module is still needed) that silently miscomputes
``axis=0`` block-quantized ``DequantizeLinear`` feeding a plain (i.e. not
``Gemm transB=1``) MatMul/Gemm, under ONNX Runtime's *default* graph
optimization settings.

Every onnxsim INT4 weight-only scheme built on ``DequantizeLinear`` +
MatMul/Gemm -- :func:`onnxsim.quantize_weight_only_int4` itself, and
everything that refines its output in place (:func:`onnxsim.apply_adaround`,
:func:`onnxsim.apply_awq`, :func:`onnxsim.apply_gptq`,
:func:`onnxsim.apply_autoround`), plus :func:`onnxsim.quantize_weight_only_int4_hqq`,
which builds its own analogous ``DequantizeLinear`` -- picks
``axis = 0`` for a *plain* (not ``transB=1``) MatMul/Gemm weight, keeping
the weight in its natural ``[K, N]`` storage layout with no ``Transpose``
node needed (see ``passes/weight_only_quantize_int4_matmul.h``'s own
``channel_axis``/``reduction_axis`` comment). That choice is fully
ONNX-spec-compliant -- ``DequantizeLinear``'s ``axis`` attribute is
documented to support any axis -- but at ONNX Runtime's default graph
optimization level (``ORT_ENABLE_EXTENDED``/``ORT_ENABLE_ALL``, i.e.
level 2+), its ``MatMulNBitsFusion`` transformer fuses ``DequantizeLinear
+ MatMul`` into the ``com.microsoft.MatMulNBits`` contrib op *without
checking the ``axis`` attribute first*. ``MatMulNBits`` has no ``axis``
parameter of its own at all -- ONNX Runtime's own documentation describes
it as equivalent to ``DequantizeLinear + Transpose + MatMul``, i.e. it
only supports the ``axis = 1`` (one scale block per output row) layout.
Fusing an ``axis = 0`` graph into it silently reinterprets the weight
data as if it used that other layout, producing wrong results -- verified
by comparing the same model run with ``ORT_DISABLE_ALL``/``ORT_ENABLE_BASIC``
(correct, matches the float model almost exactly) against
``ORT_ENABLE_EXTENDED``/``ORT_ENABLE_ALL`` (matches
``ORT_DISABLE_ALL``'s number of blocks apart, i.e. wrong; the ``axis = 1``
-- ``Gemm transB=1`` -- path is unaffected at every level).

This module works around it by finding every ``axis = 0`` DequantizeLinear
feeding a plain MatMul/Gemm and rewriting it to the ``axis = 1`` shape the
fusion actually supports: transpose the (already-quantized, packed) INT4/
UINT4 codes, scale, and optional zero-point tensors themselves -- an exact,
lossless re-indexing, not a requantization -- into a new ``DequantizeLinear
(axis=1)`` followed by an explicit ``Transpose`` back to the original
layout, functionally identical to what a genuinely ``transB=1``-stored
weight would already look like. Verified empirically that ONNX Runtime's
optimizer no longer misfires ``MatMulNBitsFusion`` on this shape (a
different, correct fusion -- folding the ``Transpose`` into the following
MatMul -- applies instead), and that the result is numerically identical
with graph optimizations enabled or disabled.

This is deliberately a standalone, opt-in post-processing pass rather than
a change to the underlying quantizers' own default output: it leaves
every existing ``quantize_weight_only_int4``/AWQ/GPTQ/AdaRound/AutoRound/
HQQ model's own emitted format untouched unless you explicitly call it,
and is meant to be applied once, right before deployment, to any onnxsim
INT4-quantized model that will be run under ONNX Runtime's default
optimization settings.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name


def _match_plain_weight_dequantize_linear(node: onnx.NodeProto):
    """Returns the weight input's own name for a MatMul, or a Gemm with
    ``transA=0``, ``alpha=1``, ``beta=1`` (if it has a bias) and
    ``transB=0`` -- the "plain" (not already ``transB=1``-transposed)
    layout that :func:`onnxsim.quantize_weight_only_int4` and friends
    quantize with ``axis=0``. Returns ``None`` for anything else,
    including an explicitly ``transB=1`` Gemm (that path uses ``axis=1``
    already and isn't affected by the bug this module works around).
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[1]
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        if trans_b is not None and trans_b.i:
            return None  # already axis=1; not affected
        return node.input[1]
    return None


def workaround_ort_matmul_nbits_axis0_bug(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """Rewrites every ``axis=0`` block-quantized ``DequantizeLinear``
    feeding a plain (not ``transB=1``) MatMul/Gemm into the ``axis=1``
    form ONNX Runtime's ``MatMulNBitsFusion`` transformer actually
    supports, avoiding the wrong results it silently produces for
    ``axis=0`` under default graph optimization settings. See this
    module's own docstring for the full story.

    :param model: an onnx ModelProto or file path, already quantized by
            :func:`onnxsim.quantize_weight_only_int4` (or anything built
            on it: :func:`onnxsim.apply_adaround`, :func:`onnxsim.apply_awq`,
            :func:`onnxsim.apply_gptq`, :func:`onnxsim.apply_autoround`,
            :func:`onnxsim.quantize_weight_only_int4_hqq`)
    :returns: ``model`` with every matched layer's weight input rewired to
            a new, transposed ``DequantizeLinear(axis=1)`` followed by a
            ``Transpose`` back to the original layout; the original
            ``DequantizeLinear`` and its own Wq/Ws/Wz initializers are left
            in place (unused, but not pruned, matching every onnxsim
            ``quantize_*``/``apply_*`` pass's own convention of never
            deleting a tensor it no longer needs). Layers whose weight
            isn't fed by a 2- or 3-input ``axis=0`` blockwise
            ``DequantizeLinear`` are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    dq_by_output = {
        n.output[0]: n for n in graph.node if n.op_type == "DequantizeLinear"
    }

    for node in list(graph.node):
        w_name = _match_plain_weight_dequantize_linear(node)
        if w_name is None:
            continue
        dq = dq_by_output.get(w_name)
        if dq is None or len(dq.input) not in (2, 3):
            continue
        axis = next((a.i for a in dq.attribute if a.name == "axis"), None)
        block_size = next((a.i for a in dq.attribute if a.name == "block_size"), None)
        if axis != 0 or not block_size:
            continue

        transposed_inputs = []
        for inp_name in dq.input:
            t = initializer_map.get(inp_name)
            if t is None:
                break
            arr = onnx.numpy_helper.to_array(t)
            if arr.ndim != 2:
                break
            new_name = _unique_name(f"{inp_name}_axis1", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(np.ascontiguousarray(arr.T), name=new_name)
            )
            transposed_inputs.append(new_name)
        if len(transposed_inputs) != len(dq.input):
            continue  # a required initializer was missing or not 2-D; skip

        dq_out_nk = _unique_name(f"{w_name}_axis1_dq", taken_names)
        new_dq = onnx.helper.make_node(
            "DequantizeLinear",
            transposed_inputs,
            [dq_out_nk],
            name=_unique_name(f"{w_name}_axis1_dq_node", taken_names),
            axis=1,
            block_size=block_size,
        )
        transposed_out = _unique_name(f"{w_name}_axis1_transposed", taken_names)
        transpose_node = onnx.helper.make_node(
            "Transpose",
            [dq_out_nk],
            [transposed_out],
            name=_unique_name(f"{w_name}_axis1_transpose_node", taken_names),
            perm=[1, 0],
        )

        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        graph.node.insert(node_idx, new_dq)
        graph.node.insert(node_idx + 1, transpose_node)
        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = transposed_out

    return out
