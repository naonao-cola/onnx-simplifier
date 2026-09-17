"""KV-cache quantization for autoregressive decoder graphs -- see
``docs/kv-cache-quantization.md`` for the full survey (KIVI, KVQuant) this
module implements the recommendation of, and usage examples.

Every quantizer elsewhere in onnxsim compresses a *weight* -- something
computed once, offline, before the model ever runs. A KV cache is the
opposite: it is an *activation* that keeps growing for the whole lifetime of
one autoregressive generation, one new key/value vector appended per decode
step, which is exactly why quantizing it is worth doing at all (it is the
part of an LLM's memory footprint that scales with sequence length, unlike
the weights).

Two published techniques quantize the KV cache well: KIVI (Liu et al., ICML
2024, https://arxiv.org/abs/2402.02750) and KVQuant (Hooper et al., NeurIPS
2024, https://arxiv.org/abs/2401.18079). Both share the same core empirical
finding -- Key activations have a handful of channels with persistently
large magnitude across the *whole* sequence, so quantizing Key **per
channel** (one scale shared by every cached token, along the head-dim axis)
preserves far more accuracy than quantizing it per token. This module
reproduces that part of both papers: a **static, per-channel (head-dim
axis) scale**, calibrated once from representative data, applied to
whichever of the graph's ``Concat(past, new)`` KV-cache patterns it finds --
the same op shape ``tools/onnx-deploy``'s own pipeline and this repo's own
``tests/test_symexpr_kv_cache_consistency.py`` toy model use for a decoder's
cache: a graph input (``past_key``/``past_key_values.{i}.key``, ...) and a
freshly computed activation, concatenated along the sequence axis, feeding a
graph output (``present_key``/``present.{i}.key``, ...) that the caller
feeds back in as next step's ``past_*`` input.

KIVI's *other* empirical finding is that Value activations don't have that
persistent-channel structure, so a static per-channel scale is the wrong
shape for Value -- a **fresh, per-token scale** (computed from that token's
own values, the instant it's produced) preserves much more accuracy there
instead. This module reproduces that too, for every matched stream whose
``present`` output name contains ``".value"`` (matching this repo's own
``present.{i}.decoder.value``/``present.{i}.encoder.value`` convention --
see ``tools/onnx-deploy/scripts/make_toy_seq2seq.py``) or is named
explicitly via ``value_output_names`` -- every other matched stream keeps
the per-channel treatment above. Per-token quantization needs no
calibration data at all (each token's own scale is computed from that
token's own values, at graph-run time), but it does need the scale itself
carried forward as a **second, parallel growing KV-cache stream**
alongside the codes -- see the graph rewrite below. It also needs opset 18
(``ReduceMax``'s ``axes``-as-input form, unlike ``ReduceSum``'s -- already
opset 13 -- only arrived there; each ``Reduce*`` op moved its ``axes``
attribute to an input on its own schedule, not all at once); a stream
matched as Value-style below opset 18 is left completely untouched rather
than silently downgraded to Key-style.

What this module does **not** reproduce: KIVI's residual-window
bookkeeping (the most recent ``R`` tokens kept in full precision, only
finalized into low-bit once they age out of that window). Deciding which
tokens have "aged out" and need finalizing is cross-step, host-side
bookkeeping -- not something one exported ONNX graph can express on its
own -- and belongs in
``tools/onnx-deploy/include/onnx_deploy/kv_cache_pipeline.h`` (which
already owns exactly this kind of cross-step cache state) as a follow-up,
not here.

Graph rewrite, per matched ``Concat(past, new, axis=seq)`` cache stream --
**Key-style** (default; static, calibrated, per-channel):

    Before:
      past_key: graph input, float32 [..., seq_past, head_dim]
      new_key:  float32 [..., seq_new, head_dim]        -- this step's own K/V
      present_key = Concat(past_key, new_key, axis=seq)  -- graph output,
                    and consumed by the attention math (QK^T / softmax@V)

    After:
      past_key: graph input, INT8 [..., seq_past, head_dim]   -- dtype changed
      key_scale: initializer, float32 [head_dim]                -- per-channel
      key_zero_point: initializer, INT8 [head_dim], all zero    -- symmetric
      new_key_q = QuantizeLinear(new_key, key_scale, key_zero_point, axis=-1)
      present_key = Concat(past_key, new_key_q, axis=seq)   -- INT8 graph output
      present_key_f = DequantizeLinear(present_key, key_scale, key_zero_point,
                                        axis=-1)             -- float32
      <every other consumer of the old float present_key now reads present_key_f>

Concatenating ``past_key`` (already int8) with ``new_key_q`` (freshly
quantized with the *same* per-channel scale) along the sequence axis is
lossless with respect to what was already stored -- the scale never changes
step to step, so there is no compounding requantization error the way there
would be if the whole growing cache were dequantized and requantized with a
fresh scale every step. Only this step's new tokens are ever quantized; the
cost per decode step stays constant as the sequence grows, and the graph's
own ``present_*`` output is genuinely compressed (roughly 4x smaller than
float32) the whole way through a caller's decode loop -- not just an
internal round-trip that still stores float32 everywhere.

**Value-style** (data-free, per-token, matched by name -- see above):

    Before:
      past_value: graph input, float32 [..., seq_past, head_dim]
      new_value:  float32 [..., seq_new, head_dim]
      present_value = Concat(past_value, new_value, axis=seq)

    After:
      past_value: graph input, INT8 [..., seq_past, head_dim]
      past_value_scale: graph input, float32 [..., seq_past, 1]   -- NEW input,
        one scale per already-cached token -- threaded by
        KvCachePipeline's existing present./past_key_values. convention
        with no C++ changes (it stays float32, and BorrowView already
        handled float32 before this module existed)
      new_scale = Max(ReduceMax(Abs(new_value), axes=[-1], keepdims=1), eps) / 127
        -- one scale per *new* token, computed fresh from that token's own
        values, no calibration data involved
      new_value_q = Cast(Clip(Round(new_value / new_scale), -128, 127), INT8)
      present_value = Concat(past_value, new_value_q, axis=seq)        -- INT8
      present_value_scale = Concat(past_value_scale, new_scale, axis=seq)  -- NEW
        output, float32, grows in lockstep with present_value
      present_value_f = Cast(present_value, float32) * present_value_scale
        -- broadcasts present_value_scale's trailing size-1 axis over head_dim
      <every other consumer of the old float present_value now reads present_value_f>

Past tokens' scales are never revised once set (matching the Key-style
scheme's "no compounding requantization error" property above) -- only
this step's new token(s) are ever quantized, at a fresh, tailored scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import quantize_kv_cache_cpp


@dataclass
class _KvCacheCandidate:
    past_name: str  # graph input name (e.g. "past_key", "past_key_values.0.key")
    present_name: str  # graph output name (Concat's own output, unchanged)
    concat_node: onnx.NodeProto
    new_name: str  # the freshly-computed operand of Concat (not the cache)
    new_is_first_input: bool
    seq_axis: int  # resolved (non-negative) Concat axis
    channel_axis: int  # resolved (non-negative) quantization axis -- last axis


def _resolve_axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else axis + rank


def _find_kv_cache_candidates(graph: onnx.GraphProto) -> List[_KvCacheCandidate]:
    """Structurally matches ``Concat(past, new, axis=seq)`` where ``past`` is
    a float32 graph input consumed *only* by this Concat, and the Concat's
    own output is directly a graph output -- exactly the shape
    ``tools/onnx-deploy``'s ``KvCachePipeline`` and
    ``tests/test_symexpr_kv_cache_consistency.py``'s toy model both use, and
    make no assumption about tensor names (works for ``past_key``/
    ``present_key`` as well as ``optimum-onnx``'s own
    ``past_key_values.{i}.key``/``present.{i}.key`` convention).
    """
    output_names = {o.name for o in graph.output}
    float_inputs: Dict[str, int] = {}  # name -> rank
    for inp in graph.input:
        if inp.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            continue
        float_inputs[inp.name] = len(inp.type.tensor_type.shape.dim)

    consumer_count: Dict[str, int] = {}
    for node in graph.node:
        for inp in node.input:
            consumer_count[inp] = consumer_count.get(inp, 0) + 1

    candidates = []
    for node in graph.node:
        if node.op_type != "Concat" or len(node.input) != 2:
            continue
        if len(node.output) != 1 or node.output[0] not in output_names:
            continue
        a, b = node.input
        if a in float_inputs and consumer_count.get(a, 0) == 1:
            past_name, new_name, new_is_first = a, b, False
        elif b in float_inputs and consumer_count.get(b, 0) == 1:
            past_name, new_name, new_is_first = b, a, True
        else:
            continue
        axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
        if axis_attr is None:
            continue
        rank = float_inputs[past_name]
        seq_axis = _resolve_axis(axis_attr.i, rank)
        channel_axis = rank - 1
        if seq_axis == channel_axis:
            continue  # no distinct channel axis left to quantize per-channel on
        candidates.append(
            _KvCacheCandidate(
                past_name=past_name,
                present_name=node.output[0],
                concat_node=node,
                new_name=new_name,
                new_is_first_input=new_is_first,
                seq_axis=seq_axis,
                channel_axis=channel_axis,
            )
        )
    return candidates


def quantize_kv_cache(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    value_output_names: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every ``Concat(past, new, axis=seq)`` KV-cache stream this
    module can find (see this module's own docstring for the exact pattern
    and both graph rewrites) to INT8, symmetric. Key-style streams (the
    default) get one scale per channel (the last axis -- head-dim),
    calibrated once from representative data and shared by every cached
    token for that stream's whole lifetime. Value-style streams (matched by
    name -- see ``value_output_names``) get a fresh, data-free scale per
    token instead, computed from that token's own values the moment it's
    produced.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) to calibrate Key-style streams' per-channel scale on --
            see :func:`onnxsim.generate_random_calibration_data` (the
            default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            more representative calibration than random input). Ignored
            for Value-style streams, which need no calibration data at
            all. A ``past_key``/``past_key_values.*`` input with a
            genuinely empty (statically zero) sequence-length dimension in
            ``model``'s own declared shape is filled in as an empty tensor
            by :func:`onnxsim.generate_random_calibration_data` automatically
            -- calibration only ever measures this step's own freshly
            computed activation, never the cache's prior content, so an
            empty starting cache calibrates identically to a populated one.
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to calibrate on
    :param value_output_names: which matched streams' ``present`` output
            names get Value-style (per-token) treatment instead of the
            default Key-style (per-channel) one -- if omitted, any matched
            stream whose ``present`` output name contains ``".value"`` is
            treated as Value-style automatically (matching this repo's own
            ``present.{i}.decoder.value``/``present.{i}.encoder.value``
            convention), every other stream gets Key-style. A stream
            matched as Value-style is left completely untouched (not
            downgraded to Key-style) when ``model``'s opset is below 18 --
            see this module's own docstring
    :returns: ``model`` with every matched KV-cache stream's ``past_*``
            graph input and ``present_*`` graph output changed to INT8
            (Value-style streams additionally gain a new
            ``past_*_scale``/``present_*_scale`` float32 input/output
            pair -- see the module docstring's diagram); a model with no
            matching Concat pattern, or an opset older than 13
            (``QuantizeLinear``/``DequantizeLinear``'s per-channel ``axis``,
            and ``ReduceMax``'s ``axes``-as-input, both need opset 13), is
            returned unchanged

    Delegates to the verified C++ port
    (:func:`onnxsim.quantize_kv_cache_cpp`), which shares this function's
    own full parameter set exactly -- no compatibility gap.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return quantize_kv_cache_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        value_output_names=value_output_names,
        providers=providers,
    )


def _is_value_style(
    present_name: str, value_output_names: Optional[Sequence[str]]
) -> bool:
    if value_output_names is not None:
        return present_name in value_output_names
    return ".value" in present_name
