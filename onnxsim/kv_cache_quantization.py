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
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


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


def _per_channel_absmax(
    candidates: Sequence[_KvCacheCandidate],
    model: onnx.ModelProto,
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Dict[str, np.ndarray]:
    probe = _add_probe_outputs(model, [c.new_name for c in candidates])
    absmax: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        outputs = backend.run_model(probe, batch, providers=providers)
        for c in candidates:
            arr = np.asarray(outputs[c.new_name], dtype=np.float64)
            if arr.ndim == 0:
                continue
            flat = arr.reshape(-1, arr.shape[-1])
            channel_max = np.abs(flat).max(axis=0)
            if c.new_name in absmax:
                absmax[c.new_name] = np.maximum(absmax[c.new_name], channel_max)
            else:
                absmax[c.new_name] = channel_max
    return absmax


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
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    if not _has_min_opset(model, 13):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    candidates = _find_kv_cache_candidates(graph)
    if not candidates:
        return out

    # Value-style needs ReduceMax's axes-as-input form, which (unlike
    # ReduceSum's, already opset13) only arrived at opset 18 -- each
    # Reduce* op moved its axes attribute to an input on its own schedule,
    # not all together at opset13. A stream matched as Value-style below
    # opset 18 is left completely untouched (not silently downgraded to
    # Key-style) rather than guessing.
    has_opset18 = _has_min_opset(model, 18)

    value_candidates = []
    channel_candidates = []
    for c in candidates:
        if _is_value_style(c.present_name, value_output_names):
            if has_opset18:
                value_candidates.append(c)
            # else: leave this stream untouched -- see comment above.
        else:
            channel_candidates.append(c)

    absmax: Dict[str, np.ndarray] = {}
    if channel_candidates:
        if calibration_data is None:
            calibration_data = generate_random_calibration_data(
                model, num_samples=num_samples, seed=seed
            )
        absmax = _per_channel_absmax(
            channel_candidates, model, calibration_data, providers
        )

    taken_names: Set[str] = _all_names(graph)
    input_by_name = {i.name: i for i in graph.input}
    output_by_name = {o.name: o for o in graph.output}

    for c in channel_candidates:
        if c.new_name not in absmax:
            continue  # this stream's activation never appeared in any batch
        _apply_channel_style(
            graph, c, absmax[c.new_name], taken_names, input_by_name, output_by_name
        )

    for c in value_candidates:
        _apply_value_style(graph, c, taken_names, input_by_name, output_by_name)

    return out


def _is_value_style(
    present_name: str, value_output_names: Optional[Sequence[str]]
) -> bool:
    if value_output_names is not None:
        return present_name in value_output_names
    return ".value" in present_name


def _rewire_consumers(
    graph: onnx.GraphProto, c: _KvCacheCandidate, dequant_name: str
) -> None:
    # Must run *before* any new node referencing c.present_name is
    # inserted into graph.node: RepeatedCompositeFieldContainer.insert()
    # copies the given message into a freshly allocated element rather
    # than storing the object itself, so an `is`-based identity check
    # taken afterward would never match anything actually in the
    # container (silently leaving a just-inserted node out of the
    # exclusion below and making it consume its own output). Running this
    # first sidesteps that: none of the new nodes exist in graph.node yet,
    # so there is nothing to incorrectly self-reference.
    for node in graph.node:
        if node is c.concat_node:
            continue
        for i, inp in enumerate(node.input):
            if inp == c.present_name:
                node.input[i] = dequant_name


def _apply_channel_style(
    graph: onnx.GraphProto,
    c: _KvCacheCandidate,
    channel_absmax: np.ndarray,
    taken_names: Set[str],
    input_by_name: Dict[str, onnx.ValueInfoProto],
    output_by_name: Dict[str, onnx.ValueInfoProto],
) -> None:
    """Static, calibrated, per-channel (Key-style) rewrite -- see the
    module docstring's diagram.
    """
    scale = (np.maximum(channel_absmax, 1e-12) / 127.0).astype(np.float32)
    num_channels = scale.shape[0]

    scale_name = _unique_name(f"{c.present_name}_kv_scale", taken_names)
    zp_name = _unique_name(f"{c.present_name}_kv_zero_point", taken_names)
    graph.initializer.append(onnx.numpy_helper.from_array(scale, name=scale_name))
    zp = np.zeros(num_channels, dtype=np.int8)
    graph.initializer.append(onnx.numpy_helper.from_array(zp, name=zp_name))

    # past_key/past_key_values.*: FLOAT -> INT8 (same shape).
    past_input = input_by_name[c.past_name]
    past_input.type.tensor_type.elem_type = onnx.TensorProto.INT8

    # new_key_q = QuantizeLinear(new_key, scale, zero_point, axis=channel_axis)
    new_q_name = _unique_name(f"{c.new_name}_kv_q", taken_names)
    quantize_node = onnx.helper.make_node(
        "QuantizeLinear",
        [c.new_name, scale_name, zp_name],
        [new_q_name],
        name=_unique_name(f"{c.new_name}_kv_quantize_node", taken_names),
        axis=c.channel_axis,
    )

    # Rewire Concat's "new" input to the now-quantized tensor; the "past"
    # input already reads the (now INT8) graph input as-is, so Concat's
    # own output is INT8 -- exactly present_key's new dtype.
    if c.new_is_first_input:
        c.concat_node.input[0] = new_q_name
    else:
        c.concat_node.input[1] = new_q_name

    present_output = output_by_name[c.present_name]
    present_output.type.tensor_type.elem_type = onnx.TensorProto.INT8

    # present_key_f = DequantizeLinear(present_key, scale, zero_point,
    # axis=channel_axis) -- every *node* consumer of the old float
    # present_key (the attention math) is rewired to this; the graph
    # output binding itself is untouched, so it keeps resolving to
    # Concat's own (now INT8) output tensor by name, unchanged.
    dequant_name = _unique_name(f"{c.present_name}_kv_f", taken_names)
    dequant_node = onnx.helper.make_node(
        "DequantizeLinear",
        [c.present_name, scale_name, zp_name],
        [dequant_name],
        name=_unique_name(f"{c.present_name}_kv_dequantize_node", taken_names),
        axis=c.channel_axis,
    )

    _rewire_consumers(graph, c, dequant_name)

    concat_idx = next(i for i, n in enumerate(graph.node) if n is c.concat_node)
    graph.node.insert(concat_idx, quantize_node)
    graph.node.insert(concat_idx + 2, dequant_node)


def _apply_value_style(
    graph: onnx.GraphProto,
    c: _KvCacheCandidate,
    taken_names: Set[str],
    input_by_name: Dict[str, onnx.ValueInfoProto],
    output_by_name: Dict[str, onnx.ValueInfoProto],
) -> None:
    """Data-free, per-token (Value-style) rewrite -- see the module
    docstring's diagram. Needs no calibration: each new token's own scale
    is computed from that token's own values, at graph-run time.
    """
    prefix = f"{c.present_name}_kv"
    past_input = input_by_name[c.past_name]
    present_output = output_by_name[c.present_name]
    past_rank = len(past_input.type.tensor_type.shape.dim)

    # New past_*_scale graph input: same rank/leading dims as past_* (read
    # before past_input's own dtype is mutated below), channel axis forced
    # to size 1 -- one scale per already-cached token, broadcasting over
    # head_dim. Picked up by KvCachePipeline's existing
    # present./past_key_values. string-substitution convention with no
    # C++ changes needed -- it stays float32, already handled.
    past_scale_name = _unique_name(f"{c.past_name}_scale", taken_names)
    past_scale_input = onnx.ValueInfoProto()
    past_scale_input.name = past_scale_name
    past_scale_input.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for i, d in enumerate(past_input.type.tensor_type.shape.dim):
        new_dim = past_scale_input.type.tensor_type.shape.dim.add()
        if i == c.channel_axis:
            new_dim.dim_value = 1
        elif d.HasField("dim_value"):
            new_dim.dim_value = d.dim_value
        elif d.HasField("dim_param"):
            new_dim.dim_param = d.dim_param
    graph.input.append(past_scale_input)

    past_input.type.tensor_type.elem_type = onnx.TensorProto.INT8

    eps_name = _unique_name(f"{prefix}_eps", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(1e-12, dtype=np.float32), name=eps_name)
    )
    div127_name = _unique_name(f"{prefix}_127", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(127.0, dtype=np.float32), name=div127_name
        )
    )
    clip_min_name = _unique_name(f"{prefix}_clip_min", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(-128.0, dtype=np.float32), name=clip_min_name
        )
    )
    clip_max_name = _unique_name(f"{prefix}_clip_max", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(127.0, dtype=np.float32), name=clip_max_name
        )
    )
    axes_name = _unique_name(f"{prefix}_reduce_axes", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([c.channel_axis], dtype=np.int64), name=axes_name
        )
    )

    # new_scale = max(reduce_max(abs(new_value), axis=channel_axis), eps) / 127
    abs_name = _unique_name(f"{prefix}_abs", taken_names)
    max_name = _unique_name(f"{prefix}_max", taken_names)
    safe_max_name = _unique_name(f"{prefix}_safe_max", taken_names)
    new_scale_name = _unique_name(f"{prefix}_new_scale", taken_names)
    # new_value_q = cast(clip(round(new_value / new_scale), -128, 127), INT8)
    scaled_name = _unique_name(f"{prefix}_scaled", taken_names)
    rounded_name = _unique_name(f"{prefix}_rounded", taken_names)
    clipped_name = _unique_name(f"{prefix}_clipped", taken_names)
    new_q_name = _unique_name(f"{c.new_name}_kv_q", taken_names)

    pre_nodes = [
        onnx.helper.make_node("Abs", [c.new_name], [abs_name]),
        onnx.helper.make_node(
            "ReduceMax", [abs_name, axes_name], [max_name], keepdims=1
        ),
        onnx.helper.make_node("Clip", [max_name, eps_name], [safe_max_name]),
        onnx.helper.make_node("Div", [safe_max_name, div127_name], [new_scale_name]),
        onnx.helper.make_node("Div", [c.new_name, new_scale_name], [scaled_name]),
        onnx.helper.make_node("Round", [scaled_name], [rounded_name]),
        onnx.helper.make_node(
            "Clip", [rounded_name, clip_min_name, clip_max_name], [clipped_name]
        ),
        onnx.helper.make_node(
            "Cast", [clipped_name], [new_q_name], to=onnx.TensorProto.INT8
        ),
    ]

    if c.new_is_first_input:
        c.concat_node.input[0] = new_q_name
    else:
        c.concat_node.input[1] = new_q_name
    present_output.type.tensor_type.elem_type = onnx.TensorProto.INT8

    # present_*_scale: NEW graph output, grows in lockstep with present_*
    # itself (same seq_axis Concat, same two operands' relative order).
    present_scale_name = _unique_name(f"{c.present_name}_scale", taken_names)
    present_scale_output = onnx.ValueInfoProto()
    present_scale_output.name = present_scale_name
    present_scale_output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for _ in range(past_rank):
        present_scale_output.type.tensor_type.shape.dim.add()
    present_scale_output.type.tensor_type.shape.dim[c.channel_axis].dim_value = 1
    graph.output.append(present_scale_output)

    present_f32_name = _unique_name(f"{prefix}_present_f32", taken_names)
    dequant_name = _unique_name(f"{c.present_name}_kv_f", taken_names)
    post_nodes = [
        onnx.helper.make_node(
            "Concat",
            [past_scale_name, new_scale_name],
            [present_scale_name],
            name=_unique_name(f"{prefix}_concat_scale_node", taken_names),
            axis=c.seq_axis,
        ),
        onnx.helper.make_node(
            "Cast", [c.present_name], [present_f32_name], to=onnx.TensorProto.FLOAT
        ),
        onnx.helper.make_node(
            "Mul", [present_f32_name, present_scale_name], [dequant_name]
        ),
    ]

    _rewire_consumers(graph, c, dequant_name)

    concat_idx = next(i for i, n in enumerate(graph.node) if n is c.concat_node)
    for offset, node in enumerate(pre_nodes):
        graph.node.insert(concat_idx + offset, node)
    for offset, node in enumerate(post_nodes):
        graph.node.insert(concat_idx + len(pre_nodes) + 1 + offset, node)
