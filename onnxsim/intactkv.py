"""IntactKV (Liu, Zhang, Wang, Jin, Sun, Gu, Zeng, Zhu, Wei, Cheng, Chen,
Zhang, 2024, "IntactKV: Improving Large Language Model Quantization by
Keeping Pivot Tokens Intact", https://arxiv.org/abs/2403.01241).

Distinguishing this module from every other KV-cache module already in this
repo -- :mod:`onnxsim.kv_cache_quantization` (KIVI/KVQuant), its two
siblings :mod:`onnxsim.rotatekv` and :mod:`onnxsim.gear`, and
:mod:`onnxsim.qoq`'s :func:`onnxsim.apply_smooth_attention`: **every one of
those four quantizes or transforms every cached token's own Key/Value.**
``kv_cache_quantization`` picks a scale (per-channel for Key, per-token for
Value) and quantizes every token with it; ``rotatekv`` conjugates every
token's Key by a fitted rotation *before* that same quantization;
``gear`` computes a low-rank-plus-sparse correction for the reconstruction
error *every* quantized token leaves behind; ``apply_smooth_attention``
migrates outlier magnitude out of *every* token's Key channel-by-channel.
All four accept that every position in the cache pays a quantization cost
and work to make that cost smaller. IntactKV's own finding is different in
kind, not degree: a small, fixed set of **pivot tokens** -- in practice
almost always the sequence's first few tokens (the well-documented
"attention sink" phenomenon: a handful of early positions draw a
disproportionate share of *every* later token's attention, essentially
regardless of their own content, because softmax attention needs
*somewhere* to route the score mass an irrelevant query has nowhere better
to send) -- are disproportionately *sensitive* to KV-cache quantization
error, out of proportion to their tiny share of the sequence. So instead of
quantizing them better (the other four modules' shared strategy), this
module simply **does not quantize them at all**: it identifies that fixed
pivot prefix and keeps its Key/Value entries at exact float precision,
computed once, forever -- letting a *separate* quantizer (any one of the
four above, most naturally :func:`onnxsim.quantize_kv_cache`) handle every
other position exactly as it already does. IntactKV is not a competing
KV-cache quantizer; it is a *companion* that carves a quantization-exempt
notch out of the cache for whichever quantizer runs after it.

**Scope: a two-stream split, not a per-step re-slice.** The pivot tokens
are, after the prompt's initial prefill pass, a *fixed* prefix -- always
the same ``num_pivot_tokens`` entries, never revised, never grown --
sitting in front of a cache that otherwise keeps growing one token per
decode step. Splitting those two very differently-shaped pieces back out
of one merged ``past_*`` tensor with a fresh ``Slice`` *every single decode
step* would be pure wasted work repeated forever for a boundary that never
moves. This module instead changes the KV-cache stream's own I/O contract,
once, to expose the pivot prefix as its own separate, fixed-size stream
from the start -- the same design :mod:`onnxsim.kv_cache_quantization`'s
own Value-style rewrite already uses for its per-token scale (a second,
independently-shaped tensor threaded alongside the codes, rather than
packed into one tensor with the thing it's paired with). Concretely, for
each matched ``Concat(past, new, axis=seq)`` stream (found with
:mod:`onnxsim.kv_cache_quantization`'s own
``_find_kv_cache_candidates`` -- the exact same structural pattern, not
reimplemented here):

    Before:
      past_key: graph input, float32 [..., seq_past, head_dim]
      new_key:  float32 [..., seq_new, head_dim]
      present_key = Concat(past_key, new_key, axis=seq)   -- graph output,
                    consumed by the attention math

    After:
      past_key_pivot: NEW graph input, float32 [..., num_pivot_tokens,
                      head_dim] -- fixed size, holds the pivot tokens'
                      exact Key (or Value), never revised after being set
      past_key_rest:  past_key, renamed -- float32 [..., seq_past -
                      num_pivot_tokens, head_dim], everything *but* the
                      pivots; this is an ordinary
                      ``Concat(past, new, axis=seq)`` stream in its own
                      right, ready to be matched and quantized by
                      whichever of this repo's other KV-cache quantizers
                      the caller applies next
      present_key_rest = Concat(past_key_rest, new_key, axis=seq)  --
                      graph output, same tensor Concat node/output as the
                      original, just renamed -- this is what
                      :func:`onnxsim.quantize_kv_cache` (or
                      :mod:`onnxsim.rotatekv`/:mod:`onnxsim.gear`) matches
                      and quantizes when run *after* this module
      present_key_pivot = Identity(past_key_pivot)  -- NEW graph output,
                      exact passthrough: the pivot tokens are never
                      recomputed or touched once set, so this is a
                      bit-for-bit copy every single step
      present_key = Concat(present_key_pivot, present_key_rest, axis=seq)
                      -- same output name/binding as the original
                      present_key, now produced by this reconstruction
                      node instead of the original Concat; every original
                      consumer (the attention math) is unaffected, since
                      it already referred to ``present_key`` by name

Composed with a following :func:`onnxsim.quantize_kv_cache` call, that
quantizer's own rewrite changes ``past_key_rest``/``present_key_rest`` to
INT8 and inserts a ``DequantizeLinear`` whose output feeds every consumer
of the old float ``present_key_rest`` -- which, after this module has run,
means the reconstruction ``Concat`` above. So ``present_key`` ends up
built from an *exact* pivot half and a *lossily dequantized* rest half,
which is exactly IntactKV's own claim.

**What this module does not do.** It does not compute the pivot tokens'
Key/Value in the first place -- an exported decoder step-graph already
computes Key/Value for every prompt token during prefill, pivots included,
as an ordinary consequence of running the model; this module only changes
how a *later* step's cache I/O is shaped. Slicing the first
``num_pivot_tokens`` entries off *that* prefill step's own float
``present_key`` output, exactly once, and feeding the result back in as
every subsequent step's ``past_key_pivot`` is cross-step, host-side
bookkeeping -- deciding "this was the prefill step" is not information one
exported step-graph has about itself -- and belongs in
``tools/onnx-deploy/include/onnx_deploy/kv_cache_pipeline.h`` as a
follow-up, exactly the same scope line
:mod:`onnxsim.kv_cache_quantization`'s own docstring already draws around
KIVI's residual-window bookkeeping for the identical reason. This module's
own job stops at making sure the *graph* keeps that one-time slice's
result untouched forever after -- not deciding when to take it.
"""

from __future__ import annotations

from typing import Dict, Set, Union

import onnx
import onnx.helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.kv_cache_quantization import _find_kv_cache_candidates, _KvCacheCandidate


def apply_intactkv(
    model: Union[str, onnx.ModelProto],
    num_pivot_tokens: int = 4,
) -> onnx.ModelProto:
    """Splits every ``Concat(past, new, axis=seq)`` KV-cache stream this
    module can find (the same pattern
    :func:`onnxsim.quantize_kv_cache` matches -- see this module's own
    docstring for the exact rewrite) into two independent streams: a fixed,
    ``num_pivot_tokens``-long pivot prefix that this module guarantees
    stays exact float32 forever, and the remaining, still-growing "rest" of
    the cache, restructured as an ordinary same-shaped KV-cache stream of
    its own -- ready for a *following* call to
    :func:`onnxsim.quantize_kv_cache` (or :mod:`onnxsim.rotatekv`/
    :mod:`onnxsim.gear`) to actually quantize. This module never quantizes
    anything itself, the same way :func:`onnxsim.apply_smooth_attention`
    only migrates and leaves quantizing to a later pipeline stage.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param num_pivot_tokens: how many of each stream's leading (oldest)
            cached tokens to exempt from quantization -- the paper's own
            typical choice is 4; the caller is responsible for supplying
            that many tokens' worth of pivot Key/Value (sliced once from
            the prompt's own prefill-step output -- see this module's own
            docstring) as each new ``*_pivot`` graph input from the first
            decode step onward
    :returns: ``model`` with every matched KV-cache stream split into a
            ``*_pivot``/``*_rest`` pair of graph input/output streams (see
            the module docstring's diagram) and the original
            ``present_*`` output rebuilt as their ``Concat``; a model with
            no matching ``Concat(past, new, axis=seq)`` pattern is returned
            unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    candidates = _find_kv_cache_candidates(graph)
    if not candidates:
        return out

    taken_names: Set[str] = _all_names(graph)
    input_by_name = {i.name: i for i in graph.input}

    for c in candidates:
        _split_pivot_stream(graph, c, num_pivot_tokens, taken_names, input_by_name)

    return out


def _split_pivot_stream(
    graph: onnx.GraphProto,
    c: _KvCacheCandidate,
    num_pivot_tokens: int,
    taken_names: Set[str],
    input_by_name: Dict[str, onnx.ValueInfoProto],
) -> None:
    past_input = input_by_name[c.past_name]

    pivot_past_name = _unique_name(f"{c.past_name}_pivot", taken_names)
    rest_past_name = _unique_name(f"{c.past_name}_rest", taken_names)
    pivot_present_name = _unique_name(f"{c.present_name}_pivot", taken_names)
    rest_present_name = _unique_name(f"{c.present_name}_rest", taken_names)

    # New past_*_pivot graph input: same rank/leading dims as past_*, seq
    # axis fixed to num_pivot_tokens -- read before past_input's own name
    # is mutated below.
    pivot_input = onnx.ValueInfoProto()
    pivot_input.name = pivot_past_name
    pivot_input.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for i, d in enumerate(past_input.type.tensor_type.shape.dim):
        new_dim = pivot_input.type.tensor_type.shape.dim.add()
        if i == c.seq_axis:
            new_dim.dim_value = num_pivot_tokens
        elif d.HasField("dim_value"):
            new_dim.dim_value = d.dim_value
        elif d.HasField("dim_param"):
            new_dim.dim_param = d.dim_param
    graph.input.append(pivot_input)

    # Rename past_* -> past_*_rest in place: still consumed only by
    # concat_node, still float32, now holding everything but the pivots.
    past_input.name = rest_past_name
    for i, inp in enumerate(c.concat_node.input):
        if inp == c.past_name:
            c.concat_node.input[i] = rest_past_name

    # present_* -> present_*_rest in place: concat_node's own output,
    # unchanged apart from the name, now a plain KV-cache stream in its
    # own right for a following quantizer to match.
    c.concat_node.output[0] = rest_present_name
    rest_output = onnx.ValueInfoProto()
    rest_output.name = rest_present_name
    rest_output.type.CopyFrom(past_input.type)
    # seq_past - num_pivot_tokens (statically unknown here, same as the
    # original present_* declaration) -- clear any dim_value the pivot
    # branch above would have copied and leave the seq axis symbolic.
    rest_output.type.tensor_type.shape.dim[c.seq_axis].ClearField("dim_value")
    rest_output.type.tensor_type.shape.dim[c.seq_axis].ClearField("dim_param")
    graph.output.append(rest_output)

    # present_*_pivot = Identity(past_*_pivot): exact passthrough, every
    # step -- the pivot tokens are set once and never revised.
    identity_node = onnx.helper.make_node(
        "Identity",
        [pivot_past_name],
        [pivot_present_name],
        name=_unique_name(f"{c.present_name}_intactkv_pivot", taken_names),
    )
    pivot_output = onnx.ValueInfoProto()
    pivot_output.name = pivot_present_name
    pivot_output.type.CopyFrom(pivot_input.type)
    graph.output.append(pivot_output)

    # present_* (original name/binding, untouched) = Concat(pivot, rest):
    # every original consumer of present_* (the attention math) keeps
    # referring to it by that same name, so nothing downstream needs
    # rewiring.
    reconstruct_node = onnx.helper.make_node(
        "Concat",
        [pivot_present_name, rest_present_name],
        [c.present_name],
        name=_unique_name(f"{c.present_name}_intactkv_reconstruct", taken_names),
        axis=c.seq_axis,
    )

    concat_idx = next(i for i, n in enumerate(graph.node) if n is c.concat_node)
    graph.node.insert(concat_idx + 1, identity_node)
    graph.node.insert(concat_idx + 2, reconstruct_node)
