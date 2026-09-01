"""Reconstructs an ONNX graph *and* its weights directly from a GGUF LLM
checkpoint, for a small set of recognized decoder-only transformer
architectures:

- The Llama family (Llama/Llama2/Llama3, Mistral, and Qwen2), which all
  share the same block shape -- RMSNorm, rotary position embeddings,
  grouped-query attention, a per-layer feed-forward block that is either a
  plain SwiGLU FFN or, for a Mixtral-style checkpoint, a
  ``com.microsoft.MoE`` node -- differing only in things
  :func:`read_gguf_metadata` already reports: head counts, RoPE base,
  whether q/k/v projections carry a bias, whether the LM head is tied to
  the token embedding, and whether ``expert_count`` marks the checkpoint as
  MoE. Built by :func:`_reconstruct_llama_family`.
- ``gpt-oss`` (OpenAI's gpt-oss-20b/120b), a genuinely different
  architecture with no shared block shape with the family above: YaRN-
  scaled RoPE, alternating sliding-window/full attention layers with a
  learned per-head "attention sink" folded into the softmax denominator,
  and an always-MoE FFN using ``com.microsoft.MoE``'s
  ``activation_type="swiglu"``/``swiglu_fusion=1`` reference decomposition
  (gate/up fused from llama.cpp's own separate ``ffn_gate_exps``/
  ``ffn_up_exps`` tensors at graph-build time). Built by
  :func:`_reconstruct_gpt_oss`; see that function's and
  :func:`_gpt_oss_attention_block`'s/:func:`_gpt_oss_moe_ffn`'s own
  docstrings for exactly which llama.cpp source (a real, offline
  ``src/models/openai-moe.cpp`` checkout) confirmed every detail.

Mixtral-style MoE: llama.cpp gives a Mixtral checkpoint the same
``general.architecture`` ("llama") as any dense Llama checkpoint --
``expert_count`` > 0 is what actually switches its own graph builder
(``src/models/llama.cpp``) onto the MoE branch, so this module does the
same, dispatching to :func:`_moe_ffn`'s ``com.microsoft.MoE`` node (the
reference decomposition registered for that schema in
``onnxsim/contrib_schemas.cpp``) instead of the plain FFN. See
:func:`_moe_ffn`'s own docstring for exactly which llama.cpp source
confirmed this maps onto that decomposition unchanged. ``gpt-oss`` is
always MoE (``general.architecture`` alone identifies it, no
``expert_count`` sniffing needed) and always uses the ``swiglu`` activation
rather than Mixtral's ``silu``+separate-``fc3`` -- see
:func:`_gpt_oss_moe_ffn`'s own docstring for why these are mathematically
compatible with the same schema's gating convention despite gpt-oss's own
gating function being nominally different on paper.

This is deliberately the "known architecture template" approach, not a
generic GGUF/ggml graph-structure reconstructor: vLLM and SGLang's own GGUF
support works the same way (match ``general.architecture`` against a
maintained, hand-written model implementation; fail clearly -- "architecture
X is not supported yet" -- rather than guess when it isn't recognized), and
their own issue trackers show that's a deliberate, load-bearing choice, not
a shortcut: coupling to a generic-but-unstable source (llama.cpp's internal,
non-public compute-graph construction, revised on nearly every commit) is a
worse tradeoff than a curated, explicit template per architecture family.

:func:`read_gguf_metadata` supplies everything this needs to know -- which
architecture, its hyperparameters, and its tensors' names/shapes -- without
reading any tensor byte data; :func:`import_gguf_weights` (reused here
unmodified) supplies the actual values, including its existing K-quant
(Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0), legacy (Q4_0/Q4_1/Q5_0/Q5_1), and MXFP4
decode.

Scope note on shapes: the returned graph's ``batch_size``/``seq_len`` are
concrete, caller-chosen static dimensions, not dynamic axes. Real llama.cpp
inference builds a *different* concrete compute graph per call anyway (see
this feature's design discussion for why a cache-free, single-shape forward
graph is the right starting point at all) -- generalizing this to dynamic
axes, and to KV-cache-aware incremental decoding, is future work, not
something this first slice claims to solve.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.onnx_simplifier import import_gguf_weights, read_gguf_metadata

# Opset 17: modern enough for everything this builder needs (Trilu has been
# available since opset 14), while ReduceMean's reduction axes are still an
# *attribute* rather than an input (that migration happened at opset 18) --
# picking 17 avoids threading yet another small int64 constant through every
# RMSNorm call for no benefit here.
_OPSET = 17
# The IR version opset 17 actually requires (see onnx.helper.VERSION_TABLE --
# opset 17 first shipped in onnx 1.12.0, IR version 8), NOT
# onnx.IR_VERSION/onnx.helper.make_model's own default, which is whatever IR
# version the *installed* onnx package's newest opset needs regardless of
# what opset_imports says -- setting it that way declares a model far newer
# than this graph's actual opset, which older onnxruntime builds (bundled
# with e.g. a cross-compiled wheel's smoke test) then refuse to load at all.
_IR_VERSION = 8

# Mirrors onnxsim/gguf_dtype.h's GgmlType enum and ToOnnx/IsKQuant/IsMxfp4
# mapping -- duplicated here rather than exposed through a new C++ binding,
# the same choice tests/test_import_gguf_weights.py already made for its own
# small, stable GGML_TYPE_* constants. See gguf_dtype.h's file comment: GGML
# never reassigns an existing type ID, so this mapping does not drift.
_GGML_RAW_TO_ONNX = {
    0: onnx.TensorProto.FLOAT,  # F32
    1: onnx.TensorProto.FLOAT16,  # F16
    24: onnx.TensorProto.INT8,  # I8
    25: onnx.TensorProto.INT16,  # I16
    26: onnx.TensorProto.INT32,  # I32
    27: onnx.TensorProto.INT64,  # I64
    28: onnx.TensorProto.DOUBLE,  # F64
    30: onnx.TensorProto.BFLOAT16,  # BF16
}
# Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 (K-quant), Q4_0, Q4_1, Q5_0, Q5_1
# (legacy), MXFP4 -- import_gguf_weights forces these to FLOAT regardless of
# what the initializer previously declared (see tensor_pool_gguf_bridge.h's
# HydrateTensorProtoFromGGUF), so that is what must be declared here too.
_GGML_KQUANT_TYPES = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 39}

_ONNX_DTYPE_ITEMSIZE = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
}

# general.architecture values this builder recognizes. "llama"/"qwen2"/
# "mistral" share the same Llama-family block shape (RMSNorm/RoPE/GQA/
# SwiGLU) and are built by _reconstruct_llama_family; "gpt-oss" is a
# distinct architecture (YaRN-scaled RoPE, alternating sliding-window/full
# attention with attention sinks, and a swiglu_fusion=1 MoE FFN) built by
# _reconstruct_gpt_oss -- see the module docstring and each function's own
# docstring.
_SUPPORTED_ARCHITECTURES = ("llama", "qwen2", "mistral", "gpt-oss")


class UnsupportedArchitectureError(NotImplementedError):
    """Raised for a GGUF checkpoint whose ``general.architecture`` (or a
    quantization format among the tensors this graph needs) this builder
    does not have a template for -- mirrors vLLM/SGLang's own "architecture
    X is not supported yet" failure mode: fail clearly rather than guess."""


class _Builder:
    """Accumulates nodes/initializers for one ONNX graph. Not reusable
    across graphs -- one instance per :func:`reconstruct_gguf_graph` call."""

    def __init__(self):
        self.nodes: List[onnx.NodeProto] = []
        self.initializers: List[onnx.TensorProto] = []
        self._counter = 0
        self._const_cache: Dict[Tuple, str] = {}

    def _name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}.{self._counter}"

    def op(self, op_type: str, inputs: List[str], prefix: str, **attrs) -> str:
        out = self._name(prefix)
        self.nodes.append(onnx.helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def placeholder_weight(self, name: str, shape: List[int], onnx_dtype: int) -> None:
        """A zero-filled initializer with `name`'s exact GGUF-reported shape
        and the ONNX dtype import_gguf_weights will actually write into it
        (see _GGML_RAW_TO_ONNX/_GGML_KQUANT_TYPES) -- its *values* come from
        import_gguf_weights right after the graph this builds is assembled,
        but its declared dims/data_type must already be correct: hydration
        overwrites raw_data only (see tensor_pool_gguf_bridge.h's
        HydrateTensorProto/HydrateTensorProtoFromGGUF), never dims, and
        never data_type on the plain-raw-dtype path."""
        nbytes = _ONNX_DTYPE_ITEMSIZE[onnx_dtype]
        for d in shape:
            nbytes *= d
        t = onnx.helper.make_tensor(
            name, onnx_dtype, shape, vals=b"\x00" * nbytes, raw=True
        )
        self.initializers.append(t)

    def const(self, array: np.ndarray, prefix: str = "const") -> str:
        """A constant initializer from a numpy array this builder computed
        itself (RoPE's inv_freq, a causal mask, a reshape's target shape,
        ...) -- distinct from placeholder_weight, whose values come from the
        GGUF file, not from Python. Small integer-shape constants (Reshape
        targets, Slice bounds) recur often enough across layers to dedupe by
        value."""
        key = (array.shape, array.dtype.str, array.tobytes())
        cached = self._const_cache.get(key)
        if cached is not None:
            return cached
        name = self._name(prefix)
        self.initializers.append(onnx.numpy_helper.from_array(array, name=name))
        self._const_cache[key] = name
        return name

    def shape_const(self, dims: List[int]) -> str:
        return self.const(np.array(dims, dtype=np.int64), prefix="shape")


def _unsqueeze(b: _Builder, x: str, axes: List[int], prefix: str) -> str:
    # Unsqueeze's `axes` has been a second *input*, not an attribute, since
    # opset 13 -- unlike Concat's `axis` (always an attribute) or
    # ReduceMean's `axes` (still an attribute until opset 18, which _OPSET
    # predates).
    axes_c = b.const(np.array(axes, dtype=np.int64), prefix="unsqueeze_axes")
    return b.op("Unsqueeze", [x, axes_c], prefix)


def _linear(
    b: _Builder,
    x: str,
    weight_name: str,
    bias_name: Optional[str],
    prefix: str,
) -> str:
    """``x @ weight.T (+ bias)`` -- nn.Linear semantics. `weight_name` is a
    GGUF tensor already declared (via placeholder_weight) with its ORIGINAL
    [out_features, in_features] shape (GGUF/ggml round-trips a PyTorch
    nn.Linear.weight's shape unchanged -- see read_gguf_metadata's own
    dimension-order note), so this transposes at graph-build time via an
    explicit Transpose node rather than pre-transposing the (not-yet-known)
    weight values. A later `onnxsim.simplify()` constant-folds that
    Transpose away for free once the weight is actually hydrated."""
    wt = b.op("Transpose", [weight_name], f"{prefix}.wt", perm=[1, 0])
    out = b.op("MatMul", [x, wt], f"{prefix}.matmul")
    if bias_name is not None:
        out = b.op("Add", [out, bias_name], f"{prefix}.bias")
    return out


def _rmsnorm(b: _Builder, x: str, weight_name: str, eps: float, prefix: str) -> str:
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="eps")
    x2 = b.op("Mul", [x, x], f"{prefix}.sq")
    mean = b.op("ReduceMean", [x2], f"{prefix}.mean", axes=[-1], keepdims=1)
    var_eps = b.op("Add", [mean, eps_c], f"{prefix}.var_eps")
    rms = b.op("Sqrt", [var_eps], f"{prefix}.rms")
    normed = b.op("Div", [x, rms], f"{prefix}.normed")
    return b.op("Mul", [normed, weight_name], f"{prefix}.scaled")


def _slice_last_dim(b: _Builder, x: str, start: int, end: int, prefix: str) -> str:
    starts = b.const(np.array([start], dtype=np.int64), prefix="slice_start")
    ends = b.const(np.array([end], dtype=np.int64), prefix="slice_end")
    axes = b.const(np.array([-1], dtype=np.int64), prefix="slice_axis")
    return b.op("Slice", [x, starts, ends, axes], prefix)


def _rotate_half(b: _Builder, x: str, head_dim: int, prefix: str) -> str:
    half = head_dim // 2
    x1 = _slice_last_dim(b, x, 0, half, f"{prefix}.x1")
    x2 = _slice_last_dim(b, x, half, head_dim, f"{prefix}.x2")
    neg_x2 = b.op("Neg", [x2], f"{prefix}.negx2")
    return b.op("Concat", [neg_x2, x1], prefix, axis=-1)


def _apply_rope(
    b: _Builder, x: str, cos: str, sin: str, head_dim: int, prefix: str
) -> str:
    rotated = _rotate_half(b, x, head_dim, f"{prefix}.rot")
    a = b.op("Mul", [x, cos], f"{prefix}.a")
    c = b.op("Mul", [rotated, sin], f"{prefix}.c")
    return b.op("Add", [a, c], prefix)


def _moe_ffn(
    b: _Builder,
    h: str,
    p: str,
    n_embd: int,
    n_ff: int,
    n_expert: int,
    n_expert_used: int,
    declare,
) -> str:
    """A Mixtral-style MoE feed-forward block, as one ``com.microsoft.MoE``
    node -- the reference decomposition registered for that schema in
    ``onnxsim/contrib_schemas.cpp`` (``fc2(silu(fc1(x)) * fc3(x))``, softmax-
    over-all-experts routing, top-k, renormalized) is byte-for-byte the same
    computation llama.cpp's own ``build_moe_ffn`` performs for this exact
    call shape: gating_op=SOFTMAX, norm_w=true, no expert-selection bias, no
    expert groups, and no weight scale (validated by the caller) --
    confirmed by reading ``llama.cpp/src/models/llama.cpp``'s MoE branch and
    ``llama.cpp/src/llama-graph.cpp``'s ``build_moe_ffn`` directly, not
    inferred from the schema docs alone. ``fc1``=``ffn_gate_exps`` (gate),
    ``fc3``=``ffn_up_exps`` (up, no activation), ``fc2``=``ffn_down_exps``
    (down) -- the same naming ``BuildMoEFunctionBody``'s own comment uses.

    ``ffn_gate_exps``/``ffn_up_exps``/``ffn_down_exps``'s GGML shapes
    (``{n_embd, n_ff, n_expert}``/``{n_embd, n_ff, n_expert}``/
    ``{n_ff, n_embd, n_expert}``) reverse -- via ``declare``'s existing,
    unmodified rule -- directly onto ``com.microsoft.MoE``'s own
    ``fc1_experts_weights``/``fc3_experts_weights``/``fc2_experts_weights``
    layouts (``[num_experts, inter_size, hidden_size]``/same/
    ``[num_experts, hidden_size, inter_size]``); no MoE-specific shape
    handling is needed here beyond calling ``declare`` with those shapes.
    """
    router_w = declare(f"{p}.ffn_gate_inp.weight", [n_expert, n_embd])
    logits = _linear(b, h, router_w, None, f"{p}.moe_router")
    router_probs = b.op(
        "Reshape",
        [logits, b.shape_const([-1, n_expert])],
        f"{p}.moe_router_flat",
    )

    gate_w = declare(f"{p}.ffn_gate_exps.weight", [n_expert, n_ff, n_embd])
    up_w = declare(f"{p}.ffn_up_exps.weight", [n_expert, n_ff, n_embd])
    down_w = declare(f"{p}.ffn_down_exps.weight", [n_expert, n_embd, n_ff])

    return b.op(
        "MoE",
        [h, router_probs, gate_w, "", down_w, "", up_w],
        f"{p}.moe",
        domain="com.microsoft",
        k=n_expert_used,
        activation_type="silu",
        normalize_routing_weights=1,
    )


# ---------------------------------------------------------------------------
# gpt-oss attention block -- standalone, NOT wired into
# _reconstruct_llama_family, _SUPPORTED_ARCHITECTURES, or
# reconstruct_gguf_graph's dispatch. Produced independently of this module's
# MoE/FFN half (see _moe_ffn/_reconstruct_llama_family's own FFN branch,
# which a sibling change handles for gpt-oss's Mixture-of-Experts FFN) so
# the two can be reviewed/wired in separately.
#
# Verified directly against a local llama.cpp clone (the commit checked out
# under /home/user/llama.cpp at the time this was written) -- specifically:
#   - src/models/openai-moe.cpp: llama_model_openai_moe's
#     load_arch_hparams/load_arch_tensors/graph ctor -- the actual gpt-oss
#     graph builder and tensor shapes/names.
#   - src/llama-graph.cpp: build_attn_mha (the shared softmax+sinks kernel
#     every build_attn_* variant calls into) and
#     llm_graph_input_attn_no_cache::set_input (the causal/SWA mask fill,
#     "fill_mask" lambda).
#   - src/llama-hparams.cpp/.h: set_swa_pattern (the alternating-layer
#     pattern), is_masked_swa's LLAMA_SWA_TYPE_STANDARD case (the banded
#     window rule).
#   - src/llama-arch.cpp: LLM_ARCH_OPENAI_MOE's tensor-name map
#     (LLM_TENSOR_ATTN_SINKS -> "blk.%d.attn_sinks") and its LLM_KV_*
#     metadata key strings (attention.sliding_window[_pattern],
#     attention.key_length/value_length, rope.scaling.*); and
#     src/llama-model.cpp's rope-type classification switch, which puts
#     LLM_ARCH_OPENAI_MOE in the "pairs of head values are offset by
#     n_rot/2" (LLAMA_ROPE_TYPE_NEOX) group -- the same rotate-half
#     convention _rotate_half/_apply_rope above already implement, so gpt-oss
#     needs no different rotation *shape*, only a different theta/scale
#     schedule (YaRN) and mask/sink treatment.
#   - src/llama-context.cpp (llama_context's cparams derivation -- how
#     yarn_ext_factor/yarn_attn_factor fall out of rope_scaling_type==yarn)
#     and ggml/src/ggml.c + ggml/src/ggml-cpu/ops.cpp
#     (ggml_rope_yarn_corr_dims, rope_yarn/ggml_rope_cache_init, and
#     ggml_compute_forward_soft_max_f32's sink handling) for the exact
#     numeric formulas transcribed below.
#   - conversion/gpt_oss.py + conversion/base.py: what a real gpt-oss
#     checkpoint's HF-to-GGUF conversion actually *writes* -- confirms which
#     of the generic mechanisms above this specific architecture actually
#     exercises (e.g. that no real gpt-oss GGUF sets a SWA-specific RoPE
#     freq_base override or a non-default sliding_window_pattern).
# and openai/gpt-oss-20b's own config.json on the Hugging Face Hub (fetched
# directly for this work, not recalled from memory) for the concrete
# hyperparameter values cited in the comments below (head_dim=64 vs
# embedding_length/head_count=2880/64=45 -- i.e. head_dim is NOT
# embedding_length/head_count for this architecture, unlike every
# _SUPPORTED_ARCHITECTURES entry today; layer_types alternating
# sliding_attention/full_attention starting at layer 0; sliding_window=128;
# rope_scaling type "yarn" with factor=32, beta_fast=32, beta_slow=1,
# original_max_position_embeddings=4096; rope_theta=150000).
#
# Corrections to the brief this was scoped from:
#   - The attention-sink formula is NOT "append a per-head sink logit as an
#     extra column, then drop that column's weight before the value
#     matmul". It is folded into the softmax denominator only, and a sink
#     "column" is never materialized against V at all -- see
#     _gpt_oss_attention_block's own docstring for the exact kernel-level
#     formula this was read off of.
#   - "Possibly a distinct RoPE scaling scheme" is confirmed, not merely
#     possible: gpt-oss-20b's real config.json uses YaRN (rope_scaling.type
#     == "yarn"), not a bare freq_base RoPE -- see _gpt_oss_yarn_cos_sin.
#   - RoPE's own freq_base/freq_scale are NOT windowed per sliding-vs-full
#     layer for gpt-oss specifically (only the attention mask is) -- see
#     _gpt_oss_attention_block's docstring for why, so a single cos/sin pair
#     is computed once and reused across every layer, exactly like
#     _reconstruct_llama_family already does for the plain-RoPE case.
# ---------------------------------------------------------------------------


def _gpt_oss_is_sliding_layer(layer_idx: int, swa_period: int) -> bool:
    """Mirrors llama-hparams.cpp's ``set_swa_pattern(n_pattern,
    dense_first=false)`` -- the default gpt-oss always uses (see
    src/models/openai-moe.cpp's ``load_arch_hparams``, which never passes
    ``dense_first=true``)::

        is_swa_impl[il] = n_pattern == 0 || (il % n_pattern < (n_pattern - 1))

    For gpt-oss's period of 2 (the hardcoded default in
    ``load_arch_hparams`` -- no real gpt-oss GGUF sets
    ``{arch}.attention.sliding_window_pattern``, since
    ``conversion/gpt_oss.py`` never calls ``add_sliding_window_pattern``),
    this alternates sliding/local at even layer indices (0, 2, 4, ...) and
    full/global at odd ones -- confirmed against openai/gpt-oss-20b's own
    ``config.json`` on the HF Hub, whose ``layer_types`` list literally
    starts ``["sliding_attention", "full_attention", ...]``.
    """
    return swa_period == 0 or (layer_idx % swa_period) < (swa_period - 1)


def _gpt_oss_attn_mask(seq_len: int, sliding: bool, n_swa: int) -> np.ndarray:
    """A causal (``sliding=False``) or banded causal-plus-sliding-window
    (``sliding=True``) additive attention mask, ``[seq_len, seq_len]``
    float32, ``-1e9`` where disallowed and ``0`` where allowed.

    Mirrors ``llm_graph_input_attn_no_cache::set_input``'s ``fill_mask``
    lambda in llama-graph.cpp (gpt-oss sets no ALiBi, so that lambda's
    ``hparams.use_alibi ? -abs(p0-p1) : 0.0f`` collapses to plain ``0.0f``
    for every allowed cell) combined with
    ``llama_hparams::is_masked_swa``'s ``LLAMA_SWA_TYPE_STANDARD`` case --
    the SWA type gpt-oss's own ``load_arch_hparams`` hardcodes
    (``hparams.swa_type = LLAMA_SWA_TYPE_STANDARD``) -- which masks a
    (key pos p0, query pos p1) pair when ``p1 - p0 >= n_swa``, i.e. allows
    exactly the ``n_swa`` most recent key positions up to and including the
    query position itself.
    """
    q_pos = np.arange(seq_len)[:, None]
    k_pos = np.arange(seq_len)[None, :]
    diff = q_pos - k_pos
    allowed = diff >= 0
    if sliding:
        allowed = allowed & (diff < n_swa)
    return np.where(allowed, 0.0, -1e9).astype(np.float32)


def _gpt_oss_yarn_cos_sin(
    b: _Builder,
    position_ids: str,
    head_dim: int,
    freq_base: float,
    yarn_factor: float,
    yarn_orig_ctx: float,
    yarn_beta_fast: float,
    yarn_beta_slow: float,
    prefix: str,
) -> Tuple[str, str]:
    """YaRN-scaled RoPE cos/sin, each ``[batch, 1, seq, head_dim]`` --
    transcribed from ggml/src/ggml.c's ``ggml_rope_yarn_corr_dims`` (the
    ``low``/``high`` correction-dimension bounds) and
    ggml/src/ggml-cpu/ops.cpp's ``rope_yarn``/``ggml_rope_cache_init`` (the
    per-frequency-bin ramp mix and magnitude scale), with the
    context-parameter-level ``yarn_attn_factor`` derivation in
    src/llama-context.cpp folded in algebraically: that code computes a
    ``cparams``-level attention-scale factor specifically so it cancels
    back out against ``rope_yarn``'s own
    ``mscale *= 1 + 0.1*log(1/freq_scale)`` for every model except
    DeepSeek-V2 (which sets a nonzero ``rope_yarn_log_mul`` -- gpt-oss does
    not), so the *net* magnitude scale actually reaching gpt-oss's RoPE is
    exactly the textbook YaRN ``mscale`` below, not the intermediate value
    either source file computes standalone.

    ``ext_factor`` (how much of the ramp mix applies) is only ever 1.0 or
    0.0 in llama.cpp when not explicitly overridden by a caller
    (``cparams.yarn_ext_factor = rope_scaling_type == YARN ? 1.0f : 0.0f``,
    src/llama-context.cpp) -- nothing in gpt-oss's own graph/model code
    overrides it -- so this only implements that fixed ext_factor=1 case
    (the one a real gpt-oss GGUF's ``rope.scaling.type=yarn`` metadata
    always selects). It degenerates correctly to plain RoPE when
    ``yarn_factor<=1.0``: freq_scale becomes 1.0 so the interpolated and
    extrapolated angles coincide regardless of the ramp, and mscale's
    ``scale<=1`` branch is exactly 1.0 -- matching every
    _SUPPORTED_ARCHITECTURES checkpoint's plain-RoPE behavior today.

    Only full rotary (``n_rot == head_dim``) is implemented, matching
    gpt-oss's own tensor shapes (no partial-rotary GGUF key is read for
    it) and this module's existing partial-rotary rejection elsewhere.
    """
    half = head_dim // 2
    freq_idx = np.arange(half, dtype=np.float64)
    inv_freq = 1.0 / (freq_base ** (2.0 * freq_idx / head_dim))

    def corr_dim(n_rot: float) -> float:
        # ggml_rope_yarn_corr_dim: n_dims * log(n_ctx_orig / (n_rot * 2pi))
        # / (2 * log(base)) -- n_dims is head_dim here (full rotary).
        return (
            head_dim
            * math.log(yarn_orig_ctx / (n_rot * 2.0 * math.pi))
            / (2.0 * math.log(freq_base))
        )

    low = max(0.0, math.floor(corr_dim(yarn_beta_fast)))
    high = min(float(head_dim - 1), math.ceil(corr_dim(yarn_beta_slow)))
    # rope_yarn_ramp(low, high, i0), evaluated at i0 = 2*freq_idx (ggml's
    # cache-fill loop steps i0 by 2, and only ever calls it with i0/2 ==
    # freq_idx): ramp==1 fully "extrapolated" (short-range, low-freq-index
    # dims), ramp==0 fully "interpolated" (long-range, high-freq-index
    # dims).
    ramp = 1.0 - np.clip((freq_idx - low) / max(1e-3, high - low), 0.0, 1.0)

    freq_scale = 1.0 / yarn_factor
    # theta(j) = pos * inv_freq[j] * weight[j], where weight[j] blends the
    # scaled ("interpolated") and unscaled ("extrapolated") angle per
    # rope_yarn's `theta = theta_interp*(1-ramp_mix) + theta_extrap*ramp_mix`
    # (ramp_mix == ramp here since ext_factor == 1) -- algebraically the
    # same theta since theta_interp/theta_extrap both factor as
    # pos*inv_freq[j]*{freq_scale,1}.
    weight = freq_scale * (1.0 - ramp) + ramp
    # rope_yarn's `mscale *= 1 + 0.1*log(1/freq_scale)`, applied only when
    # ext_factor != 0 (true here) -- this IS the net mscale reaching
    # gpt-oss's RoPE; see this function's own docstring for why the
    # cparams-level factor llama-context.cpp separately computes is not
    # double-counted here.
    mscale = 1.0 if yarn_factor <= 1.0 else 0.1 * math.log(yarn_factor) + 1.0

    inv_freq_scaled_c = b.const(
        (inv_freq * weight).reshape(1, 1, -1).astype(np.float32),
        prefix=f"{prefix}.inv_freq",
    )
    mscale_c = b.const(np.array(mscale, dtype=np.float32), prefix=f"{prefix}.mscale")

    pos_f = b.op("Cast", [position_ids], f"{prefix}.pos_f", to=onnx.TensorProto.FLOAT)
    pos_unsq = _unsqueeze(b, pos_f, [-1], f"{prefix}.pos_unsq")
    freqs = b.op("Mul", [pos_unsq, inv_freq_scaled_c], f"{prefix}.freqs")
    emb = b.op("Concat", [freqs, freqs], f"{prefix}.emb", axis=-1)
    cos_raw = b.op("Cos", [emb], f"{prefix}.cos_raw")
    sin_raw = b.op("Sin", [emb], f"{prefix}.sin_raw")
    cos = b.op("Mul", [cos_raw, mscale_c], f"{prefix}.cos")
    sin = b.op("Mul", [sin_raw, mscale_c], f"{prefix}.sin")
    cos_b = _unsqueeze(b, cos, [1], f"{prefix}.cos_b")
    sin_b = _unsqueeze(b, sin, [1], f"{prefix}.sin_b")
    return cos_b, sin_b


def _gpt_oss_attention_block(
    b: _Builder,
    x: str,
    p: str,
    layer_idx: int,
    n_embd: int,
    n_head: int,
    n_head_kv: int,
    head_dim: int,
    cos_b: str,
    sin_b: str,
    sliding_window: int,
    swa_period: int,
    batch_size: int,
    seq_len: int,
    eps: float,
    declare,
    declare_optional,
) -> str:
    """One gpt-oss transformer layer's attention sub-block: RMSNorm through
    the post-output-projection residual add. Structurally mirrors
    ``_reconstruct_llama_family``'s plain-GQA attention block (lines
    ~458-524 as of this writing), reusing the same helpers
    (``_rmsnorm``/``_linear``/``_apply_rope``/``_unsqueeze``), but differs
    in three ways gpt-oss's own tensors/graph require:

    1. Q/K/V/O projection shapes are NOT derived from
       ``n_embd``/``n_head`` the way every _SUPPORTED_ARCHITECTURES entry's
       shapes are today. gpt-oss's own ``head_dim`` is an independent GGUF
       metadata value (``{arch}.attention.key_length`` /
       ``.value_length``, generically read the same as any other
       architecture that sets it -- see conversion/base.py writing both
       from HF's ``head_dim`` config key) that is NOT
       ``embedding_length // head_count``: gpt-oss-20b's own config has
       ``head_dim=64`` while ``embedding_length/head_count = 2880/64 =
       45`` -- a different, smaller number. So ``attn_q.weight``'s GGUF
       shape is ``[n_head*head_dim, n_embd]`` (reversed to the ONNX-shape
       ``[n_head*head_dim, n_embd]`` -- see ``declare``'s own note; here
       "reversed" changes nothing since it's still 2-D) rather than
       ``[n_embd, n_embd]``, and ``attn_output.weight`` is
       ``[n_embd, n_head*head_dim]`` rather than ``[n_embd, n_embd]`` --
       see src/llama-model.cpp's ``create_tensor_qkv``/
       src/models/openai-moe.cpp's ``layer.wo`` construction.

    2. The additive attention mask alternates between a plain causal mask
       (odd ``layer_idx``) and a banded causal-plus-sliding-window mask of
       width ``sliding_window`` (even ``layer_idx``) -- see
       ``_gpt_oss_is_sliding_layer``/``_gpt_oss_attn_mask`` for the exact
       llama.cpp source this reproduces. ``cos_b``/``sin_b`` are NOT
       recomputed per layer type: gpt-oss copies its SWA-layer RoPE
       freq_base/freq_scale from the non-SWA values before ever reading an
       optional per-layer override key (src/models/openai-moe.cpp's
       ``load_arch_hparams``), and no real gpt-oss GGUF sets that override
       (``conversion/gpt_oss.py`` never writes
       ``{arch}.rope.freq_base_swa``) -- so the caller computes ONE
       ``cos_b``/``sin_b`` pair (via ``_gpt_oss_yarn_cos_sin``) and passes
       it in for every layer, exactly like
       ``_reconstruct_llama_family`` already does for the plain case.

    3. Softmax gets an extra per-Q-head "attention sink" term folded into
       its denominator. Per-head sink values live in GGUF tensor
       ``{p}.attn_sinks.weight`` (note the ``.weight`` suffix on what is
       actually a plain learned vector, not a matrix -- see
       ``conversion/gpt_oss.py``'s ``filter_tensors`` explicitly appending
       it, and src/llama-arch.cpp's ``LLM_TENSOR_ATTN_SINKS`` ->
       ``"blk.%d.attn_sinks"`` map), shape ``[n_head]`` (one scalar per
       Q head, reshaped here to broadcast against the ``[n_head_kv,
       n_rep]``-split GQA score tensor the same way this module's existing
       broadcasting-GQA scheme already splits Q's head axis). The exact
       formula, read off ggml/src/ggml-cpu/ops.cpp's
       ``ggml_compute_forward_soft_max_f32``'s sink branch (reached via
       src/llama-graph.cpp's ``build_attn_mha`` ->
       ``ggml_soft_max_add_sinks``): for each row of *masked, scaled*
       logits ``s`` and that row's head's sink scalar ``k``,
       ``m = max(max(s), k)``; ``softmax_i = exp(s_i - m) / (sum_j
       exp(s_j - m) + exp(k - m))``. The sink term participates in the
       row max and the normalizing sum ONLY -- it is never concatenated as
       an actual extra key/value position and never contributes to the
       value-weighted sum (``dp``, the kernel's actual softmax-weight
       output array, has exactly ``n_kv`` entries; the sink's
       ``exp(k-m)`` term only ever appears added into the scalar
       denominator each of those entries is divided by). This corrects
       the "append a column, then drop its weight before the value
       matmul" framing this was scoped from -- no column is ever
       materialized against V at all.
    """
    resid = x
    h = _rmsnorm(
        b, x, declare(f"{p}.attn_norm.weight", [n_embd]), eps, f"{p}.attn_norm"
    )

    n_embd_q = n_head * head_dim
    n_embd_kv = n_head_kv * head_dim
    q = _linear(
        b,
        h,
        declare(f"{p}.attn_q.weight", [n_embd_q, n_embd]),
        declare_optional(f"{p}.attn_q.bias", [n_embd_q]),
        f"{p}.q_proj",
    )
    k = _linear(
        b,
        h,
        declare(f"{p}.attn_k.weight", [n_embd_kv, n_embd]),
        declare_optional(f"{p}.attn_k.bias", [n_embd_kv]),
        f"{p}.k_proj",
    )
    v = _linear(
        b,
        h,
        declare(f"{p}.attn_v.weight", [n_embd_kv, n_embd]),
        declare_optional(f"{p}.attn_v.bias", [n_embd_kv]),
        f"{p}.v_proj",
    )

    def reshape(t: str, dims: List[int], prefix: str) -> str:
        return b.op("Reshape", [t, b.shape_const(dims)], prefix)

    q = reshape(q, [batch_size, seq_len, n_head, head_dim], f"{p}.q_r")
    q = b.op("Transpose", [q], f"{p}.q_t", perm=[0, 2, 1, 3])
    k = reshape(k, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.k_r")
    k = b.op("Transpose", [k], f"{p}.k_t", perm=[0, 2, 1, 3])
    v = reshape(v, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.v_r")
    v = b.op("Transpose", [v], f"{p}.v_t", perm=[0, 2, 1, 3])

    q = _apply_rope(b, q, cos_b, sin_b, head_dim, f"{p}.q_rope")
    k = _apply_rope(b, k, cos_b, sin_b, head_dim, f"{p}.k_rope")

    # Grouped-query attention via broadcasting -- identical scheme to
    # _reconstruct_llama_family's (see that function's own comment).
    n_rep = n_head // n_head_kv
    q5 = reshape(q, [batch_size, n_head_kv, n_rep, seq_len, head_dim], f"{p}.q5")
    k5 = _unsqueeze(b, k, [2], f"{p}.k5")
    v5 = _unsqueeze(b, v, [2], f"{p}.v5")

    k5t = b.op("Transpose", [k5], f"{p}.k5t", perm=[0, 1, 2, 4, 3])
    scores = b.op("MatMul", [q5, k5t], f"{p}.scores")
    inv_sqrt_d = b.const(
        np.array(1.0 / math.sqrt(head_dim), dtype=np.float32),
        prefix=f"{p}.inv_sqrt_d",
    )
    scores = b.op("Mul", [scores, inv_sqrt_d], f"{p}.scores_scaled")

    sliding = _gpt_oss_is_sliding_layer(layer_idx, swa_period)
    mask_c = b.const(
        _gpt_oss_attn_mask(seq_len, sliding, sliding_window),
        prefix=f"{p}.attn_mask",
    )
    scores = b.op("Add", [scores, mask_c], f"{p}.scores_masked")

    # Attention sinks: fold a per-head learned scalar into the softmax
    # denominator only (see this function's own docstring, point 3, for
    # the exact formula/citation).
    sinks = declare(f"{p}.attn_sinks.weight", [n_head])
    sinks_b = reshape(sinks, [1, n_head_kv, n_rep, 1, 1], f"{p}.sinks_b")

    row_max = b.op("ReduceMax", [scores], f"{p}.row_max", axes=[-1], keepdims=1)
    row_max = b.op("Max", [row_max, sinks_b], f"{p}.row_max_sinks")
    shifted = b.op("Sub", [scores, row_max], f"{p}.shifted")
    exp_shifted = b.op("Exp", [shifted], f"{p}.exp_shifted")
    # Unlike ReduceMax/ReduceMean, ReduceSum's `axes` has been a second
    # *input* (not an attribute) since opset 13 -- see _unsqueeze's own
    # comment on the analogous Unsqueeze migration.
    reduce_axes_c = b.const(np.array([-1], dtype=np.int64), prefix=f"{p}.reduce_axes")
    sum_exp = b.op(
        "ReduceSum", [exp_shifted, reduce_axes_c], f"{p}.sum_exp", keepdims=1
    )
    sink_shifted = b.op("Sub", [sinks_b, row_max], f"{p}.sink_shifted")
    sink_exp = b.op("Exp", [sink_shifted], f"{p}.sink_exp")
    denom = b.op("Add", [sum_exp, sink_exp], f"{p}.denom")
    attn = b.op("Div", [exp_shifted, denom], f"{p}.attn")

    out5 = b.op("MatMul", [attn, v5], f"{p}.attn_out5")
    out = reshape(out5, [batch_size, n_head, seq_len, head_dim], f"{p}.out_r")
    out = b.op("Transpose", [out], f"{p}.out_t", perm=[0, 2, 1, 3])
    out = reshape(out, [batch_size, seq_len, n_embd_q], f"{p}.out_flat")
    out = _linear(
        b,
        out,
        declare(f"{p}.attn_output.weight", [n_embd, n_embd_q]),
        declare_optional(f"{p}.attn_output.bias", [n_embd]),
        f"{p}.o_proj",
    )
    return b.op("Add", [resid, out], f"{p}.attn_resid")


def _interleave_gate_up(
    b: _Builder,
    gate: str,
    up: str,
    n_expert: int,
    n_ff: int,
    trailing: List[int],
    prefix: str,
) -> str:
    """Repacks two separately-stored per-expert tensors (GGUF's own
    ``ffn_gate_exps``/``ffn_up_exps`` convention -- see
    :func:`_gpt_oss_moe_ffn`'s docstring) into the single, column-
    interleaved tensor ``com.microsoft.MoE``'s ``swiglu_fusion=1`` reference
    decomposition expects for ``fc1_experts_weights``/``fc1_experts_bias``:
    row (or, for the 2D bias case, column) ``2*i`` of the fused tensor is
    expert-local row ``i`` of `gate`, ``2*i+1`` is row ``i`` of `up` --
    exactly the ``h1[:, 0::2]``/``h1[:, 1::2]`` de-interleaving
    ``BuildMoEFunctionBody``'s generated function body (see
    ``scripts/codegen/generate_moe_function_templates.py``'s
    ``_swiglu_activation_lines``) undoes at runtime.

    Implemented as ``Reshape`` (insert a size-1 axis right after the
    per-expert row axis) -> ``Concat`` (stack gate/up along that new axis)
    -> ``Reshape`` (flatten the row axis back down, twice as long) rather
    than a strided ``Scatter``/interleave op that ONNX has no direct op
    for. This exact reshape/concat/reshape sequence -- and specifically
    that row-major flattening of a trailing ``(n_ff, 2)`` pair of axes
    really does put `gate`'s row ``i`` at flat row ``2*i`` and `up`'s at
    ``2*i+1``, not the other way around or interleaved some other way --
    was checked two ways before being relied on here: an independent numpy
    ``fused[..., 0::2, ...] = gate; fused[..., 1::2, ...] = up`` reference,
    and a real onnxruntime CPU session running exactly these ops (see
    ``tests/test_gguf_reconstruct.py``'s
    ``test_gate_up_interleave_matches_independent_numpy_reference``).

    `gate`/`up` are both shaped ``[n_expert, n_ff] + trailing`` (`trailing`
    is ``[n_embd]`` for the two weight tensors, ``[]`` for the two 1D-per-
    expert bias tensors) -- `trailing` is a parameter rather than hardcoded
    so this one helper serves both cases identically. Like ``_linear``'s
    weight Transpose, this whole reshape/concat/reshape chain runs on
    initializers only, so a later ``onnxsim.simplify()`` constant-folds it
    away for free once `gate`/`up` are actually hydrated by
    ``import_gguf_weights`` -- it costs nothing at runtime once simplified.
    """
    split_shape = [n_expert, n_ff, 1] + trailing
    fused_shape = [n_expert, 2 * n_ff] + trailing
    g = b.op("Reshape", [gate, b.shape_const(split_shape)], f"{prefix}.g")
    u = b.op("Reshape", [up, b.shape_const(split_shape)], f"{prefix}.u")
    cat = b.op("Concat", [g, u], f"{prefix}.cat", axis=2)
    return b.op("Reshape", [cat, b.shape_const(fused_shape)], prefix)


def _gpt_oss_moe_ffn(
    b: _Builder,
    h: str,
    p: str,
    n_embd: int,
    n_ff: int,
    n_expert: int,
    n_expert_used: int,
    declare,
    declare_optional,
) -> str:
    """A gpt-oss-20b-style MoE feed-forward block, as one
    ``com.microsoft.MoE`` node using the ``activation_type="swiglu"``,
    ``swiglu_fusion=1`` reference decomposition (as opposed to
    :func:`_moe_ffn`'s Mixtral-style ``activation_type="silu"`` + separate
    ``fc3`` one) -- confirmed against a real (offline, sparse-checked-out)
    llama.cpp clone, not inferred from the schema docs alone:

    * Tensor names/shapes: ``llama.cpp/src/models/openai-moe.cpp``'s
      ``load_arch_tensors`` -- gpt-oss stores gate/up as two SEPARATE
      per-expert tensors, ``blk.N.ffn_gate_exps.weight``/
      ``blk.N.ffn_up_exps.weight`` (GGML shape ``{n_embd, n_ff_exp,
      n_expert}`` each, reversing -- via `declare`'s existing rule, same as
      :func:`_moe_ffn` -- onto ``[n_expert, n_ff, n_embd]``), NOT a single
      pre-fused tensor; there is no fused tensor anywhere in the GGUF file.
      Unlike Mixtral, gpt-oss also stores a REQUIRED (not
      ``TENSOR_NOT_REQUIRED`` -- see ``llama-model-loader.h``'s
      ``create_tensor`` flag argument, passed as ``0`` for every one of
      these in ``load_arch_tensors``) bias for every one of router/gate/
      up/down: ``blk.N.ffn_gate_inp.bias`` (``[n_expert]``),
      ``blk.N.ffn_gate_exps.bias``/``blk.N.ffn_up_exps.bias`` (GGML
      ``{n_ff_exp, n_expert}``, reversing to ``[n_expert, n_ff]``), and
      ``blk.N.ffn_down_exps.bias`` (GGML ``{n_embd, n_expert}``, reversing
      to ``[n_expert, n_embd]``) -- declared here via `declare_optional`
      rather than `declare` purely as defense-in-depth against a
      non-standard checkpoint; a real one always has them. Note gpt-oss
      reports its expert intermediate size under a DIFFERENT GGUF key than
      the dense case, ``<arch>.expert_feed_forward_length``
      (``LLM_KV_EXPERT_FEED_FORWARD_LENGTH`` in ``llama-arch.cpp``), not
      ``<arch>.feed_forward_length`` -- this function takes the already-
      resolved `n_ff` as a plain argument, so it is the caller's job (not
      done here, since this function is not wired into any dispatch yet)
      to read the right key. ``general.architecture`` for this family is
      ``"gpt-oss"`` (``LLM_ARCH_OPENAI_MOE`` in ``llama-arch.cpp``).

    * fc1 fusion: ``com.microsoft.MoE``'s ``swiglu_fusion=1`` decomposition
      needs ONE ``fc1_experts_weights`` tensor,
      ``[n_expert, 2*n_ff, n_embd]``, gate/up interleaved row by row (see
      :func:`_interleave_gate_up`'s own docstring for exactly how and how
      that was verified) -- built here as ordinary graph ops from the two
      separate GGUF tensors, matching ``fc1_experts_bias``
      (``[n_expert, 2*n_ff]``) the same way. This is a graph-construction
      convenience, not something llama.cpp's own graph builder does --
      llama.cpp's ``build_moe_ffn`` (``llama-graph.cpp``) keeps gate/up as
      two separate ``mul_mat_id`` calls and calls ``ggml_swiglu_oai(gate,
      up, alpha, limit)`` on the pair directly (see below); the interleave
      exists purely so this maps onto ONNX Runtime's actual CPU MoE kernel,
      whose constructor (``onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc``)
      throws unless ``swiglu_fusion == 1`` for a SwiGLU node. Interleaving
      first and then de-interleaving via the generated function body's own
      strided ``Slice`` is a repacking, not a reordering of any actual
      values, so it changes nothing numerically.

    * Activation constants: ``llama.cpp/src/models/openai-moe.cpp``'s
      ``build_moe_ffn`` call passes ``type_op=LLM_FFN_SWIGLU_OAI_MOE``;
      ``llama-graph.cpp``'s ``LLM_FFN_SWIGLU_OAI_MOE`` case (the per-expert
      branch, ~line 2226) hardcodes ``alpha=1.702f``/``limit=7.0f`` as
      local ``constexpr``s -- NOT read from GGUF metadata, confirming the
      caller's assumption -- and calls ``ggml_swiglu_oai(cur, up, alpha,
      limit)``, whose CPU kernel
      (``ggml/src/ggml-cpu/ops.cpp``'s ``ggml_compute_forward_swiglu_oai_f32``)
      computes ``x = min(gate, limit); y = clamp(up, -limit, limit);
      out = x / (1 + exp(-alpha*x)) * (y + 1.0)`` -- i.e. beta=1.0,
      hardcoded the same way (there is no separate beta parameter/variable
      at all; it's the literal ``+ 1.f`` in that line). This is
      byte-for-byte the same formula
      ``scripts/codegen/generate_moe_function_templates.py``'s
      ``_swiglu_activation_lines`` builds for ``swiglu_fusion=1`` (gate
      clamped with `Min` only, linear/up clamped both ways, same swish*
      (linear+beta) shape) -- so ``activation_alpha=1.702``,
      ``activation_beta=1.0``, ``swiglu_limit=7.0`` are exactly right.

    * Gating: gpt-oss uses ``gating_op=LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT``
      (hardcoded in the ``build_moe_ffn`` call, not a GGUF-configurable
      hparam for this architecture), which ``llama-graph.cpp``'s
      ``build_moe_ffn`` (~line 1997) implements as: skip any softmax over
      *all* experts (``probs = logits`` unchanged), pick the top-k experts
      by raw logit value, gather their raw logits as `weights`, then
      ``ggml_soft_max`` over just those `k` selected values (~line 2077) --
      i.e. softmax is applied AFTER top-k, over only the selected experts'
      logits, not before over all of them. ``norm_w`` (the extra
      sum-and-divide renormalization step, ~line 2082) is passed as
      ``false`` for gpt-oss, but is mathematically a no-op here regardless
      (a softmax's own output already sums to 1, so dividing by that sum
      changes nothing outside of the meaningless case the sum underflows
      the routine's own ``6.103515625e-5`` floor).

      ``com.microsoft.MoE``'s own routing (see
      ``generate_moe_function_templates.py``'s routing-section comment,
      and :func:`_moe_ffn`'s own docstring) is instead: softmax over ALL
      experts, top-k selection of the resulting probabilities, then --
      only when ``normalize_routing_weights=1`` -- renormalize the
      selected probabilities by their own sum. This is a DIFFERENT
      computation on paper, but is mathematically IDENTICAL to gpt-oss's
      real gating, re-derived from scratch here (not just trusted from an
      earlier, unverified pass): softmax is a strictly monotonic
      (order-preserving) function of its input, so the set of top-k
      experts selected by raw logit is exactly the set selected by
      softmax-over-all-experts probability -- the same experts either way.
      For a selected expert ``i`` among the top-k set ``S``, softmax-over-
      all gives probability ``p_i = exp(logit_i) / Z`` where
      ``Z = sum over ALL experts``. Renormalizing over ``S`` gives
      ``p_i / sum_{j in S} p_j = [exp(logit_i)/Z] / [sum_{j in S}
      exp(logit_j)/Z] = exp(logit_i) / sum_{j in S} exp(logit_j)`` -- the
      ``Z`` cancels completely -- which is EXACTLY
      ``softmax(logit_S)_i``, i.e. gpt-oss's own "softmax after top-k,
      over only the selected logits" weight. So gpt-oss's gating needs NO
      special wiring beyond what :func:`_moe_ffn` already does for
      Mixtral: raw router logits straight into ``router_probs``,
      ``normalize_routing_weights=1``. (This is the one place this
      docstring confirms, rather than merely repeats, an assumption
      supplied up front -- the derivation above was redone independently
      against the real ``build_moe_ffn`` source read above, not copied
      from an unverified earlier claim.)

    Not wired into :func:`_reconstruct_llama_family` or
    ``_SUPPORTED_ARCHITECTURES`` -- this is a standalone building block
    only, for a caller to integrate once gpt-oss's attention block (sliding
    -window pattern, attention sinks, RoPE scaling) also has a home there.
    """
    router_w = declare(f"{p}.ffn_gate_inp.weight", [n_expert, n_embd])
    router_b = declare_optional(f"{p}.ffn_gate_inp.bias", [n_expert])
    logits = _linear(b, h, router_w, router_b, f"{p}.moe_router")
    router_probs = b.op(
        "Reshape",
        [logits, b.shape_const([-1, n_expert])],
        f"{p}.moe_router_flat",
    )

    gate_w = declare(f"{p}.ffn_gate_exps.weight", [n_expert, n_ff, n_embd])
    up_w = declare(f"{p}.ffn_up_exps.weight", [n_expert, n_ff, n_embd])
    down_w = declare(f"{p}.ffn_down_exps.weight", [n_expert, n_embd, n_ff])
    gate_b = declare_optional(f"{p}.ffn_gate_exps.bias", [n_expert, n_ff])
    up_b = declare_optional(f"{p}.ffn_up_exps.bias", [n_expert, n_ff])
    down_b = declare_optional(f"{p}.ffn_down_exps.bias", [n_expert, n_embd])

    if (gate_b is None) != (up_b is None):
        raise UnsupportedArchitectureError(
            f"checkpoint has exactly one of '{p}.ffn_gate_exps.bias'/"
            f"'{p}.ffn_up_exps.bias' -- both or neither are required to "
            "build the fused fc1 bias swiglu_fusion=1 needs"
        )

    fc1_w = _interleave_gate_up(
        b, gate_w, up_w, n_expert, n_ff, [n_embd], f"{p}.moe_fc1_w"
    )
    fc1_b = (
        _interleave_gate_up(b, gate_b, up_b, n_expert, n_ff, [], f"{p}.moe_fc1_b")
        if gate_b is not None
        else ""
    )

    return b.op(
        "MoE",
        [h, router_probs, fc1_w, fc1_b, down_w, down_b or ""],
        f"{p}.moe",
        domain="com.microsoft",
        k=n_expert_used,
        activation_type="swiglu",
        swiglu_fusion=1,
        activation_alpha=1.702,
        activation_beta=1.0,
        swiglu_limit=7.0,
        normalize_routing_weights=1,
    )


def _reconstruct_llama_family(
    meta: dict, batch_size: int, seq_len: int
) -> onnx.GraphProto:
    kv = meta["kv"]
    tensors = {t["name"]: t for t in meta["tensors"]}
    arch = kv["general.architecture"]

    def key(suffix: str):
        return f"{arch}.{suffix}"

    n_embd = int(kv[key("embedding_length")])
    n_layer = int(kv[key("block_count")])
    n_ff = int(kv[key("feed_forward_length")])
    n_head = int(kv[key("attention.head_count")])
    n_head_kv = int(kv.get(key("attention.head_count_kv"), n_head))
    eps = float(
        kv.get(
            key("attention.layer_norm_rms_epsilon"),
            kv.get(key("attention.layer_norm_epsilon"), 1e-5),
        )
    )
    freq_base = float(kv.get(key("rope.freq_base"), 10000.0))

    # Mixture-of-experts: llama.cpp doesn't give Mixtral-style checkpoints a
    # distinct general.architecture value -- a Mixtral GGUF reports
    # architecture "llama" like any dense Llama checkpoint, and is
    # distinguished purely by expert_count > 0 (see llama.cpp's
    # src/models/llama.cpp: `if (model.layers[il].ffn_gate_inp == nullptr)`
    # selects the plain FFN branch, else the MoE one, both under the same
    # architecture). expert_weights_scale/expert_used_count are llama.cpp's
    # own hparim names for k and an optional post-softmax weight multiplier;
    # only the case matching com.microsoft.MoE's own reference decomposition
    # (plain softmax-over-all-experts routing, top-k, renormalized, no extra
    # scale) is implemented -- see _moe_ffn's own comment for exactly how
    # this was cross-checked against llama.cpp's build_moe_ffn source.
    n_expert = int(kv.get(key("expert_count"), 0))
    n_expert_used = int(kv.get(key("expert_used_count"), 0))
    expert_weights_scale = float(kv.get(key("expert_weights_scale"), 0.0))
    if n_expert > 0:
        if n_expert_used <= 0 or n_expert_used > n_expert:
            raise UnsupportedArchitectureError(
                f"{key('expert_count')}={n_expert} but "
                f"{key('expert_used_count')}={n_expert_used} is not a "
                "positive number <= expert_count"
            )
        if expert_weights_scale not in (0.0, 1.0):
            raise UnsupportedArchitectureError(
                f"{key('expert_weights_scale')}={expert_weights_scale} is "
                "not implemented (only the default of no extra scaling is)"
            )

    if n_embd % n_head != 0:
        raise UnsupportedArchitectureError(
            f"{key('embedding_length')}={n_embd} is not divisible by "
            f"{key('attention.head_count')}={n_head}"
        )
    head_dim = n_embd // n_head
    if n_head % n_head_kv != 0:
        raise UnsupportedArchitectureError(
            f"{key('attention.head_count')}={n_head} is not a multiple of "
            f"{key('attention.head_count_kv')}={n_head_kv} (grouped-query "
            "attention requires an integer head-repeat factor)"
        )
    n_rep = n_head // n_head_kv

    rope_dims = kv.get(key("rope.dimension_count"))
    if rope_dims is not None and int(rope_dims) != head_dim:
        raise UnsupportedArchitectureError(
            f"{key('rope.dimension_count')}={rope_dims} != head_dim={head_dim} "
            "(partial rotary embeddings are not implemented)"
        )

    def require(name: str) -> dict:
        info = tensors.get(name)
        if info is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor '{name}' for "
                f"architecture '{arch}'"
            )
        return info

    def optional(name: str) -> Optional[dict]:
        return tensors.get(name)

    b = _Builder()

    def declare(name: str, expected_shape: List[int]) -> str:
        info = require(name)
        if info["shape"] != expected_shape:
            raise UnsupportedArchitectureError(
                f"tensor '{name}' has shape {info['shape']}, expected "
                f"{expected_shape} for architecture '{arch}' with "
                f"embedding_length={n_embd}, feed_forward_length={n_ff}, "
                f"head_count={n_head}, head_count_kv={n_head_kv}"
            )
        ggml_type = info["ggml_type"]
        if ggml_type in _GGML_KQUANT_TYPES:
            onnx_dtype = onnx.TensorProto.FLOAT
        elif ggml_type in _GGML_RAW_TO_ONNX:
            onnx_dtype = _GGML_RAW_TO_ONNX[ggml_type]
        else:
            raise UnsupportedArchitectureError(
                f"tensor '{name}' uses ggml_type {ggml_type}, which "
                "import_gguf_weights cannot decode (only F32/F16/BF16/F64/"
                "I8/I16/I32/I64, the Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 K-quant "
                "family, the Q4_0/Q4_1/Q5_0/Q5_1 legacy family, and MXFP4 are "
                "supported)"
            )
        b.placeholder_weight(name, expected_shape, onnx_dtype)
        return name

    def declare_optional(name: str, expected_shape: List[int]) -> Optional[str]:
        return declare(name, expected_shape) if optional(name) is not None else None

    token_embd_info = require("token_embd.weight")
    if len(token_embd_info["shape"]) != 2 or token_embd_info["shape"][1] != n_embd:
        raise UnsupportedArchitectureError(
            f"token_embd.weight shape {token_embd_info['shape']} does not "
            f"match {key('embedding_length')}={n_embd}"
        )
    vocab_size = token_embd_info["shape"][0]
    token_embd = declare("token_embd.weight", [vocab_size, n_embd])

    input_ids = "input_ids"
    position_ids = "position_ids"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            input_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
        onnx.helper.make_tensor_value_info(
            position_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
    ]

    x = b.op("Gather", [token_embd, input_ids], "embed", axis=0)

    # RoPE cos/sin: identical across every layer (same position_ids, same
    # freq_base/head_dim), so computed once here rather than per layer.
    inv_freq = 1.0 / (
        freq_base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    inv_freq_c = b.const(
        inv_freq.reshape(1, 1, -1).astype(np.float32), prefix="inv_freq"
    )
    pos_f = b.op("Cast", [position_ids], "pos_f", to=onnx.TensorProto.FLOAT)
    pos_unsq = _unsqueeze(b, pos_f, [-1], "pos_unsq")
    freqs = b.op("Mul", [pos_unsq, inv_freq_c], "freqs")
    emb = b.op("Concat", [freqs, freqs], "rope_emb", axis=-1)
    cos = b.op("Cos", [emb], "rope_cos")
    sin = b.op("Sin", [emb], "rope_sin")
    # [B, S, D] -> [B, 1, S, D], broadcasting over both the H-head and the
    # HKV-head cases uniformly (dim 1 has size 1 either way).
    cos_b = _unsqueeze(b, cos, [1], "rope_cos_b")
    sin_b = _unsqueeze(b, sin, [1], "rope_sin_b")

    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    mask_c = b.const(causal_mask, prefix="causal_mask")
    inv_sqrt_d = b.const(
        np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), prefix="inv_sqrt_d"
    )

    def reshape(t: str, dims: List[int], prefix: str) -> str:
        return b.op("Reshape", [t, b.shape_const(dims)], prefix)

    n_embd_gqa = n_head_kv * head_dim
    for i in range(n_layer):
        p = f"blk.{i}"
        resid = x
        h = _rmsnorm(
            b, x, declare(f"{p}.attn_norm.weight", [n_embd]), eps, f"{p}.attn_norm"
        )

        q = _linear(
            b,
            h,
            declare(f"{p}.attn_q.weight", [n_embd, n_embd]),
            declare_optional(f"{p}.attn_q.bias", [n_embd]),
            f"{p}.q_proj",
        )
        k = _linear(
            b,
            h,
            declare(f"{p}.attn_k.weight", [n_embd_gqa, n_embd]),
            declare_optional(f"{p}.attn_k.bias", [n_embd_gqa]),
            f"{p}.k_proj",
        )
        v = _linear(
            b,
            h,
            declare(f"{p}.attn_v.weight", [n_embd_gqa, n_embd]),
            declare_optional(f"{p}.attn_v.bias", [n_embd_gqa]),
            f"{p}.v_proj",
        )

        q = reshape(q, [batch_size, seq_len, n_head, head_dim], f"{p}.q_r")
        q = b.op("Transpose", [q], f"{p}.q_t", perm=[0, 2, 1, 3])
        k = reshape(k, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.k_r")
        k = b.op("Transpose", [k], f"{p}.k_t", perm=[0, 2, 1, 3])
        v = reshape(v, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.v_r")
        v = b.op("Transpose", [v], f"{p}.v_t", perm=[0, 2, 1, 3])

        q = _apply_rope(b, q, cos_b, sin_b, head_dim, f"{p}.q_rope")
        k = _apply_rope(b, k, cos_b, sin_b, head_dim, f"{p}.k_rope")

        # Grouped-query attention via broadcasting, not an explicit
        # repeat/tile of k/v: split q's H heads into [HKV, REP] and give k/v
        # a size-1 REP axis, so MatMul's own batch-dimension broadcasting
        # does the repeat implicitly. See the module docstring's design note
        # and this function's own comment on _linear for the same
        # "let a later onnxsim.simplify() fold what it can" philosophy.
        q5 = reshape(q, [batch_size, n_head_kv, n_rep, seq_len, head_dim], f"{p}.q5")
        k5 = _unsqueeze(b, k, [2], f"{p}.k5")
        v5 = _unsqueeze(b, v, [2], f"{p}.v5")

        k5t = b.op("Transpose", [k5], f"{p}.k5t", perm=[0, 1, 2, 4, 3])
        scores = b.op("MatMul", [q5, k5t], f"{p}.scores")
        scores = b.op("Mul", [scores, inv_sqrt_d], f"{p}.scores_scaled")
        scores = b.op("Add", [scores, mask_c], f"{p}.scores_masked")
        attn = b.op("Softmax", [scores], f"{p}.softmax", axis=-1)
        out5 = b.op("MatMul", [attn, v5], f"{p}.attn_out5")

        out = reshape(out5, [batch_size, n_head, seq_len, head_dim], f"{p}.out_r")
        out = b.op("Transpose", [out], f"{p}.out_t", perm=[0, 2, 1, 3])
        out = reshape(out, [batch_size, seq_len, n_embd], f"{p}.out_flat")
        out = _linear(
            b,
            out,
            declare(f"{p}.attn_output.weight", [n_embd, n_embd]),
            declare_optional(f"{p}.attn_output.bias", [n_embd]),
            f"{p}.o_proj",
        )
        x = b.op("Add", [resid, out], f"{p}.attn_resid")

        resid = x
        h = _rmsnorm(
            b, x, declare(f"{p}.ffn_norm.weight", [n_embd]), eps, f"{p}.ffn_norm"
        )
        if n_expert > 0:
            ffn_out = _moe_ffn(b, h, p, n_embd, n_ff, n_expert, n_expert_used, declare)
        else:
            gate = _linear(
                b,
                h,
                declare(f"{p}.ffn_gate.weight", [n_ff, n_embd]),
                declare_optional(f"{p}.ffn_gate.bias", [n_ff]),
                f"{p}.gate_proj",
            )
            up = _linear(
                b,
                h,
                declare(f"{p}.ffn_up.weight", [n_ff, n_embd]),
                declare_optional(f"{p}.ffn_up.bias", [n_ff]),
                f"{p}.up_proj",
            )
            silu = b.op("Sigmoid", [gate], f"{p}.silu_sig")
            silu = b.op("Mul", [gate, silu], f"{p}.silu")
            act = b.op("Mul", [silu, up], f"{p}.act")
            ffn_out = _linear(
                b,
                act,
                declare(f"{p}.ffn_down.weight", [n_embd, n_ff]),
                declare_optional(f"{p}.ffn_down.bias", [n_embd]),
                f"{p}.down_proj",
            )
        x = b.op("Add", [resid, ffn_out], f"{p}.ffn_resid")

    x = _rmsnorm(b, x, declare("output_norm.weight", [n_embd]), eps, "output_norm")

    if optional("output.weight") is not None:
        lm_head = declare("output.weight", [vocab_size, n_embd])
    else:
        # Tied embeddings: some Llama-family checkpoints (small models
        # especially) have no separate LM head tensor at all and reuse
        # token_embd.weight for both -- token_embd was already declared
        # above, so there is nothing new to hydrate here.
        lm_head = token_embd
    logits = _linear(b, x, lm_head, None, "lm_head")

    graph = onnx.helper.make_graph(
        b.nodes,
        f"gguf_{arch}",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                logits, onnx.TensorProto.FLOAT, [batch_size, seq_len, vocab_size]
            )
        ],
        initializer=b.initializers,
    )
    return graph


def _reconstruct_gpt_oss(meta: dict, batch_size: int, seq_len: int) -> onnx.GraphProto:
    """Builds gpt-oss-20b's real per-layer structure directly from a GGUF
    checkpoint: RMSNorm -> ``_gpt_oss_attention_block`` (YaRN-scaled RoPE,
    alternating sliding-window/full masking, attention sinks) -> RMSNorm ->
    ``_gpt_oss_moe_ffn`` (a ``swiglu_fusion=1`` ``com.microsoft.MoE`` node
    with fc1 fused at graph-build time from llama.cpp's own separate
    ``ffn_gate_exps``/``ffn_up_exps`` tensors). A distinct architecture
    family from :func:`_reconstruct_llama_family`'s (they share no block
    shape beyond "decoder-only transformer"), verified directly against
    ``llama.cpp/src/models/openai-moe.cpp``'s
    ``llama_model_openai_moe::graph`` constructor -- see that file's
    per-layer loop for the exact op sequence this reproduces.

    One naming detail this integration step is specifically responsible
    for getting right (neither ``_gpt_oss_attention_block`` nor
    ``_gpt_oss_moe_ffn`` reads a norm tensor themselves): the tensor
    normalizing FFN's input is GGUF-named ``blk.N.post_attention_norm.weight``
    for this architecture (``LLM_TENSOR_ATTN_POST_NORM`` in
    ``llama-arch.cpp``) -- NOT ``blk.N.ffn_norm.weight``, the name
    ``_reconstruct_llama_family``'s architectures use for the
    functionally-identical "norm right before the FFN" role. llama.cpp
    itself has two different GGUF tensor names for that same role, and
    gpt-oss uses the other one (openai-moe.cpp's graph ctor: the residual
    add after attention happens BEFORE this norm, and the FFN's own
    residual add happens after it -- i.e. this is a pre-FFN norm exactly
    like ``ffn_norm`` elsewhere, just named differently in this file).

    gpt-oss's expert-router details differ from
    ``_reconstruct_llama_family``'s validated Mixtral case in ways specific
    to this function: a required ``ffn_gate_inp.bias`` (Mixtral's router has
    none), and expert intermediate size reported under a different GGUF key
    (``expert_feed_forward_length``, not the dense ``feed_forward_length``
    -- see ``LLM_KV_EXPERT_FEED_FORWARD_LENGTH`` in ``llama-arch.cpp``).
    ``head_dim`` is likewise independent GGUF metadata
    (``attention.key_length``/``.value_length``), not
    ``embedding_length // head_count`` -- see
    ``_gpt_oss_attention_block``'s own docstring, point 1.
    """
    kv = meta["kv"]
    tensors = {t["name"]: t for t in meta["tensors"]}
    arch = kv["general.architecture"]

    def key(suffix: str):
        return f"{arch}.{suffix}"

    n_embd = int(kv[key("embedding_length")])
    n_layer = int(kv[key("block_count")])
    n_ff = int(kv[key("expert_feed_forward_length")])
    n_head = int(kv[key("attention.head_count")])
    n_head_kv = int(kv.get(key("attention.head_count_kv"), n_head))
    eps = float(
        kv.get(
            key("attention.layer_norm_rms_epsilon"),
            kv.get(key("attention.layer_norm_epsilon"), 1e-5),
        )
    )
    freq_base = float(kv.get(key("rope.freq_base"), 10000.0))

    if n_head % n_head_kv != 0:
        raise UnsupportedArchitectureError(
            f"{key('attention.head_count')}={n_head} is not a multiple of "
            f"{key('attention.head_count_kv')}={n_head_kv} (grouped-query "
            "attention requires an integer head-repeat factor)"
        )

    # gpt-oss's head_dim is independent GGUF metadata, NOT
    # embedding_length // head_count (see _gpt_oss_attention_block's own
    # docstring, point 1, for gpt-oss-20b's real numbers: head_dim=64 vs
    # embedding_length/head_count=2880/64=45).
    head_dim_kv = kv.get(key("attention.key_length"))
    head_dim_v = kv.get(key("attention.value_length"))
    if head_dim_kv is None and head_dim_v is None:
        raise UnsupportedArchitectureError(
            f"gpt-oss requires {key('attention.key_length')} or "
            f"{key('attention.value_length')} in GGUF metadata"
        )
    if head_dim_kv is not None and head_dim_v is not None:
        if int(head_dim_kv) != int(head_dim_v):
            raise UnsupportedArchitectureError(
                f"{key('attention.key_length')}={head_dim_kv} != "
                f"{key('attention.value_length')}={head_dim_v} (distinct "
                "key/value head dims are not implemented)"
            )
    head_dim = int(head_dim_kv if head_dim_kv is not None else head_dim_v)

    n_expert = int(kv.get(key("expert_count"), 0))
    n_expert_used = int(kv.get(key("expert_used_count"), 0))
    if n_expert <= 0 or n_expert_used <= 0 or n_expert_used > n_expert:
        raise UnsupportedArchitectureError(
            f"gpt-oss requires {key('expert_count')} > 0 and a positive "
            f"{key('expert_used_count')} <= expert_count (got "
            f"expert_count={n_expert}, expert_used_count={n_expert_used})"
        )
    # _gpt_oss_moe_ffn has no expert_weights_scale parameter at all (it
    # assumes the default of no extra scaling), so the same guard
    # _reconstruct_llama_family applies for Mixtral applies here too --
    # see that function's identical check for why.
    expert_weights_scale = float(kv.get(key("expert_weights_scale"), 0.0))
    if expert_weights_scale not in (0.0, 1.0):
        raise UnsupportedArchitectureError(
            f"{key('expert_weights_scale')}={expert_weights_scale} is not "
            "implemented (only the default of no extra scaling is)"
        )

    # llama.cpp's own load_arch_hparams reads this key as REQUIRED (a plain
    # 2-arg ml.get_key call, which throws if absent) -- mirrored here rather
    # than defaulted.
    if key("attention.sliding_window") not in kv:
        raise UnsupportedArchitectureError(
            f"gpt-oss requires {key('attention.sliding_window')} in GGUF metadata"
        )
    sliding_window = int(kv[key("attention.sliding_window")])
    # Unlike sliding_window above, llama.cpp reads this one as optional
    # with a hardcoded default of 2 (load_arch_hparams's local
    # `uint32_t swa_period = 2`) -- no real gpt-oss GGUF sets this key at
    # all (see _gpt_oss_is_sliding_layer's own docstring).
    swa_period = int(kv.get(key("attention.sliding_window_pattern"), 2))

    rope_scaling_type = kv.get(key("rope.scaling.type"))
    if rope_scaling_type is not None and rope_scaling_type not in ("yarn", "none"):
        raise UnsupportedArchitectureError(
            f"{key('rope.scaling.type')}={rope_scaling_type!r} is not "
            "implemented (only 'yarn' and 'none' are)"
        )
    # _gpt_oss_yarn_cos_sin's own docstring: this degenerates correctly to
    # plain RoPE when yarn_factor<=1.0 (the "none"/absent-key case), so no
    # separate code path is needed for a non-YaRN checkpoint.
    yarn_factor = float(kv.get(key("rope.scaling.factor"), 1.0))
    yarn_orig_ctx = float(kv.get(key("rope.scaling.original_context_length"), 4096.0))
    yarn_beta_fast = float(kv.get(key("rope.scaling.yarn_beta_fast"), 32.0))
    yarn_beta_slow = float(kv.get(key("rope.scaling.yarn_beta_slow"), 1.0))

    def require(name: str) -> dict:
        info = tensors.get(name)
        if info is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor '{name}' for "
                f"architecture '{arch}'"
            )
        return info

    def optional(name: str) -> Optional[dict]:
        return tensors.get(name)

    b = _Builder()

    def declare(name: str, expected_shape: List[int]) -> str:
        info = require(name)
        if info["shape"] != expected_shape:
            raise UnsupportedArchitectureError(
                f"tensor '{name}' has shape {info['shape']}, expected "
                f"{expected_shape} for architecture '{arch}' with "
                f"embedding_length={n_embd}, "
                f"expert_feed_forward_length={n_ff}, head_count={n_head}, "
                f"head_count_kv={n_head_kv}"
            )
        ggml_type = info["ggml_type"]
        if ggml_type in _GGML_KQUANT_TYPES:
            onnx_dtype = onnx.TensorProto.FLOAT
        elif ggml_type in _GGML_RAW_TO_ONNX:
            onnx_dtype = _GGML_RAW_TO_ONNX[ggml_type]
        else:
            raise UnsupportedArchitectureError(
                f"tensor '{name}' uses ggml_type {ggml_type}, which "
                "import_gguf_weights cannot decode (only F32/F16/BF16/F64/"
                "I8/I16/I32/I64, the Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 K-quant "
                "family, the Q4_0/Q4_1/Q5_0/Q5_1 legacy family, and MXFP4 are "
                "supported)"
            )
        b.placeholder_weight(name, expected_shape, onnx_dtype)
        return name

    def declare_optional(name: str, expected_shape: List[int]) -> Optional[str]:
        return declare(name, expected_shape) if optional(name) is not None else None

    token_embd_info = require("token_embd.weight")
    if len(token_embd_info["shape"]) != 2 or token_embd_info["shape"][1] != n_embd:
        raise UnsupportedArchitectureError(
            f"token_embd.weight shape {token_embd_info['shape']} does not "
            f"match {key('embedding_length')}={n_embd}"
        )
    vocab_size = token_embd_info["shape"][0]
    token_embd = declare("token_embd.weight", [vocab_size, n_embd])

    input_ids = "input_ids"
    position_ids = "position_ids"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            input_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
        onnx.helper.make_tensor_value_info(
            position_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
    ]

    x = b.op("Gather", [token_embd, input_ids], "embed", axis=0)

    # YaRN cos/sin: identical across every layer (same position_ids, same
    # freq_base/head_dim/YaRN params; the sliding-vs-full distinction is
    # purely a masking difference -- see _gpt_oss_attention_block's own
    # docstring, point 2, for why), so computed once here.
    cos_b, sin_b = _gpt_oss_yarn_cos_sin(
        b,
        position_ids,
        head_dim,
        freq_base,
        yarn_factor,
        yarn_orig_ctx,
        yarn_beta_fast,
        yarn_beta_slow,
        "rope",
    )

    for i in range(n_layer):
        p = f"blk.{i}"
        x = _gpt_oss_attention_block(
            b,
            x,
            p,
            i,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            cos_b,
            sin_b,
            sliding_window,
            swa_period,
            batch_size,
            seq_len,
            eps,
            declare,
            declare_optional,
        )

        resid = x
        # Named "post_attention_norm" for this architecture, not "ffn_norm"
        # -- see this function's own docstring.
        h = _rmsnorm(
            b,
            x,
            declare(f"{p}.post_attention_norm.weight", [n_embd]),
            eps,
            f"{p}.ffn_norm",
        )
        ffn_out = _gpt_oss_moe_ffn(
            b, h, p, n_embd, n_ff, n_expert, n_expert_used, declare, declare_optional
        )
        x = b.op("Add", [resid, ffn_out], f"{p}.ffn_resid")

    x = _rmsnorm(b, x, declare("output_norm.weight", [n_embd]), eps, "output_norm")

    # Unlike _reconstruct_llama_family's tied-embeddings fallback,
    # load_arch_tensors always creates a separate "output.weight" tensor
    # for gpt-oss unconditionally -- no tied-embeddings case to handle.
    lm_head = declare("output.weight", [vocab_size, n_embd])
    logits = _linear(b, x, lm_head, None, "lm_head")

    graph = onnx.helper.make_graph(
        b.nodes,
        f"gguf_{arch}",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                logits, onnx.TensorProto.FLOAT, [batch_size, seq_len, vocab_size]
            )
        ],
        initializer=b.initializers,
    )
    return graph


def reconstruct_gguf_graph(
    gguf_path: str, batch_size: int = 1, seq_len: int = 8
) -> Tuple[onnx.ModelProto, List[str]]:
    """
    Build a runnable ONNX graph -- structure *and* weights -- directly from
    a GGUF checkpoint, for a recognized architecture (currently the Llama
    family: ``llama``, ``qwen2``, ``mistral``; and ``gpt-oss`` -- see the
    module docstring).

    Unlike :func:`import_gguf_weights`, which only ever fills in an
    existing graph's initializer *values*, this constructs the graph
    itself from the checkpoint's own declared hyperparameters
    (:func:`read_gguf_metadata`), then calls ``import_gguf_weights``
    internally to hydrate it -- so it reuses that function's existing
    K-quant (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0), legacy (Q4_0/Q4_1/Q5_0/Q5_1), and
    MXFP4 decode unchanged.

    :param gguf_path: path to the GGUF checkpoint
    :param batch_size: static batch dimension baked into the returned
            graph's input/output shapes (see the module docstring's scope
            note: this is not a dynamic axis)
    :param seq_len: static sequence-length dimension, likewise baked in
    :returns: ``(model, skipped)`` -- the constructed, hydrated model
            (inputs ``input_ids``/``position_ids``, both
            ``int64[batch_size, seq_len]``; output ``logits``,
            ``float32[batch_size, seq_len, vocab_size]``), and the names of
            any GGUF tensors present in the file but left un-hydrated
            (always empty in practice: every tensor this graph references
            is validated against the supported dtype set before the graph
            is even built, and ``import_gguf_weights`` never touches a
            tensor that is not present in ``model``'s initializers).
    :raises UnsupportedArchitectureError: if ``general.architecture`` is not
            one this builder has a template for, a required tensor is
            missing, or a required tensor's quantization format has no
            decoder (see :func:`import_gguf_weights`'s own scope note).
    """
    meta = read_gguf_metadata(gguf_path)
    arch = meta["kv"].get("general.architecture")
    if arch not in _SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitectureError(
            f"general.architecture={arch!r} has no graph template here -- "
            f"supported: {', '.join(_SUPPORTED_ARCHITECTURES)}"
        )

    if arch == "gpt-oss":
        graph = _reconstruct_gpt_oss(meta, batch_size, seq_len)
    else:
        graph = _reconstruct_llama_family(meta, batch_size, seq_len)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", _OPSET),
            # Only ever used when the checkpoint is a MoE architecture (see
            # _moe_ffn) -- harmless to declare unconditionally otherwise, an
            # unused opset_import is valid ONNX (checked directly).
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = _IR_VERSION

    model, skipped = import_gguf_weights(model, gguf_path)
    return model, skipped
