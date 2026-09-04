"""QServe's QoQ quantization (Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang,
Gan, Han, MLSys 2025, "QServe: W4A8KV4 Quantization and System Co-design for
Efficient LLM Serving", https://arxiv.org/abs/2405.04532). onnxsim ports the
*algorithm*, not QServe's own CUDA kernels, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.smoothquant` (QServe has
no ONNX export path).

QoQ ("quattuor-octo-quattuor", 4-8-4) genuinely combines two independent
contributions; this module implements one directly and documents the other's
scope.

**1. Progressive (two-stage) weight quantization -- this module's primary
contribution (:func:`quantize_weight_only_qoq`).** Every other weight-only
INT4 quantizer in onnxsim (``quantize_weight_only_int4`` and everything
built on it -- :mod:`onnxsim.awq`, :mod:`onnxsim.gptq`, :mod:`onnxsim.hqq`,
...) rounds the original float weight directly to a single INT4 grid, one
scale per block. QServe's own motivation for *not* doing that is a hardware
one: dequantizing INT4 straight to FP16 needs an expensive, irregular
per-element conversion path, whereas INT8-to-FP16 dequantization can stay on
a GPU's INT8 tensor cores. QServe therefore quantizes in two stages instead
of one -- first the whole float weight, per output channel, to INT8, using a
**protective clipping range** (``int8_clip_max``, the paper's own headroom
below the full ``[-127, 127]`` INT8 range) that leaves room for the second
stage's own rounding error; then, within that already-INT8-quantized
tensor, each ``block_size``-element group of the reduction dimension is
quantized again, down to INT4. This module reproduces that numerically: the
key difference from ``quantize_weight_only_int4``'s single-stage rounding is
that the INT4 code here is derived by rounding an *already-INT8-quantized*
value, not the original float weight, and every reconstructed value passes
through the INT8 grid on the way back to float (``code4 -> INT8 grid value
-> float``), even though the two per-stage scales are folded into one
combined per-(channel, group) scale so the graph itself only ever needs to
emit a single ``DequantizeLinear`` -- exactly
``quantize_weight_only_int4``'s own graph shape (INT4 codes plus a
block-wise scale), differing only in how those codes and that scale were
computed.

    Stage 1 (per output channel, protective INT8):
        s1 = max(|W_row|) / int8_clip_max
        code8 = clip(round(W_row / s1), -int8_clip_max, int8_clip_max)

    Stage 2 (per (channel, block-of-K) group, INT8 grid -> INT4):
        s2 = max(|code8_group|) / 7
        code4 = clip(round(code8_group / s2), -7, 7)

    Reconstruction (two-stage, folded into one scale for the graph):
        W_hat = code4 * s2 * s1
              = (code4 -> code8 grid value via s2) -> float via s1

**2. SmoothAttention -- QoQ's KV-cache-side contribution
(:func:`apply_smooth_attention`).** :mod:`onnxsim.kv_cache_quantization`
already implements KIVI/KVQuant-style per-channel Key quantization,
QServe's own aggressive 4-bit Key cache needs a smoothing step *before*
that: the same outlier channels that make per-channel Key quantization
worthwhile in the first place still limit how low it can go on their own.
SmoothAttention migrates that difficulty out of Key (which gets quantized)
and into Query (which never does -- attention math always keeps Q in
float): the exact diagonal-rescaling identity
:mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression` already use
for MatMul/Gemm (``(X / s) @ (W * s) == X @ W``), applied instead to the
``QK^T`` dot product inside attention -- for channel ``j`` of the shared
head-dim axis, ``(K_j / s_j) . (Q_j * s_j) == K_j . Q_j``, so dividing Key's
channel ``j`` by ``s_j`` and multiplying Query's matching channel by the
same ``s_j`` leaves the attention scores exactly (up to floating-point
rounding) unchanged while flattening Key's own per-channel range -- exactly
what a *following* call to :func:`onnxsim.quantize_kv_cache` (Key-style)
needs to quantize it well. Like :mod:`onnxsim.smoothquant`,
:func:`apply_smooth_attention` only performs the *migration* -- it returns a
float-equivalent model, no quantization happens here at all -- meant to run
immediately before :func:`onnxsim.quantize_kv_cache` in a pipeline, the same
way :mod:`onnxsim.smoothquant` is meant to run before
:func:`onnxsim.quantize_static`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _pack_int4
from onnxsim.attention_quantization import _find_attention_candidates
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data

_EPS = 1e-12


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(x_name, w_name, weight_transposed)`` or
    ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[0], node.input[1], False
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
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[0], node.input[1], weight_transposed
    return None


def quantize_weight_only_qoq(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    int8_clip_max: int = 119,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into QoQ-style progressive (INT8-then-INT4) block-wise
    INT4 -- see this module's own docstring for the two-stage technique and
    how it differs from ``quantize_weight_only_int4``'s single-stage
    rounding. Needs no calibration data: like ``quantize_weight_only_int4``,
    every quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) quantization
            group along the reduction dimension, for the second (INT4)
            stage
    :param int8_clip_max: the first stage's protective INT8 clipping range
            (the paper's own headroom below the full 127-magnitude INT8
            range, leaving room for the second stage's own rounding error
            without overflowing); must be in ``(0, 127]``
    :returns: ``model`` with every matched layer's weight replaced by
            ``DequantizeLinear(Wq, Ws, axis=<reduction axis>,
            block_size=block_size)`` feeding the original MatMul/Gemm node
            (signed INT4 codes, symmetric -- no zero-point input, matching
            ``quantize_weight_only_int4``'s own convention), where ``Ws``
            is each ``(output channel, block)`` group's *combined*
            two-stage scale; layers with a non-constant, non-2-D, or
            non-block-divisible weight are left untouched, as is the whole
            model if its opset is below 21 (``DequantizeLinear``'s
            ``block_size`` attribute and the native INT4 tensor type both
            need opset 21+)
    """
    if not 0 < int8_clip_max <= 127:
        raise ValueError("int8_clip_max must be in (0, 127]")

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    opset_ge_21 = any(
        o.domain in ("", "ai.onnx") and o.version >= 21 for o in out.opset_import
    )
    if not opset_ge_21:
        return model

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape
        if k % block_size != 0:
            continue
        num_groups = k // block_size

        # Stage 1: FP16 -> INT8, per output channel, protective clip range.
        channel_absmax = np.maximum(np.abs(w_nk).max(axis=1), _EPS)  # [N]
        s1 = channel_absmax / int8_clip_max
        code8 = np.clip(
            np.round(w_nk / s1[:, np.newaxis]), -int8_clip_max, int8_clip_max
        )  # [N, K], already-quantized INT8 grid values

        # Stage 2: that INT8 grid -> INT4, per (channel, block-of-K) group --
        # rounds the already-quantized code8, not the original float weight.
        code8_blocks = code8.reshape(n, num_groups, block_size)
        group_absmax = np.maximum(np.abs(code8_blocks).max(axis=2), _EPS)
        s2 = group_absmax / 7.0  # [N, num_groups]
        code4 = np.clip(
            np.round(code8_blocks / s2[:, :, np.newaxis]), -7.0, 7.0
        ).reshape(n, k)

        # Final reconstruction folds both stages into one combined scale --
        # code4 * s2 * s1 -- so the graph only needs a single
        # DequantizeLinear, even though the codes were derived via the
        # two-stage round-trip through the INT8 grid described above.
        combined_scale = s1[:, np.newaxis] * s2  # [N, num_groups]

        codes_orig = code4 if weight_transposed else code4.T
        scale_orig = combined_scale if weight_transposed else combined_scale.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.TensorProto()
        wq.name = _unique_name(f"{w_name}_qoq_q", taken_names)
        wq.data_type = onnx.TensorProto.INT4
        wq.dims.extend(codes_orig.shape)
        wq.raw_data = _pack_int4(codes_orig)
        graph.initializer.append(wq)

        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_qoq_scale", taken_names),
        )
        graph.initializer.append(ws)

        reduction_axis = 1 if weight_transposed else 0
        dq_out = _unique_name(f"{w_name}_qoq_dq", taken_names)
        dq_node = onnx.helper.make_node(
            "DequantizeLinear",
            [wq.name, ws.name],
            [dq_out],
            name=_unique_name(f"{w_name}_qoq_dequant", taken_names),
            axis=reduction_axis,
            block_size=block_size,
        )
        graph.node.insert(
            next(i for i, n in enumerate(graph.node) if n is node), dq_node
        )
        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out


def apply_smooth_attention(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Migrates Key's per-channel quantization difficulty into Query (which
    stays float) for every decomposed attention subgraph
    (``MatMul(Q,Kt) -> [Mul/Div] -> [Add] -> Softmax -> MatMul(_,V)``, the
    same pattern :func:`onnxsim.apply_attention_quantization` matches) --
    see this module's own docstring for the technique. Returns a float
    model, provably equivalent to the input up to floating-point rounding --
    no quantization happens here at all: pass the result to
    :func:`onnxsim.quantize_kv_cache` (Key-style, the default) to actually
    quantize the now-flatter Key cache.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            Key head-dim channel's activation range on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            more representative migration than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param epsilon: floor applied to every per-channel Key max-abs value
            before dividing by it, avoiding a divide-by-zero on an
            all-zero channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched subgraph's ``Q`` operand
            multiplied by a new per-head-dim-channel scale ``s`` and its
            ``Kt`` (Key, transposed) operand divided by the same ``s``,
            via two new ``Mul``/``Div`` nodes inserted right before the
            ``QK^T`` MatMul; a subgraph whose Key tensor never appeared as
            a plain-enough (rank >= 2) probe, or a model with no matching
            subgraph at all, is left untouched for that subgraph (or
            returned unchanged, respectively)
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    taken_names = _all_names(graph)

    candidates = []
    seen_matmuls = set()
    for c in _find_attention_candidates(graph):
        if id(c.qk_matmul) in seen_matmuls:
            continue
        seen_matmuls.add(id(c.qk_matmul))
        candidates.append(c)
    if not candidates:
        return out

    probe_names = sorted({c.qk_matmul.input[1] for c in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    # Key's own per-head-dim-channel max-abs value, over the calibration
    # set -- Kt (Key, transposed) has shape [..., head_dim, seq_k], so the
    # channel axis is the second-to-last one.
    k_absmax: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            arr = np.asarray(result[name], dtype=np.float64)
            if arr.ndim < 2:
                continue
            flat = np.moveaxis(arr, -2, -1).reshape(-1, arr.shape[-2])
            m = np.abs(flat).max(axis=0)
            k_absmax[name] = (
                m if name not in k_absmax else np.maximum(k_absmax[name], m)
            )

    for c in candidates:
        kt_name = c.qk_matmul.input[1]
        absmax = k_absmax.get(kt_name)
        if absmax is None:
            continue  # never observed as a rank >= 2 tensor; skip

        s = np.maximum(absmax, epsilon).astype(np.float32)  # [head_dim]
        head_dim = s.shape[0]
        q_name = c.qk_matmul.input[0]

        # Q *= s -- broadcasts over Q's own last axis (head_dim).
        s_row_name = _unique_name(f"{q_name}_smooth_attn_s", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(s, name=s_row_name))
        q_scaled_name = _unique_name(f"{q_name}_smooth_attn_q", taken_names)
        q_mul_node = onnx.helper.make_node(
            "Mul",
            [q_name, s_row_name],
            [q_scaled_name],
            name=_unique_name(f"{q_name}_smooth_attn_q_node", taken_names),
        )

        # Kt /= s -- broadcasts over Kt's second-to-last axis (head_dim),
        # via a [head_dim, 1]-shaped divisor.
        s_col_name = _unique_name(f"{kt_name}_smooth_attn_s_col", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(s.reshape(head_dim, 1), name=s_col_name)
        )
        kt_scaled_name = _unique_name(f"{kt_name}_smooth_attn_k", taken_names)
        k_div_node = onnx.helper.make_node(
            "Div",
            [kt_name, s_col_name],
            [kt_scaled_name],
            name=_unique_name(f"{kt_name}_smooth_attn_k_node", taken_names),
        )

        node_idx = next(i for i, n in enumerate(graph.node) if n is c.qk_matmul)
        graph.node.insert(node_idx, q_mul_node)
        graph.node.insert(node_idx + 1, k_div_node)
        c.qk_matmul.input[0] = q_scaled_name
        c.qk_matmul.input[1] = kt_scaled_name

    return out
