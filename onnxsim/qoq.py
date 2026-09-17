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

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_qoq_cpp


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

    Delegates to the verified C++ port (:func:`onnxsim.apply_qoq_cpp`),
    which hardcodes this function's own defaults (``block_size=32``,
    ``int8_clip_max=119``) and builds the exact same
    ``DequantizeLinear(Wq, Ws, ...)`` graph rewrite this function's own
    former implementation did (INT4 codes plus a combined per-group
    scale) -- no storage-format change here, unlike several of this
    module's own siblings.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) quantization
            group along the reduction dimension, for the second (INT4)
            stage. **The C++ port's own only supported value is 32** -- a
            non-default value raises ``ValueError``.
    :param int8_clip_max: the first stage's protective INT8 clipping range
            (see this module's own docstring); must be in ``(0, 127]``.
            **The C++ port's own only supported value is 119** -- a
            non-default value raises ``ValueError``.
    :returns: ``model`` with every matched layer's weight replaced by
            ``DequantizeLinear(Wq, Ws, axis=<reduction axis>,
            block_size=32)`` feeding the original MatMul/Gemm node; layers
            with a non-constant, non-2-D, or non-block-divisible weight
            are left untouched, as is the whole model if its opset is
            below 21.
    """
    if not 0 < int8_clip_max <= 127:
        raise ValueError("int8_clip_max must be in (0, 127]")
    if block_size != 32 or int8_clip_max != 119:
        raise ValueError(
            "quantize_weight_only_qoq now delegates to apply_qoq_cpp, "
            "which hardcodes block_size=32, int8_clip_max=119 and cannot "
            "honor other values"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_qoq_cpp(model)


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

    Delegates to the verified C++ port
    (:func:`onnxsim.apply_smooth_attention_cpp`), which reimplements this
    function's own attention-subgraph matching (transcribed from
    :func:`onnxsim.attention_quantization._find_attention_candidates` at
    the protobuf level -- see ``smooth_attention_entry.h`` for why) and
    per-head-dim-channel absmax/scale computation exactly -- a closed-form
    diagonal rescaling with no RNG. This function's own former
    pure-Python implementation is preserved as-is in this module's own
    git history.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    from onnxsim.onnx_simplifier import apply_smooth_attention_cpp

    return apply_smooth_attention_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        epsilon=epsilon,
        providers=providers,
    )
