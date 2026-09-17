"""DAQ (Yu, Tang, Yu, Xie, Liu, Zhu, Li, 2026, "DAQ: Delta-Aware
Quantization for Post-Training LLM Weight Compression",
https://arxiv.org/abs/2603.22324, Tencent Hunyuan/Yuanbao AI Infra Team) --
a **delta-aware** FP8 weight quantizer for a *fine-tuned* checkpoint.

Every weight-only PTQ pass already in onnxsim
(:func:`onnxsim.apply_deepseek_fp8`, :func:`onnxsim.apply_gptq`,
:mod:`onnxsim.awq`, ...) scores a candidate quantization of a layer's
weight ``W`` against ``W`` itself: pick whatever minimizes
``||W - Ŵ||`` (possibly weighted by activation statistics). DAQ's own
observation is that this objective is the wrong one for the very common
case of quantizing a model that is itself a *fine-tune* (SFT/RLHF) of some
earlier base checkpoint. What that fine-tune actually produced is not
``W_post`` but the **update** ``ΔW = W_post - W_base``, and ``ΔW`` is
typically orders of magnitude smaller than ``W_post`` itself. A quantizer
blind to that decomposition can wipe out a large fraction of ``ΔW`` --
the entire signal the fine-tune contributed -- while still reporting a
perfectly respectable absolute reconstruction error on ``W_post``,
because the error it does make is small *relative to* ``W_post``'s own
(much larger) magnitude. DAQ therefore chooses the layer's quantization
scale to preserve ``ΔW``, not to minimize raw reconstruction MSE.

Concretely, for each matched layer this module:

1. pairs the layer with its counterpart in ``base_model`` (same node
   output name -- the same two-model correspondence assumption
   :func:`onnxsim.apply_gptq`/:mod:`onnxsim.adaround` already make) and
   forms ``ΔW = W_post - W_base``;
2. grid-searches a **single scalar** FP8 E4M3 scale for the whole layer,
   scoring each candidate ``s`` by how well the *reconstructed delta*
   ``ΔŴ = (fp8_round_trip(W_post / s) * s) - W_base`` agrees with the
   real ``ΔW`` -- by one of the paper's own two named delta-fidelity
   metrics, **cosine similarity** (default) or **sign preservation
   rate** (the fraction of elements whose update kept its direction);
3. writes back ``fp8_round_trip(W_post / s_best) * s_best`` as a plain
   float32 initializer.

The search starts from the naive absmax scale
``max(|W_post|) / 448`` (``448`` is FLOAT8E4M3FN's own largest finite
magnitude -- the same constant, and the same round-trip helper,
:mod:`onnxsim.deepseek_fp8` uses) and is **coarse-to-fine**: 9 log-spaced
multipliers over ``[0.5, 2.0]``, then 9 linearly-spaced multipliers over
``[0.9 * m_best, 1.1 * m_best]``. Because FP8 is a *floating-point*
grid, a scale change is not a uniform rescale of the rounding error the
way it would be for INT8: it slides the format's binade boundaries
relative to the weight's own values (and moves the clipping threshold),
so different scales trade quantization error between the layer's large
and small entries very differently -- which is exactly the freedom DAQ
spends on ``ΔW`` instead of on ``W_post``.

**Honesty about scope**: this ports the paper's core delta-preservation
objective and both of its named delta-fidelity metrics. The specific
search *schedule* (9 log-spaced coarse candidates + 9 linear fine ones,
one scalar scale per layer) is this module's own simple, deterministic
choice -- it is not claimed to be a literal transcription of the paper's
own search hyper-parameters, the same qualification
:mod:`onnxsim.adaquant` makes about what it does and does not reproduce
from its own source. Granularity is deliberately coarser than
:mod:`onnxsim.deepseek_fp8`'s (one scale per 128x128 block) or
:mod:`onnxsim.llm_fp4`'s: DAQ's own scope is per-layer, and a per-layer
scalar is what makes the delta-fidelity objective a cheap 1-D search.

The pass is **data-free**: it never runs either model, needs no
calibration activations, and takes no ``providers`` argument -- every
decision comes from the two weight tensors alone.

**Scope**: only ``MatMul`` and "vanilla" ``Gemm`` (via
:func:`onnxsim.mx_quantization._match_matmul_like`) with a constant 2-D
float32 weight, whose ``base_model`` counterpart is another such node
with a same-shaped constant 2-D float32 weight, are touched. A layer
whose ``ΔW`` is exactly zero (an untouched-by-fine-tuning weight) is
left **completely unquantized** -- there is no delta signal for this
technique to preserve, so it declines to guess, the same
"decline rather than guess" convention every sibling module follows.
Since the scale is folded straight back into a float32 initializer, no
graph node, no ``Cast``, and no minimum opset is needed -- no FP8-typed
tensor ever enters the graph.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import onnx

from onnxsim.onnx_simplifier import apply_daq_cpp

#: The two delta-fidelity metrics the paper names, and the only values
#: :func:`apply_daq`'s ``metric`` accepts.
DAQ_METRICS: Tuple[str, ...] = ("cosine", "sign_preservation")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two arrays, compared as flat vectors. Returns
    ``-1.0`` (the worst possible similarity) when either side is the zero
    vector, so a degenerate candidate can never win a search.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(x)) * float(np.linalg.norm(y))
    if not np.isfinite(denom) or denom <= 0.0:
        return -1.0
    similarity = float(np.dot(x, y) / denom)
    return similarity if np.isfinite(similarity) else -1.0


def _sign_preservation_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of elements whose sign agrees between ``a`` and ``b`` (an
    element that is exactly zero in both counts as agreeing, since
    ``sign(0) == sign(0)``). In ``[0.0, 1.0]``, higher is better.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.sign(x) == np.sign(y)))


def apply_daq(
    base_model: Union[str, onnx.ModelProto],
    post_trained_model: Union[str, onnx.ModelProto],
    metric: str = "cosine",
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """FP8-quantizes every matched MatMul/"vanilla" Gemm weight of
    ``post_trained_model`` with a per-layer scalar scale chosen to
    preserve that layer's fine-tuning update ``ΔW = W_post - W_base``
    rather than to minimize its raw reconstruction error -- see this
    module's own docstring for the technique. Data-free: neither model is
    ever run, and no calibration data is needed.

    :param base_model: the pre-fine-tuning checkpoint (onnx ModelProto or
            file path), exported the same way as ``post_trained_model`` so
            corresponding nodes share their output tensor names -- the
            same correspondence assumption :func:`onnxsim.apply_gptq` makes
            about its own two model arguments
    :param post_trained_model: the fine-tuned checkpoint (onnx ModelProto
            or file path); this is the model that gets quantized and
            returned
    :param metric: which delta-fidelity metric drives the scale search --
            ``"cosine"`` (cosine similarity between ``ΔW`` and its
            reconstruction, the default) or ``"sign_preservation"`` (the
            fraction of elements whose update kept its sign). Any other
            value raises :class:`ValueError`.
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``post_trained_model`` with every matched layer's weight
            replaced by its delta-aware FP8 round trip (a plain new float32
            initializer -- no graph nodes and no opset requirement, exactly
            :func:`onnxsim.apply_deepseek_fp8`'s own weight-side shape).
            A layer with no ``base_model`` counterpart, a counterpart of a
            different op type, a non-constant/non-2-D/non-float32 weight on
            either side, a shape mismatch between the two weights, or an
            all-zero ``ΔW`` is left completely untouched.
    :raises ValueError: if ``metric`` is neither ``"cosine"`` nor
            ``"sign_preservation"``

    Delegates to the verified C++ port (:func:`onnxsim.apply_daq_cpp`).
    """
    return apply_daq_cpp(base_model, post_trained_model, metric, skip_names)
