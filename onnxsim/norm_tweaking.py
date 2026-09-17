"""Norm Tweaking (Li, Xu, Ni, Chen, Ye, Sun, 2023, "Norm Tweaking:
High-performance Low-bit Quantization of Large Language Models",
https://arxiv.org/abs/2309.02784). onnxsim ports the algorithm, not any
framework's code, per the same rationale as :mod:`onnxsim.awq`/
:mod:`onnxsim.gptq` (the paper's own reference implementation tweaks live
PyTorch ``nn.LayerNorm`` modules with no ONNX export path).

Every weight-quantization pass already in onnxsim -- ``quantize_weight_only_
int4`` and everything built on it -- changes what a MatMul/Gemm/Conv
computes, but leaves every LayerNormalization node in the graph completely
untouched: its own ``scale``/``bias`` parameters were fit (during original
model training) to the *float* activation distribution flowing into it, and
nothing about weight quantization updates them to match the now-shifted
distribution a quantized upstream layer actually produces. Norm Tweaking's
own observation: a LayerNormalization node's ``scale``/``bias`` are exactly
the right (and only) knobs to correct that shift with, because they're the
very last operation before the corrected distribution needs to be correct,
and the correction is nearly free -- one channel-wise scale and one
channel-wise shift per LayerNormalization node, no gradient descent, no
extra graph nodes, unlike a full reconstruction pass (:mod:`onnxsim.adaround`
/:mod:`onnxsim.gptq`) or an inserted correction op (:mod:`onnxsim.
bias_correction`, which adds a new ``Add`` node after a Conv/Gemm/MatMul
instead of ever touching an existing parameter).

This module's own version of the technique (a reproduction of the paper's
own described mechanism -- match the quantized model's LayerNorm *output*
distribution back to the float model's own, per channel -- not a
transcription of the paper's own reported per-model results, which this
module does not claim to reproduce): for every ``LayerNormalization`` node
present (by output tensor name) in both ``float_model`` and
``quantized_model``, this runs both models on the same calibration data and
measures that node's own output tensor's per-channel (last axis) mean
``mu`` and standard deviation ``sigma``, in both models. Because
``LayerNormalization``'s own definition is
``scale * normalize(x) + bias`` (``normalize`` is exactly mean-0/std-1 per
instance, so ``scale``/``bias`` are the *only* remaining source of any
distributional difference in a well-formed graph), a single closed-form
per-channel affine transform recovers the float distribution exactly on the
calibration data seen: solving
``alpha * quantized_output + beta == float_output`` for matching first and
second moments gives ``alpha = sigma_float / sigma_quantized`` and
``beta = mu_float - alpha * mu_quantized``, and because
``alpha * (scale * normalize(x) + bias) + beta
    == (alpha * scale) * normalize(x) + (alpha * bias + beta)``,
that correction folds directly into a new ``scale`` / ``bias`` for the same
node -- exactly the paper's own "tweak the norm's own parameters in place"
mechanism, with no extra graph nodes needed at all (unlike this repo's own
:func:`onnxsim.correct_bias`, whose correction targets a Conv/Gemm/MatMul's
additive output bias instead, a different operator with no built-in
post-normalization affine to fold into).

**Scope note**: only ``LayerNormalization`` nodes (opset 17+'s single fused
op) with a 1-D ``scale`` (and optional ``bias``) whose length matches the
node's own output last-axis size are handled -- the overwhelmingly common
shape for a transformer's per-token normalization (``axis=-1``, the
default). A LayerNorm normalizing over more than one trailing axis, or a
graph still using the older ``ReduceMean``/``Sub``/``Pow``/... decomposition
instead of the fused op, is left untouched rather than guessed at.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_norm_tweaking_cpp


def apply_norm_tweaking(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> onnx.ModelProto:
    """Recalibrates every matched ``LayerNormalization`` node's own
    ``scale``/``bias`` parameters in ``quantized_model`` so its output
    distribution's per-channel mean and standard deviation, measured on
    ``calibration_data``, matches ``float_model``'s own -- see this
    module's own docstring for the closed-form derivation.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), e.g. from
            :func:`onnxsim.quantize_weight_only_int4` or any ``quantize_*``
            function. Assumes ``quantized_model`` was produced from
            ``float_model`` without renaming any ``LayerNormalization``
            node's own output tensor -- true of every onnxsim ``quantize_*``
            function.
    :param calibration_data: representative input batches to measure each
            node's own float-vs-quantized output distribution on -- see
            :func:`onnxsim.correct_bias`'s own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :param eps: added to the measured quantized-side standard deviation
            before dividing, to avoid blowing up on a (near-)constant
            channel
    :returns: ``quantized_model`` with every matched ``LayerNormalization``
            node's ``scale``/``bias`` initializers replaced by tweaked
            copies (new initializers, uniquely named -- the originals are
            left in the model only if some other node still references
            them)

    Delegates to :func:`onnxsim.apply_norm_tweaking_cpp` (the verified C++
    port, which has full parameter parity with this function -- see
    ``norm_tweaking_entry.h`` for its own scope note); this pure-Python
    name is kept only for backward compatibility with existing callers.
    """
    return apply_norm_tweaking_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        providers=providers,
        eps=eps,
    )
