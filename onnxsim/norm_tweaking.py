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

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data


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
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    quantized_by_output: Dict[str, onnx.NodeProto] = {}
    for n in quantized_model.graph.node:
        if n.op_type == "LayerNormalization" and n.output:
            quantized_by_output[n.output[0]] = n

    q_initializers = {t.name: t for t in quantized_model.graph.initializer}

    candidates: List[Tuple[str, onnx.NodeProto]] = []
    for n in float_model.graph.node:
        if n.op_type != "LayerNormalization" or not n.output:
            continue
        q_node = quantized_by_output.get(n.output[0])
        if q_node is None or len(q_node.input) < 2:
            continue
        scale_init = q_initializers.get(q_node.input[1])
        if scale_init is None or len(scale_init.dims) != 1:
            continue
        candidates.append((n.output[0], q_node))
    if not candidates:
        return quantized_model

    names = [name for name, _ in candidates]
    float_probe = _add_probe_outputs(float_model, names)
    quantized_probe = _add_probe_outputs(quantized_model, names)

    f_sum: Dict[str, np.ndarray] = {}
    f_sumsq: Dict[str, np.ndarray] = {}
    q_sum: Dict[str, np.ndarray] = {}
    q_sumsq: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    for batch in calibration_data:
        f_out = backend.run_model(float_probe, batch, providers=providers)
        q_out = backend.run_model(quantized_probe, batch, providers=providers)
        for name in names:
            f = np.asarray(f_out[name], dtype=np.float64)
            q = np.asarray(q_out[name], dtype=np.float64)
            if f.shape != q.shape or f.ndim == 0:
                continue
            channels = f.shape[-1]
            f2 = f.reshape(-1, channels)
            q2 = q.reshape(-1, channels)
            if name in counts:
                f_sum[name] += f2.sum(axis=0)
                f_sumsq[name] += np.square(f2).sum(axis=0)
                q_sum[name] += q2.sum(axis=0)
                q_sumsq[name] += np.square(q2).sum(axis=0)
                counts[name] += f2.shape[0]
            else:
                f_sum[name] = f2.sum(axis=0)
                f_sumsq[name] = np.square(f2).sum(axis=0)
                q_sum[name] = q2.sum(axis=0)
                q_sumsq[name] = np.square(q2).sum(axis=0)
                counts[name] = f2.shape[0]

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    taken_names = _all_names(corrected.graph)
    initializer_index = {t.name: i for i, t in enumerate(corrected.graph.initializer)}
    node_by_output = {
        n.output[0]: n
        for n in corrected.graph.node
        if n.op_type == "LayerNormalization" and n.output
    }

    for name in names:
        if name not in counts:
            continue
        n = counts[name]
        mu_f = f_sum[name] / n
        var_f = np.maximum(f_sumsq[name] / n - mu_f**2, 0.0)
        sigma_f = np.sqrt(var_f)
        mu_q = q_sum[name] / n
        var_q = np.maximum(q_sumsq[name] / n - mu_q**2, 0.0)
        sigma_q = np.sqrt(var_q)

        alpha = sigma_f / (sigma_q + eps)
        beta = mu_f - alpha * mu_q

        q_node = node_by_output[name]
        scale_init = corrected.graph.initializer[initializer_index[q_node.input[1]]]
        old_scale = onnx.numpy_helper.to_array(scale_init).astype(np.float64)
        new_scale = (old_scale * alpha).astype(np.float32)
        new_scale_name = _unique_name(f"{name}_norm_tweak_scale", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(new_scale, name=new_scale_name)
        )
        q_node.input[1] = new_scale_name

        if len(q_node.input) >= 3 and q_node.input[2]:
            old_bias_init = corrected.graph.initializer[
                initializer_index[q_node.input[2]]
            ]
            old_bias = onnx.numpy_helper.to_array(old_bias_init).astype(np.float64)
            new_bias = (alpha * old_bias + beta).astype(np.float32)
        else:
            new_bias = beta.astype(np.float32)
        new_bias_name = _unique_name(f"{name}_norm_tweak_bias", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(new_bias, name=new_bias_name)
        )
        if len(q_node.input) >= 3:
            q_node.input[2] = new_bias_name
        else:
            q_node.input.append(new_bias_name)

    return corrected
