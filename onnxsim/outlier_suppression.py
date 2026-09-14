"""Outlier Suppression (Wei, Zhang, Zhang, Gong, Zhang, Zhang, Chi, Yuan and
Liu, 2022, "Outlier Suppression: Pushing the Limit of Low-bit Transformer
Language Models", NeurIPS 2022, https://arxiv.org/abs/2209.13325). The
original paper this repo's own :mod:`onnxsim.outlier_suppression_plus`
extends -- that module's own docstring already flags this earlier paper as
"distinct and not what this module ports"; this module ports it.

:mod:`onnxsim.smoothquant` and :mod:`onnxsim.outlier_suppression_plus` both
migrate activation quantization difficulty into the weight via a per-channel
scale, realized by inserting a new ``Mul`` node (and, for OS+, ``Sub``/``Add``
too) right before the consuming MatMul/Gemm. Outlier Suppression's own
"Gamma Migration" does the same *kind* of per-channel scale migration, but
realizes it completely differently when the activation being scaled is a
``LayerNormalization``'s own output (transformers' by far most common
producer of a Linear layer's input): instead of inserting a node, it folds
the scale directly into the LayerNormalization's own affine parameters,
adding **zero** runtime nodes at all.

The algebra: ``LayerNormalization`` computes
``out = normalize(x) * gamma + beta`` (elementwise per channel, ``beta``
optional). Dividing the *whole* affine output by a per-channel scale ``s``
is itself just

    out / s = normalize(x) * (gamma / s) + (beta / s)

-- i.e. exactly what a LayerNormalization with ``gamma' = gamma / s`` and
``beta' = beta / s`` already computes, with no new node needed. Scaling a
downstream consumer's weight rows by the same ``s`` (:mod:`onnxsim.
smoothquant`'s own compensating step) then makes the composition exact,
identical in spirit to how :mod:`onnxsim.smoothquant` compensates its own
inserted ``Mul``, but here the "insertion" costs nothing at runtime because
the LayerNormalization node already existed and already had to compute an
affine transform anyway.

This is a narrower structural target than :mod:`onnxsim.smoothquant`'s own
"any MatMul/Gemm with a 2-D activation input" -- gamma migration is only
correct when the LayerNormalization's output has **no other consumer**
besides the MatMul/Gemm layers being compensated (dividing its output by
``s`` changes what *every* consumer of that tensor sees, and an
uncompensated consumer -- e.g. a residual ``Add``, or the LayerNormalization
output being a graph output itself -- would silently see the wrong,
scaled-down activation). This module therefore only migrates a
``LayerNormalization`` node whose output feeds exclusively into one or more
plain MatMul/vanilla-Gemm nodes (as their activation input) and is not
itself a graph output -- exactly the shape a transformer's own QKV or MLP
input projection takes (one shared pre-projection ``LayerNorm`` feeding
several parallel `Linear`s, or one feeding a single `Linear`, and nothing
else), which is also the paper's own primary target. A LayerNormalization
with any other kind of consumer is left completely untouched, not partially
migrated.

The per-channel scale itself reuses :mod:`onnxsim.smoothquant`'s own
closed-form, alpha-parameterized formula (``s_j = max(|X_j|) ** alpha /
max(|W_j|) ** (1 - alpha)``, maximized over every compensated consumer's own
weight when there is more than one), matching that module's own documented
practice of a single fixed ``alpha`` rather than a per-layer search.
Deliberately not ported: the paper's own second contribution, "Token-Wise
Clipping" (a searched, rather than plain min-max/entropy, activation
clipping range for the subsequent quantizer) -- a calibration-range-search
technique orthogonal to this module's own scale migration, and out of this
module's own scope (:func:`onnxsim.calibrate`'s existing ``"minmax"``/
``"entropy"`` methods already cover the "how to pick a clip range" question
this repo answers elsewhere).

Like :mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression_plus`,
this only performs the *migration*: :func:`apply_outlier_suppression`
returns a float model, provably equivalent to the input up to floating-point
rounding -- no quantization happens here. The result is meant to be fed to a
W8A8 quantizer afterwards (e.g. :func:`onnxsim.quantize_static`).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_outlier_suppression(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies Outlier Suppression's "Gamma Migration" to every
    ``LayerNormalization`` node whose output feeds exclusively into one or
    more plain MatMul/vanilla-Gemm layers (and is not itself a graph
    output), using real calibration activations. See this module's own
    docstring for the technique. Returns a float model -- pass the result
    to a W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`) to actually
    quantize it.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            LayerNormalization output channel's activation range on. Each
            batch is a ``{input_name: np.ndarray}`` dict matching
            ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a much more representative migration than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength, identical in meaning to
            :func:`onnxsim.apply_smoothquant`'s own ``alpha``
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the scale, avoiding a divide-by-
            zero on an all-zero channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched ``LayerNormalization``'s
            ``scale``/``bias`` initializers divided by the migration scale
            in place, and every compensated consumer's weight rows
            multiplied by the same scale in place -- no new nodes are ever
            inserted. A ``LayerNormalization`` with any consumer other than
            a plain MatMul/vanilla-Gemm (as its activation input), or whose
            output is itself a graph output, is left completely untouched.

    This pure-Python implementation has been retired in favor of the
    verified-bit-exact C++ port -- this is now a thin alias for
    :func:`onnxsim.apply_outlier_suppression_cpp`
    (``onnxsim/outlier_suppression_entry.cpp``'s own
    ``ApplyOutlierSuppression``), forwarding every argument unchanged.
    Exact (bit-for-bit) parity was verified against this function's own
    pre-alias implementation across single/multi-consumer topologies,
    transB-Gemm consumers, bias-free LayerNormalization, rank-3
    activations, every decline shape, empty calibration, and alpha in {0,
    0.25, 0.5, 0.75, 1} -- see tests/test_outlier_suppression_cpp.py --
    before this alias was made. Imported lazily (inside the function body,
    not at module scope) to avoid a circular import:
    ``onnxsim.onnx_simplifier`` already imports from this module, so
    importing it back at module load time here would deadlock the import
    machinery.
    """
    from onnxsim.onnx_simplifier import apply_outlier_suppression_cpp

    return apply_outlier_suppression_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        alpha=alpha,
        epsilon=epsilon,
        providers=providers,
    )
