"""Outlier Suppression+ (Wei et al., 2023, "Outlier Suppression+: Accurate
quantization of large language models by equivalent and optimal shifting and
scaling", EMNLP 2023, https://arxiv.org/abs/2304.09145). Its predecessor,
Outlier Suppression (the same authors' earlier NeurIPS 2022 paper), is
distinct and not what this module ports. onnxsim ports the *algorithm*, not
any framework's code, per the same rationale as :mod:`onnxsim.smoothquant`.

:mod:`onnxsim.smoothquant` migrates per-channel activation quantization
difficulty into the weight via a single elementwise *scale* -- ``s_j``
shrinks activation channel ``j``'s range at the cost of expanding weight row
``j``'s range. That works well for channels whose outliers are large in
*magnitude* but roughly *symmetric* around zero. Outlier Suppression+'s
observation: many transformer activation channels (especially post-LayerNorm)
carry a large, consistent *asymmetric* component too -- a channel sitting
mostly on one side of zero -- which a symmetric scale cannot address at all
(scaling a lopsided range by any positive constant leaves it just as
lopsided), forcing a symmetric quantizer's range to cover the channel's full
excursion from its own worst-case value down to (or up from) zero.

Outlier Suppression+ therefore adds a **channel-wise shift** ahead of
SmoothQuant's own scale: for each input channel ``j``,

    z_j = (max(X_j) + min(X_j)) / 2

(the midpoint of that channel's observed calibration range) is subtracted
from ``X_j`` before scaling, re-centering it around zero and roughly halving
the range a symmetric quantizer must cover for a channel that was originally
one-sided. Shifting an affine layer's input is not, on its own, a free
transformation the way scaling is -- ``(X - z) @ W`` differs from ``X @ W``
by exactly ``z @ W``, a *constant* (calibration-independent, per-output-
-channel) vector -- so this module folds that constant back in as an
additive correction on the layer's own output, via a new ``Add`` node
inserted right after it (the same "measure a per-channel constant, fold it
back in with an ``Add``" mechanics :func:`onnxsim.correct_bias` already
uses, though what's being corrected for here is an exact algebraic identity
from the shift, not an empirically-measured quantization error). The result:
``Y = ((X - z) / s) @ (W * s)^T + z @ W^T`` is *exactly* ``X @ W^T`` for any
choice of ``z``/``s`` (up to floating-point rounding) -- provably so, not
just approximately, since every step is a linear re-parameterization of the
same affine map, never an approximation.

For the scaling step itself, this module reuses
:mod:`onnxsim.smoothquant`'s own closed-form, alpha-parameterized formula
(``s_j = max(|X_j - z_j|) ** alpha / max(|W_j|) ** (1 - alpha)``), applied to
the now-*shifted* activation rather than the raw one, matching
:mod:`onnxsim.smoothquant`'s own documented practice of using a single fixed
``alpha`` rather than a per-layer search. The Outlier Suppression+ paper
additionally proposes its own iterative grid refinement on top of that
formula to squeeze out a further, typically small, improvement over the
plain closed form -- that refinement is not reproduced here; what onnxsim
ports is the paper's headline structural contribution over SmoothQuant (the
shift), not its secondary scale-search refinement.

Like :func:`onnxsim.apply_smoothquant`, this only performs the *migration*:
:func:`apply_outlier_suppression_plus` returns a float model, provably
equivalent to the input up to floating-point rounding -- no quantization
happens here. The result is meant to be fed to a W8A8 quantizer afterwards
(e.g. :func:`onnxsim.quantize_static`/:func:`onnxsim.quantize_qoperator_gemm`),
exactly how :func:`onnxsim.apply_smoothquant` is used.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_outlier_suppression_plus(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies Outlier Suppression+'s channel-wise shifting and scaling to
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight and a
    plain 2-D activation input, using real calibration activations. See this
    module's own docstring for the technique. Returns a float model -- pass
    the result to a W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`) to
    actually quantize it.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation range on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative migration than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the scaling step's migration strength, identical in
            meaning to :func:`onnxsim.apply_smoothquant`'s own ``alpha``
            (applied to the *shifted* activation's range rather than the
            raw activation's)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the scale, avoiding a divide-by-
            zero on an all-zero (or exactly-centered) channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight columns rescaled
            in place, a new ``Mul``+``Sub`` pair inserted before it applying
            the shift and scale to its activation input, and a new ``Add``
            inserted after it restoring the shift's constant contribution to
            the output; layers with a non-constant, non-2-D weight, or whose
            activation input isn't a plain 2-D tensor matching the weight's
            reduction dimension, are left untouched

    This pure-Python implementation has been retired in favor of the
    verified-bit-exact C++ port -- this is now a thin alias for
    :func:`onnxsim.apply_outlier_suppression_plus_cpp`
    (``onnxsim/outlier_suppression_plus_entry.cpp``'s own
    ``ApplyOutlierSuppressionPlus``), forwarding every argument unchanged.
    Exact (bit-for-bit) parity was verified against this function's own
    pre-alias implementation across MatMul/Gemm/transB-Gemm/biased-Gemm,
    shared activations, lopsided and all-negative channels, multi-batch
    ranges, every skip shape, empty calibration, and alpha in {0, 0.25,
    0.5, 0.75, 1} -- see tests/test_outlier_suppression_plus_cpp.py --
    before this alias was made. Imported lazily (inside the function body,
    not at module scope) to avoid a circular import:
    ``onnxsim.onnx_simplifier`` already imports from this module, so
    importing it back at module load time here would deadlock the import
    machinery.
    """
    from onnxsim.onnx_simplifier import apply_outlier_suppression_plus_cpp

    return apply_outlier_suppression_plus_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        alpha=alpha,
        epsilon=epsilon,
        providers=providers,
    )
