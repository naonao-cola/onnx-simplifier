"""AdpQ (Ghaffari et al., 2024, "AdpQ: A Zero-shot Calibration Free Adaptive
Post Training Quantization Method for LLMs", https://arxiv.org/abs/2405.13358).

This repo's other salient/non-salient weight splitters --
:mod:`onnxsim.owq`, :mod:`onnxsim.spqr`, :mod:`onnxsim.gptq`,
:mod:`onnxsim.billm` -- all need real calibration *activations* to decide
which weights matter: each builds (a version of) the layer's Hessian
``H = X^T X`` from a batch of representative inputs, then ranks
sensitivity by some function of ``H`` (or its inverse). AdpQ's own headline
contribution is doing the same kind of split -- separate a small "salient"
set of weights from the rest, quantize only the rest to a low-bit grid --
**with no calibration data whatsoever**. Its closest sibling in spirit is
instead :mod:`onnxsim.nf4`: both decide every quantization choice purely
from a weight tensor's own values, nothing else. Where they part ways is
what that means in practice -- NF4 quantizes every element onto the same
fixed, data-independent 16-point codebook; AdpQ still does a salient/
non-salient *split* (like OWQ/SpQR/GPTQ/BiLLM), just deciding it from the
weight's own magnitude distribution rather than a Hessian.

**Picking which elements are salient.** AdpQ borrows its threshold rule
from Adaptive LASSO regression (Zou, 2006): ordinary LASSO soft-thresholds
every coefficient by the same fixed ``lambda``; Adaptive LASSO instead
scales that threshold per-coefficient by (a power of) that coefficient's
own estimated scale, so naturally larger-magnitude groups keep
comparatively more of their mass. This module applies the same idea
per-group along a layer's reduction dimension (groups of ``group_size``
elements, matching :func:`onnxsim.quantize_weight_only_int4`'s own
block-wise convention): for each ``(output channel, group)`` slice, a
robust scale estimate

    sigma_hat = 1.4826 * MAD(w_group)

(the median absolute deviation, scaled by the usual constant that makes it
a consistent estimator of the standard deviation for normally-distributed
data -- robust because a handful of large weights in the group can't drag a
*median*-based estimate the way they would a max-abs or plain-std one), and
an adaptive soft-threshold derived from it,

    threshold = lambda_ * sigma_hat ** (1 - gamma)

directly mirroring Adaptive LASSO's own per-coefficient threshold
``lambda * scale^(1-gamma)``. ``gamma in [0, 1)`` is the adaptive weighting
exponent: at ``gamma = 0`` this is just an ordinary constant multiple of
each group's own robust sigma (a plain robust z-score threshold); as
``gamma`` grows, the exponent on ``sigma_hat`` shrinks towards 0 and the
threshold flattens out towards the constant ``lambda_`` regardless of the
group's own scale -- i.e. groups with a naturally wider spread get
comparatively *more* of their mass counted salient, the same
scale-adaptive asymmetry Adaptive LASSO's weighting scheme produces for
regression coefficients. Every element whose magnitude exceeds its own
group's threshold is salient; the rest are not. Unlike
:mod:`onnxsim.spqr`'s ``outlier_fraction`` (a fixed target count picked in
advance), the number of salient elements here falls out of the threshold
rule itself and can differ from group to group and layer to layer -- the
same "adaptive" character the paper's own name refers to.

**Minimizing the weight distribution's own KL divergence, not the layer's
output error.** Every non-salient group is quantized onto a uniform
symmetric INT4 grid, using a scale computed *excluding* that group's own
salient elements (the same "exclude the outliers from the scale, so
everything else quantizes tighter" trick :mod:`onnxsim.spqr` uses for its
own per-element outliers) -- this keeps the quantized sub-population's own
empirical distribution close to the original weights' distribution
restricted to that same sub-population, rather than optimizing (as
:mod:`onnxsim.gptq`/:mod:`onnxsim.owq` do) for the layer's *output*
reconstruction error against real activations, since no activations are
available here to optimize against in the first place. Salient elements are
restored to exact float32 precision via a sparse correction overlay --
``W_reconstructed = block_quantized(W) + sparse_correction`` -- the same
``ScatterND``-over-``ConstantOfShape`` mechanics :mod:`onnxsim.spqr` already
uses, chosen here (over e.g. a separate INT8 sub-tensor for salient
elements) because it needs no second quantization grid or extra dequant
branch: salient elements simply reconstruct exactly, non-salient elements
reconstruct through the INT4 grid, and both paths share one
``DequantizeLinear``.
"""

from __future__ import annotations

from typing import Union

import onnx

from onnxsim.onnx_simplifier import apply_adpq_cpp


def quantize_weight_only_adpq(
    model: Union[str, onnx.ModelProto],
    group_size: int = 128,
    lambda_: float = 3.0,
    gamma: float = 0.3,
) -> onnx.ModelProto:
    """Applies AdpQ-style calibration-free adaptive INT4 quantization (see
    this module's own docstring) to every MatMul/vanilla-Gemm layer with a
    constant 2-D float32 weight whose reduction dimension ``K`` is divisible
    by ``group_size``. Needs no calibration data: every quantization
    decision -- which elements are salient, and the scale used for the
    rest -- comes from the weight tensor's own values.

    Delegates to the verified C++ port (:func:`onnxsim.apply_adpq_cpp`),
    which hardcodes this function's own defaults (``group_size=128``,
    ``lambda_=3.0``, ``gamma=0.3``) and folds the round trip directly into
    a replacement float32 initializer rather than building a real
    ``DequantizeLinear``+``ScatterND``+``Add`` graph rewrite -- a
    storage-format change from this function's own former behavior, not a
    numeric one.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param group_size: elements per quantization group along ``K``, matching
            :func:`onnxsim.quantize_weight_only_int4`'s own block-wise
            granularity. **The C++ port's own only supported value is 128**
            -- a non-default value raises ``ValueError``.
    :param lambda_: overall threshold scale -- at ``gamma=0`` this is the
            number of robust sigmas (see this module's own docstring) above
            which a weight counts as salient. **Not supported by the C++
            port** -- a non-default value raises ``ValueError``.
    :param gamma: Adaptive LASSO exponent in ``[0, 1)`` controlling how much
            a group's own scale flattens the threshold (see this module's
            own docstring). **Not supported by the C++ port** -- a
            non-default value raises ``ValueError``.
    :returns: ``model`` with every matched layer's weight replaced by its
            AdpQ-quantized float32 version, stored under a new
            initializer. Layers with a non-constant, non-2-D weight, or a
            reduction dimension not divisible by ``group_size``, are left
            untouched; a model with no matching layer, or an opset older
            than 21, is returned unchanged.
    """
    if group_size != 128 or lambda_ != 3.0 or gamma != 0.3:
        raise ValueError(
            "quantize_weight_only_adpq now delegates to apply_adpq_cpp, "
            "which hardcodes group_size=128, lambda_=3.0, gamma=0.3 and "
            "cannot honor other values"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_adpq_cpp(model)
