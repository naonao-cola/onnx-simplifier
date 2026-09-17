"""OWQ (Lee, Park, Kim, Kim and Sung, 2023, "OWQ: Outlier-Aware Weight
Quantization for Efficient Fine-Tuning and Inference of Large Language
Models", https://arxiv.org/abs/2306.02272; AAAI 2024). A fifth lever
targeting :func:`onnxsim.quantize_weight_only_int4`'s output, alongside
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.awq`/
:mod:`onnxsim.quantease` -- but unlike all four of those (which only ever
change *which integer* an already-fixed-scale quantizer rounds a column to),
OWQ instead rescues a small number of columns from quantization **entirely**,
restoring them to exact float32 precision via a correction term, based on a
different notion of "sensitive" than :mod:`onnxsim.awq`'s or
:mod:`onnxsim.spqr`'s own.

Three onnxsim modules already single out specific weight/activation
elements as needing special treatment, each by a different signal:
:mod:`onnxsim.llm_int8` excludes *activation* columns whose calibration
magnitude exceeds a fixed threshold from INT8 entirely (runtime-observed,
activation-side); :mod:`onnxsim.spqr` extracts individual outlier *weight
values* (not whole columns) into a sparse correction, by how far each
element's own error deviates from its layer's typical error; :mod:`onnxsim.
awq` rescales -- never excludes -- whole input channels in proportion to
their own average activation magnitude. OWQ's own signal is different from
all three: the classic Optimal Brain Surgeon (OBS) saliency metric --

    sensitivity_j = mean_n[(W[n, j] - RTN(W[n, j]))^2] / [H^-1]_jj

where the numerator is column ``j``'s own existing round-to-nearest
quantization error, squared and averaged over output channels ``n`` (a
per-column scalar proxy for the paper's own ``error_j^2`` term), and
``H = X^T X`` is the same calibration-derived Hessian :mod:`onnxsim.gptq`
already computes (this module reuses its ``_inverse_hessian_cholesky``
directly). ``[H^-1]_jj``
measures how much *other* columns could compensate for column ``j``'s own
error if it were left as-is (a small value means little compensation
capacity elsewhere, i.e. column ``j``'s own error matters more directly to
the layer's output) -- the same metric the original Optimal Brain Surgeon
pruning method (LeCun et al./Hassibi & Stork) uses to rank which weights to
remove, applied here to rank which *columns* are least safe to quantize.

For the top ``outlier_fraction`` columns by this score (the paper's own
default is a small fraction, ~0.1-1%), this module does not change how they
are quantized at all -- ``quantized_model``'s INT4 codes are left completely
untouched, including for those columns. Instead, it inserts a *correction*
term at graph-run time: ``Gather`` those columns of the activation, multiply
by a precomputed ``(W_float - W_rtn)`` residual (an exact, static
initializer), and ``Add`` the result to the layer's output. ``W_rtn`` here
is unpacked directly from ``quantized_model``'s own existing INT4
initializer -- not recomputed from scratch in Python -- specifically so the
residual is exact against whatever ``quantized_model`` actually contains:
recomputing round-to-nearest independently (the way, e.g.,
:mod:`onnxsim.adaround` does for its own starting point) can differ from
the real codes by a single rounding tie at a bin boundary, which would
silently turn this module's "exact restoration" claim into an
approximation off by that tie's own quantization step. -- the same "rename producer output, insert an op reproducing the
original name" mechanics :mod:`onnxsim.bias_correction`'s
``_apply_correction`` and :mod:`onnxsim.outlier_suppression_plus` already
use. Because the correction term is exactly ``Gather(X, weak) @ (W_float -
W_rtn)[weak]^T``, adding it to the INT4 branch's own output makes the
selected columns' contribution to the layer's output *exactly* what the
float model would have produced -- not an approximation, a full restoration
-- while every other column stays INT4-quantized as before, so this module
never needs to touch or resize any existing packed INT4 initializer.

Deliberately not ported: the paper's own additional fine-tuning step (a
short LoRA-style adaptation *after* the weak-column split, to recover
further accuracy) -- out of scope for the same reason QAT is throughout
this repo (see ``docs/nncf-comparison-future-work.md``'s own "Explicitly
out of scope" section): it needs a training loop, not a graph rewrite.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_owq_cpp


def apply_owq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    outlier_fraction: float = 0.01,
    percdamp: float = 0.01,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Restores OWQ's most quantization-sensitive input columns (by the
    classic Optimal Brain Surgeon saliency metric) to exact float32
    precision, for every ``quantize_weight_only_int4``-quantized MatMul/Gemm
    layer present (by node output name) in both ``float_model`` and
    ``quantized_model``, using real activations captured from
    ``float_model``. See this module's own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to compute each
            layer's Hessian and per-column error from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative Hessian than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param outlier_fraction: fraction of each layer's input columns to
            restore to full precision (the paper's own default range is
            roughly 0.1% to 1%), rounded to at least 1 column
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :mod:`onnxsim.gptq`'s own default -- keeps the inversion
            numerically stable when calibration data doesn't fully activate
            (or correlates too tightly across) a layer's input channels
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with a new ``Gather``/``MatMul``/``Add``
            correction inserted after every matched layer, restoring its
            most sensitive input columns' contribution to exact float32
            precision; the layer's own INT4 codes are never modified. A
            layer with fewer calibration-observed columns than needed for a
            meaningful split, or that OWQ found no columns worth restoring
            for (``outlier_fraction`` rounds to 0), is left untouched.

    Delegates to the verified C++ port (:func:`onnxsim.apply_owq_cpp`),
    which takes the exact same parameters -- no gap to bridge here.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    return apply_owq_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        outlier_fraction=outlier_fraction,
        percdamp=percdamp,
        providers=providers,
    )
