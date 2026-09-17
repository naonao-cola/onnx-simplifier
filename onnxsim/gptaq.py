"""GPTAQ (Li, Yin, Lee, Xiao, Panda, 2025, "GPTAQ: Efficient Finetuning-Free
Quantization for Asymmetric Calibration", https://arxiv.org/abs/2504.02692)
-- a small, closed-form correction to :mod:`onnxsim.gptq` that this module
re-derives from first principles below rather than transcribing the paper's
own notation (the paper states the result but not every intermediate step;
the derivation here is onnxsim's own, checked algebraically, not a
transcription of the authors' proof).

The problem GPTAQ points out: real GPTQ quantizes a network layer by layer,
so by the time layer ``i`` is quantized, the activations flowing into it
already come from every *earlier* layer's own (already-quantized) weights
-- call that corrupted activation ``X``. But GPTQ's own per-column
objective (minimize ``||W X^T - Ŵ X^T||²``, ``W`` float, ``Ŵ`` quantized)
implicitly targets reconstructing *that corrupted signal*, not what the
original float network would have actually produced there (call the true,
never-corrupted activation ``X̃``). GPTQ calibrated this way is quietly
optimizing against its own accumulated error instead of correcting for it.

:mod:`onnxsim.gptq` sidesteps this by construction -- it always captures
activations from ``float_model`` alone, so every layer already calibrates
against ``X̃``, not a corrupted ``X`` at all. GPTAQ's asymmetric-calibration
idea is nonetheless available to onnxsim specifically *because* it already
threads both ``float_model`` and ``quantized_model`` through every
``apply_*`` correction pass here: this module captures ``X̃`` from
``float_model`` exactly like GPTQ, but *additionally* captures ``X`` at the
same probe point from ``quantized_model`` (whatever it was already
quantized/corrected by), and folds the gap between them into GPTQ's own
per-column procedure via one small, exact pre-computation.

Derivation. Write ``δX = X̃ - X`` (the accumulated upstream corruption at
this layer's input, fixed and independent of this layer's own weight
quantization). For one output row ``w`` (float) / ``ŵ`` (quantized) and
error ``e = w - ŵ``:

```
w X̃^T - ŵ X^T = w (X + δX)^T - (w - e) X^T = w δX^T + e X^T
```

so the per-row squared objective ``||w δX^T + e X^T||²`` expands to
``e H e^T + 2 c^T e^T + const`` with ``H = X^T X`` (GPTQ's own Hessian,
computed here from the *quantized*-model's activations, since that is
what multiplies ``e`` above) and the new linear term's coefficient
``c = X^T (δX w^T)`` -- a fixed, precomputable vector, since it only
depends on the already-known float weight row ``w`` and the fixed
activations, not on ``ŵ``. Completing the square (dropping the resulting
constant, which does not depend on ``ŵ``) shows this is *exactly* GPTQ's
own quadratic objective ``e' H e'^T``, but for a shifted error
``e' = (w + shift) - ŵ`` where ``shift = (H^{-1} c)^T`` -- i.e., GPTAQ is
GPTQ's own column algorithm, applied unchanged, to the weight matrix
``W + Shift`` instead of ``W``. This matches the paper's own description
of the fix as one small, closed-form residual term layered on top of
GPTQ, not a different algorithm.

When a candidate layer has no upstream quantization yet (``X == X̃``,
``δX ≈ 0``), ``Shift`` is (numerically) zero and this module's output
matches plain GPTQ's exactly -- the expected degenerate case, and one this
module's own tests check directly.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_gptaq_cpp


def apply_gptaq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes GPTAQ-style (asymmetric-calibration) sequential,
    Hessian-compensated rounding for every ``quantize_weight_only_int4``-
    quantized MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``. See this module's own
    docstring for the technique and its relationship to
    :func:`onnxsim.apply_gptq`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`, optionally already
            refined by other passes (e.g. :func:`onnxsim.apply_gptq`,
            :func:`onnxsim.correct_bias`) -- this is exactly what makes the
            asymmetric calibration meaningful: ``quantized_model``'s own
            activations at a candidate layer's input reflect whatever
            upstream layers' quantization already did. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches, run through
            *both* models to capture the true (``float_model``) and
            corrupted (``quantized_model``) activation at each candidate's
            input -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data`.
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.apply_gptq`'s own parameter and default
    :param proc_block_size: GPTQ's own column-processing block size, passed
            through unchanged to :func:`onnxsim.apply_gptq`'s internals
    :param providers: onnxruntime execution providers to run both models on
            when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its GPTAQ-optimized codes (same shape,
            dtype, and scale -- only which integer each element rounds to
            changes)

    Delegates to :func:`onnxsim.apply_gptaq_cpp` (the verified C++ port,
    which has full parameter parity with this function -- see
    ``gptaq_entry.h`` for its own scope note); this pure-Python name is
    kept only for backward compatibility with existing callers.
    """
    return apply_gptaq_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        percdamp=percdamp,
        proc_block_size=proc_block_size,
        providers=providers,
    )
