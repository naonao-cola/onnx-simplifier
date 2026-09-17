"""QuantEase (Behdin, Acharya, Gupta, Song, Zhu and Keerthi, 2023,
"QuantEase: Optimization-based Quantization for Language Models",
https://arxiv.org/abs/2309.01885) -- a fourth lever on
:func:`onnxsim.quantize_weight_only_int4`'s own block-wise symmetric INT4
scheme, alongside :mod:`onnxsim.adaround` (per-element additive rounding
relaxation, gradient descent), :mod:`onnxsim.gptq` (sequential, Hessian-
compensated greedy rounding, one pass, sorted left to right), and
:mod:`onnxsim.awq` (whole-input-channel rescaling before quantizing).
onnxsim ports the *algorithm*, not any framework's code, per the same
rationale as those modules' own docstrings.

Like :mod:`onnxsim.gptq`, QuantEase starts from the same layer
reconstruction objective: for weight ``W`` ([N, K], output channel first)
and calibration activations ``X`` ([samples, K]), minimizing
``||X @ W^T - X @ Ŵ^T||^2`` over quantized ``Ŵ`` is, per output row ``n``
independently (the Hessian ``H = X^T X``, a ``[K, K]`` matrix, is the same
for every row), exactly

    (w_n - ŵ_n)^T H (w_n - ŵ_n)

GPTQ minimizes this via a single greedy left-to-right sweep: quantize
column ``i`` to its nearest grid point, then propagate the resulting error
into every *not-yet-processed* column via ``H``'s own (Cholesky-factored)
inverse -- an exact one-shot correction under the assumption that later
columns get to fully absorb it, but a *single* pass, never revisiting a
column once it's been quantized.

QuantEase instead solves the same per-row objective by plain **cyclic
coordinate descent**: repeatedly sweep over every column ``k``, and at each
step, hold every other column of ``ŵ_n`` fixed and move ``ŵ_n[k]`` to
whichever grid point minimizes the *objective itself* (not a linearized
correction) -- a classical, textbook technique for a quadratic objective
with no matrix inversion at all, needing only ``H`` itself (never
``H^{-1}``). Per column ``k``, holding a row's current quantized weight
``ŵ_n`` fixed everywhere else, the unconstrained optimum of the same
quadratic form is a single closed-form step:

    r = w_n - ŵ_n                       # current residual, this row
    delta* = (H[k, :] @ r) / H[k, k]     # unconstrained best move for ŵ_n[k]
    ŵ_n[k] <- round_to_grid(ŵ_n[k] + delta*)

repeated for every column, for several full sweeps ("epochs") over all
columns -- unlike GPTQ's single pass, a column already processed can (and
typically does) get revisited and revised again once later columns have
had a chance to move, since each sweep only ever *decreases* (never
increases) the shared quadratic objective. This module runs a small,
fixed number of cyclic sweeps (not to full convergence) as its own
practical compute/accuracy trade-off, matching the paper's own reported
practice of a handful of epochs being enough to beat plain round-to-nearest
and match or beat GPTQ; it does not reproduce the paper's own additional
outlier-aware extension (a small number of weights kept at higher
precision), which is a distinct, separately-scoped idea.

Every column update above is fully vectorized across all ``N`` output
rows at once (``H`` doesn't depend on the row), so a full sweep costs
``O(N * K^2)`` -- the same per-sweep cost order GPTQ's own Cholesky
correction pays for its single sweep, just repeated a handful of times
here instead of once. Plain numpy, no framework dependency, matching
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`'s own
"post-hoc adjustment from real activations" style.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_quantease(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_epochs: int = 4,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes QuantEase-style cyclic coordinate-descent rounding for
    every ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present
    (by node output name) in both ``float_model`` and ``quantized_model``,
    using real activations captured from ``float_model``. See this module's
    own docstring for the technique.

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
            layer's Hessian from. Each batch is a
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
    :param num_epochs: number of full cyclic sweeps over every column --
            each sweep only ever decreases the shared reconstruction
            objective, so more epochs never hurt accuracy, only cost
            (``O(N * K^2)`` per sweep, per layer)
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its QuantEase-optimized codes (same
            shape, dtype, and scale -- only which integer each element
            rounds to changes)

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_quantease_cpp`) -- full parameter parity, no
    functionality gap and no accepted numerical-algorithm divergence (see
    that function's own docstring, and ``onnxsim/quantease_entry.h``: this
    port needs no matrix inversion/factorization at all, unlike
    :func:`onnxsim.apply_gptq_cpp`'s own Cholesky-inverse note, so there is
    no linear-algebra routine whose choice of algorithm could diverge from
    numpy's own).
    """
    from onnxsim.onnx_simplifier import apply_quantease_cpp

    return apply_quantease_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_epochs=num_epochs,
        providers=providers,
    )
