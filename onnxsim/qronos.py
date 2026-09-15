"""Qronos (2025, ICLR 2026, "Qronos: Correcting the Past by Shaping the
Future in Post-Training Quantization", https://arxiv.org/abs/2505.11695) --
a sequential, whole-model generalization of :mod:`onnxsim.gptq`.

**How this differs from** :mod:`onnxsim.gptq`: GPTQ's own error-compensation
scope is narrow in a specific way. Within *one* layer, after quantizing a
column, it charges that column's own leftover rounding error forward onto
the *still-unquantized* columns of the *same* weight matrix (via the
Hessian-inverse-Cholesky OBC/OBS mechanism :mod:`onnxsim.gptq` already
implements) -- but it implicitly assumes the *activations* feeding that
layer are exact. GPTQ's own Hessian ``H = X^T X`` is always computed from
``float_model``'s real activations, even when correcting a layer deep inside
an otherwise-already-quantized network -- it never accounts for the fact
that, at actual deployment, that layer's real input already carries
whatever error every upstream layer's own quantization introduced. Qronos's
distinguishing contribution is an explicit, more complete correction that
accounts for **both** (1) this layer's own weight-rounding error (same as
GPTQ) **and** (2) the error already baked into this layer's activations
because upstream layers were quantized first -- a genuine cross-layer error
term GPTQ's own per-layer Hessian doesn't model.

The mechanism: consider a single layer with float weight ``W`` ([N, K],
output channel first). Let ``X_float`` be the calibration activations this
layer would see if every upstream layer were still exact (what
:mod:`onnxsim.gptq` uses), and ``X_quant`` the activations this layer
*actually* sees once upstream layers have already been quantized (computed
by running the calibration data through the model with every upstream layer
already Qronos-corrected). GPTQ minimizes ``||(W - Wq) @ X_float||^2``
-- matching the *quantized* layer's output to the *float* layer's output,
both computed on the *same*, exact input, so the only error being modeled is
this layer's own rounding. Qronos instead minimizes what actually matters at
deployment: ``||W @ X_float - Wq @ X_quant||^2`` -- the float network's ideal
output (clean input, clean weight) versus the deployed network's actual
output (corrupted input, quantized weight).

Let ``dX = X_quant - X_float`` -- the upstream error itself, a genuinely
*small*, bounded quantity (it is nothing but earlier layers' own INT4
rounding noise), as opposed to ``X_float``/``X_quant`` themselves, which can
be arbitrarily large or ill-conditioned. Writing ``H = X_quant^T @ X_quant``
(the Hessian of the *real* input this layer receives) and reusing
:func:`onnxsim.gptq._inverse_hessian_cholesky` for its damped-Cholesky-of-
``H^{-1}`` reformulation, define

    ``W_opt = W - W @ dX @ X_quant^T @ H^{-1}``

-- ``W``'s own value, shifted by a Hessian-weighted correction driven only
by ``dX``. By construction ``W_opt @ X_quant`` recovers ``W @ X_float``
whenever ``H^{-1}`` exactly inverts ``H`` (substitute ``X_quant = X_float +
dX`` and expand), so ``W_opt`` is a least-squares solution for matching
``W``'s own ideal (clean-input) target using the *real*, corrupted input.
Basic OLS orthogonality then means minimizing the original objective over
quantized ``Wq`` is *exactly* equivalent to minimizing
``||(W_opt - Wq) @ X_quant||^2`` -- precisely GPTQ's own per-column greedy
objective, unchanged, with ``W_opt`` standing in for the float weight and
``H`` (from ``X_quant``, not ``X_float``) standing in for GPTQ's Hessian.
That is this module's whole implementation: compute ``W_opt`` and ``H`` as
above, then hand them unchanged to
:func:`onnxsim.gptq._gptq_quantize_columns` -- the same Cholesky-based
least-squares machinery GPTQ itself uses, reused rather than reimplemented.

Deriving ``W_opt`` as a shift proportional to ``dX`` (rather than
reconstructing it from scratch via ``Y @ X_quant @ H^{-1}`` for some
separately-computed ideal target ``Y`` -- algebraically identical when
``H^{-1}`` is exact, but numerically very different) matters in practice:
GPTQ's own damping deliberately makes ``H^{-1}`` an *inexact* inverse of
``H`` along near-singular directions (the correlated-calibration-channel
scenario :mod:`onnxsim.gptq`'s own docstring motivates), and reconstructing
``W_opt`` from scratch amplifies that inexactness by ``W``'s own
(unrelated, potentially large) magnitude. Shifting *from* ``W`` by an amount
proportional to ``dX`` instead keeps that same inexactness scaled by ``dX``
-- small by construction -- and makes the reduction to plain GPTQ exact
(bit-for-bit, not just approximately) whenever ``dX`` is exactly zero:
``W_opt == W`` and :func:`_gptq_quantize_columns` is called with the
identical arguments :func:`onnxsim.gptq.apply_gptq` itself would use. This
is always the case for a layer with no already-quantized upstream layer
feeding it (e.g. the very first layer) -- Qronos is a strict generalization
of GPTQ, not a different algorithm bolted on.

The paper frames this per-column update as alternating between an "error
correction" step (undoing the input's own already-baked-in error, this
module's ``W_opt`` shift) and a "diffusion" step (spreading this column's
own new rounding error onto future columns, GPTQ's own mechanism) -- both
happen here, just factored as one least-squares retarget followed by one
unmodified GPTQ pass, rather than interleaved column-by-column; the two
formulations are algebraically equivalent for this module's own (fixed,
already-computed) scale grid.

**Scope**: correcting for cross-layer error requires processing layers in
real forward-execution order, quantizing each one and using its own
already-quantized output as the next layer's real ("corrupted") calibration
activation -- unlike :mod:`onnxsim.gptq`, which computes every layer's
Hessian from the untouched float model independently and in any order.
:func:`apply_qronos` does exactly this for every
``quantize_weight_only_int4``-quantized MatMul/Gemm layer present, ordering
candidates by their node's position in ``float_model.graph.node`` (the ONNX
IR spec requires a graph's nodes be topologically sorted, so this order is a
valid, and for a plain feedforward stack the *only*, forward-execution
order) and re-probing the partially-Qronos-corrected model before each
subsequent layer. A layer with more than one immediate quantized predecessor
(e.g. a residual join) still gets a *some* real upstream-corrected input --
just not disentangled per branch -- since probing reads whatever single
tensor actually reaches that layer's input, regardless of how many quantized
paths fed into it upstream.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx
import onnx.numpy_helper

from onnxsim.calibration import Tensors


def apply_qronos(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Qronos-corrects every ``quantize_weight_only_int4``-quantized
    MatMul/Gemm layer present (by node output name) in both ``float_model``
    and ``quantized_model``, processing layers in forward-execution order so
    each one's correction accounts for every upstream layer's own
    already-applied quantization error, not just its own rounding. See this
    module's own docstring for the technique and how it differs from
    :func:`onnxsim.gptq.apply_gptq` (which this reduces to when a layer has
    no already-quantized upstream layer feeding it).

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
            layer's target/Hessian from -- see
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor -- see
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same name
    :param proc_block_size: GPTQ's own column-processing block size -- see
            :func:`onnxsim.gptq._gptq_quantize_columns`
    :param providers: onnxruntime execution providers to run the model on
            when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its Qronos-corrected codes (same shape,
            dtype, and scale -- only which integer each element rounds to
            changes)

    The pure-Python machinery above (``_find_int4_matmul_candidates``,
    ``_gptq_quantize_columns``/``_inverse_hessian_cholesky`` reused from
    :mod:`onnxsim.gptq`) stays available for other modules to import, but
    this entry point is now a thin alias for the verified C++ port
    :func:`onnxsim.apply_qronos_cpp` (``onnxsim/qronos_entry.cpp``'s own
    ``ApplyQronos``), forwarding every argument unchanged. Exact
    (bit-for-bit) agreement was verified against this function's own
    pre-alias implementation across single- and multi-layer (real
    cross-layer correction) models, MatMul/transB-Gemm, block sizes,
    damping levels, and dead/duplicate calibration channels -- see
    tests/test_qronos_cpp.py -- before this alias was made. (The port's
    dense inverse/Cholesky use scalar double-precision kernels rather
    than LAPACK, the same accepted numerical scope
    :func:`onnxsim.apply_gptq_cpp` already documents; no divergence was
    observed anywhere measured.) Imported lazily (inside the function
    body, not at module scope) to avoid a circular import:
    ``onnxsim.onnx_simplifier`` already imports from this module, so
    importing it back at module load time here would deadlock the import
    machinery.
    """
    from onnxsim.onnx_simplifier import apply_qronos_cpp

    return apply_qronos_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        percdamp=percdamp,
        proc_block_size=proc_block_size,
        providers=providers,
    )
