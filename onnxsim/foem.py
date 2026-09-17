"""FOEM (2025, "First-Order Error Matters: Accurate Compensation for
Quantized Large Language Models", https://arxiv.org/abs/2507.11017).
onnxsim ports the *algorithm*, not any framework's code, per the same
rationale as :mod:`onnxsim.gptq`/:mod:`onnxsim.awq` (FOEM's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

Read :mod:`onnxsim.gptq` first -- this module extends it directly, in the
sense the paper itself frames its own contribution: as an add-on
correction to GPTQ's own compensation, not a replacement for it.

GPTQ's own OBS-style correction, applied at each column ``i``, treats the
quantization error at that column (``w_col - code_col * s``) as *the*
error to propagate forward, implicitly via a second-order (Hessian-only)
Taylor argument -- one that quietly assumes the ``w_col`` it just quantized
is still (to first order) the *original* column of ``W``, undisturbed by
anything except its own rounding. That assumption is false by construction
partway through the pass: every earlier column's own forward-propagation
step (``w1[:, i+1:] -= outer(err, hinv1[i, i+1:])``, see
:func:`onnxsim.gptq._gptq_quantize_columns`) already nudged column ``i``'s
own pre-quantization value away from ``W``'s own true original column --
a **first-order deviation that accumulates column by column** and, by the
time a column late in the pass gets processed, is no longer negligible.
FOEM's own fix: alongside the fresh rounding error GPTQ already
compensates, also charge forward a (damped) fraction of *that* accumulated
deviation -- how far the column's current pre-quantization value has
already drifted from ``W``'s own untouched original column -- so later
columns' own compensation accounts for both sources of error, not just the
newest one.

This module's own version of the correction (a good-faith, numerically
verified reproduction of the paper's own described mechanism -- "measuring
the raw difference between the current compensated weights and the
untouched full weights, scaled by a small factor" -- rather than a
transcription of the paper's own derivation, which this module does not
claim to reproduce exactly): at column ``i``, alongside GPTQ's own
``err = (w_col - code_col * s) / d``, this module also computes
``drift = (w_col_before_quantizing - w_orig_col) / d`` (``w_orig_col``
being ``W``'s own untouched original column, never modified by any
previous correction) and propagates ``err + foem_beta * drift`` forward
instead of ``err`` alone -- ``foem_beta`` the "small
factor" the paper describes damping the first-order term by. This
module's own default (``0.005``) is deliberately conservative: swept
empirically against several toy calibration scenarios (see
``tests/test_foem.py``), a larger ``foem_beta`` here consistently
*increases* reconstruction error rather than reducing it (the added term
overshoots -- easy to do, since it is this module's own good-faith
reconstruction of the paper's mechanism rather than a verified transcription
of its exact derivation), while a small ``foem_beta`` gives a modest,
repeatable improvement over plain GPTQ across every scenario tested. This
module does not claim the improvement is universal or matches the paper's
own much larger reported gains (measured across real LLM benchmarks, not
one toy MatMul) -- only that, empirically, a small nonzero ``foem_beta``
does not hurt and sometimes measurably helps, which is the honest, verified
extent of what this port demonstrates. Reuses
:func:`onnxsim.gptq._inverse_hessian_cholesky` for the same Cholesky-based
``H^{-1}`` reformulation GPTQ itself already provides -- no new Hessian
machinery needed, exactly the paper's own "reuses the Cholesky factors
GPTQ already stores" efficiency claim.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_foem(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    foem_beta: float = 0.005,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes FOEM-style sequential, Hessian-*and*-first-order-drift-
    compensated rounding for every ``quantize_weight_only_int4``-quantized
    MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``, using real activations
    captured from ``float_model``. See this module's own docstring for how
    this differs from :func:`onnxsim.gptq.apply_gptq` (which this module's
    own correction is an add-on to, not a replacement of).

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
            layer's Hessian from -- see :func:`onnxsim.gptq.apply_gptq`'s
            own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same
            name and default
    :param proc_block_size: GPTQ's own column-processing block size,
            matching :func:`onnxsim.gptq.apply_gptq`'s own parameter of the
            same name and default
    :param foem_beta: damping factor on the additional first-order drift
            term (see this module's own docstring) -- ``0.0`` recovers
            plain GPTQ exactly; kept small by default since this module's
            own empirical sweeps show a larger value overshoots and
            increases error rather than reducing it
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its FOEM-optimized codes (same shape,
            dtype, and scale -- only which integer each element rounds to
            changes)

    This entry point is a thin alias for the verified C++ port
    :func:`onnxsim.apply_foem_cpp` (``onnxsim/foem_entry.cpp``'s own
    ``ApplyFoem``), forwarding every argument unchanged. Exact (bit-for-bit)
    agreement was verified against this function's own pre-alias
    implementation (its former ``_foem_quantize_columns``, now removed as
    dead code -- no other module imported it) across MatMul/Gemm/transB-Gemm/
    biased-Gemm, block sizes, damping levels, multi-batch and rank-3
    calibration, dead/duplicate channels, every ``foem_beta`` (including the
    ``0.0``-recovers-plain-GPTQ degenerate case), and every skip shape --
    see tests/test_foem_cpp.py -- before this alias was made. (The port's
    dense inverse/Cholesky use scalar double-precision kernels rather than
    LAPACK, the same accepted numerical scope as
    :func:`onnxsim.apply_gptq`'s own C++ port; no divergence was observed
    anywhere measured.) Imported lazily (inside the function body, not at
    module scope) to avoid a circular import: ``onnxsim.onnx_simplifier``
    already imports from this module, so importing it back at module load
    time here would deadlock the import machinery.
    """
    from onnxsim.onnx_simplifier import apply_foem_cpp

    return apply_foem_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        percdamp=percdamp,
        proc_block_size=proc_block_size,
        foem_beta=foem_beta,
        providers=providers,
    )
