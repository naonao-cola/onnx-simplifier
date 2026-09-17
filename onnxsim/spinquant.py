"""SpinQuant (Liu et al., 2024, "SpinQuant: LLM Quantization with Learned
Rotations", https://arxiv.org/abs/2405.16406). onnxsim ports the algorithm,
not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.quip_sharp` (SpinQuant's
own reference implementation learns rotation matrices end-to-end against
live PyTorch weights via a Cayley-manifold optimizer, with no ONNX export
path).

SpinQuant's core idea, shared with QuIP#/QuIP (see
:mod:`onnxsim.quip_sharp`): conjugating a weight by an orthogonal rotation
before quantizing it can make the rotated weight far more
quantization-friendly than the original -- fewer/less extreme outlier
directions for a uniform grid to waste precision on. QuIP#'s own choice is
a *random* rotation (a concentration-of-measure argument: any fixed
vector, rotated by a uniformly random orthogonal matrix, spreads out
evenly with high probability, regardless of the original weight's own
structure). SpinQuant's own contribution is that the rotation doesn't have
to be random -- a rotation *fit to the data* can do noticeably better,
since it can specifically target the directions the real weight/activation
distribution actually concentrates its outliers in, rather than relying on
a probabilistic argument that ignores that structure entirely.

SpinQuant's own reference implementation fits (typically four, one per
attention/MLP sub-block) rotation matrices per layer via gradient descent
on the quantized model's own end-to-end loss, constrained to the
orthogonal group via a Cayley-manifold optimizer -- calibration-aware,
differentiable-quantization machinery that is not independently verifiable
the way a closed-form procedure is (the same reason
:mod:`onnxsim.quip_sharp` doesn't reproduce QuIP#'s own randomized-Hadamard
construction verbatim, and :mod:`onnxsim.low_rank_compensation` uses
truncated SVD rather than a learned low-rank factorization). This module
instead reproduces SpinQuant's own "R1-only" ablation -- the paper's own
simplified configuration, reported to capture most of the improvement over
no rotation at all -- via a classical, closed-form substitute: fit a
*single* input-side rotation per layer as the eigenvector basis of that
layer's own calibration-activation covariance matrix (an ordinary,
symmetric eigendecomposition), then block-wise RTN-quantize the rotated
weight to INT4 exactly like :func:`onnxsim.quantize_weight_only_int4`
does. Reconstruction is exact before quantization, since rotating by an
orthogonal matrix is lossless:

    Before:
      Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

    After:
      U: initializer, float32 [K, K]            -- the fitted rotation
      X_rotated = MatMul(X, U)
      Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
                                                  -- INT4 codes, [K, N]
      Y = MatMul(X_rotated, Wtilde_hat) [+ bias]

Why the eigenvector basis specifically, rather than SpinQuant's own learned
rotation: it is the classical, closed-form answer to "which rotation makes
this data's second-moment structure as close to isotropic as possible" --
the eigenvectors of the covariance are exactly the directions ordinary PCA
identifies as carrying disproportionate variance, and rotating into that
basis spreads out what was concentrated along a few of them, the same
effect a learned rotation is chasing, achieved with ordinary linear algebra
(``numpy.linalg.eigh``) instead of an unverifiable optimization loop.
Unlike QuIP#'s random rotation, this needs calibration data (the whole
point is to target the real distribution's own structure) but, also unlike
QuIP#, needs no second, output-side rotation or non-uniform lattice
codebook: SpinQuant's contribution is specifically about the rotation
being *learned* (here: fit in closed form) rather than random, not about a
different quantization backend, so this module pairs it with the same
block-wise RTN backend :mod:`onnxsim.quantize_weight_only_int4` already
uses.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_spinquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies SpinQuant-style learned-rotation preprocessing (its own
    "R1-only" configuration, fit in closed form -- see this module's own
    docstring) plus block-wise INT4 quantization to every MatMul/vanilla-
    Gemm layer with a constant 2-D float32 weight whose reduction dimension
    ``K`` is divisible by ``block_size``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to fit each layer's own rotation -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative rotation fit than random
            input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            ``(X @ U) @ Ŵtilde`` (plus the original bias, if any), where
            ``U`` is the fitted rotation and ``Ŵtilde`` is reconstructed
            in-graph from packed INT4 codes and a per-block float32 scale;
            output tensor name unchanged. Layers with a non-constant,
            non-2-D weight, a reduction dimension not divisible by
            ``block_size``, or no calibration activation available, are
            left untouched; a model with no matching layer, or an opset
            older than 21 (INT4's tensor type and ``DequantizeLinear``'s
            ``block_size`` attribute both need opset 21), is returned
            unchanged

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_spinquant_cpp`) -- full parameter parity, no
    functionality gap (see that function's own docstring, and
    ``onnxsim/spinquant_entry.h``, for the one documented, immaterial-to-
    -correctness numerical divergence: the fitted rotation's own
    eigendecomposition uses a hand-rolled Jacobi algorithm rather than
    LAPACK's, so individual columns need not match sign-for-sign or
    order-for-order, only that the rotation stays orthogonal).
    """
    from onnxsim.onnx_simplifier import apply_spinquant_cpp

    return apply_spinquant_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        block_size=block_size,
        providers=providers,
    )
