"""ParoQuant (Liang et al., 2025, "ParoQuant: Pairwise Rotation Quantization
for Efficient Reasoning LLM Inference", https://arxiv.org/abs/2511.10645,
ICLR 2026). onnxsim ports the algorithm, not any framework's code, per the
same rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.spinquant`
(ParoQuant's own reference implementation optimizes rotation angles against
live PyTorch weights with no ONNX export path).

**How this differs from** :mod:`onnxsim.spinquant`: both modules conjugate a
weight by an orthogonal rotation before block-wise INT4 quantization, to
make the rotated weight less outlier-concentrated and therefore cheaper to
quantize -- the same rotate-then-quantize contract :mod:`onnxsim.quip_sharp`
and :mod:`onnxsim.duquant` also share. :mod:`onnxsim.spinquant` fits one
**dense** ``[K, K]`` rotation matrix per layer (SpinQuant's own "R1-only"
substitute: the eigenvector basis of the calibration-activation covariance),
which at inference time costs a full ``K x K`` matmul against the
activation. ParoQuant's own distinguishing contribution is replacing that
dense rotation with a set of many independent, cheap **2x2 (pairwise,
Givens) rotations** instead: each one mixes exactly *one pair* of channels
and leaves every other channel untouched, so the "rotation" as a whole is
a ``[K, K]`` matrix that is block-diagonal (2x2 blocks on the chosen pairs,
identity everywhere else) rather than dense. The paper reports this keeps
the rotation's own compute overhead under 10% of the layer's matmul cost
(a block-diagonal matmul only ever touches 2 columns per output column,
versus every dense rotation column touching all ``K``), while still
meaningfully narrowing the per-group dynamic range a dense rotation
targets -- narrower than doing nothing, even if not quite as narrow as a
fully dense rotation's.

A Givens rotation is the 2x2 orthogonal matrix
``[[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`` applied to
exactly two coordinates of a vector; applying one to a pair of channels
``(i, j)`` of an activation ``X`` and the *same* rotation to the *same*
pair of rows of the weight is an exact algebraic identity
(``(X @ R) @ (R.T @ W) == X @ W`` for any orthogonal ``R``, here block-
diagonal with 2x2 Givens blocks on the chosen pairs and identity
elsewhere -- a block-diagonal matrix of orthogonal blocks is itself
orthogonal), so applying a whole set of independent pairwise rotations is
still lossless before quantization, the same "provably exact migration,
then quantize" contract :mod:`onnxsim.spinquant`/:mod:`onnxsim.smoothquant`
already use.

This module's pairing is the cheapest, most hardware-friendly one the paper
describes: fixed adjacent channels within each quantization block --
``(0, 1), (2, 3), (4, 5), ...`` -- rather than a data-chosen pairing (no
extra bookkeeping to describe or transmit a permutation, unlike
:mod:`onnxsim.duquant`'s own calibration-driven channel reassignment). Each
pair's own rotation angle is then fit to that layer's own weight via a
small grid search over candidate angles in ``[-pi/4, pi/4]`` (covering
every distinct 2x2 rotation, since a Givens rotation is pi-periodic and
symmetric about pi/4), minimizing that pair's own contribution to its
quantization block's INT4 round-to-nearest reconstruction error -- the same
"grid-search a scalar against measured reconstruction error, not a
closed-form guarantee" style :mod:`onnxsim.awq` already uses for its own
per-channel scale, applied here per-pair instead of once per layer. Pairs
within the same block are optimized in a fixed left-to-right order, each
one seeing the already-rotated state of every earlier pair in its block
(their combined effect changes the block's own outlier structure, so
sequencing the search lets later pairs adapt to it), which the paper's own
gradient-free per-pair angle search also does.

Finally, this module combines the pairwise rotation with a SmoothQuant-style
(:mod:`onnxsim.smoothquant`) per-channel scale migration -- ParoQuant's own
second ingredient, evening out channel magnitudes before the rotation search
runs, using the exact same closed-form, alpha-parameterized formula
(``s_j = max(|X_j|) ** alpha / max(|W_j|) ** (1 - alpha)``) and the same
"one fixed global alpha, not searched" practice :mod:`onnxsim.smoothquant`
already documents. Combining a diagonal scale with an orthogonal rotation
stays exact for the same reason either piece alone is: ``X @ W ==
((X / s) @ R) @ (R.T @ (diag(s) @ W))`` for any invertible diagonal
``diag(s)`` and orthogonal ``R``, regardless of composition order.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_paroquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    alpha: float = 0.5,
    num_angle_steps: int = 9,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies ParoQuant-style channel-wise SmoothQuant scaling plus
    pairwise (Givens) rotation preprocessing (see this module's own
    docstring) followed by block-wise INT4 quantization to every
    MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
    reduction dimension ``K`` is divisible by ``block_size``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to measure each layer's own input channel
            activation range for the SmoothQuant-style scale -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative scale than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``, and per
            pairwise-rotation grouping (must be even), matching
            :func:`onnxsim.quantize_weight_only_int4`'s own default
    :param alpha: the channel-scale migration strength, same meaning and
            default as :func:`onnxsim.apply_smoothquant`'s own ``alpha``
    :param num_angle_steps: grid points per pair for the Givens angle
            search, evenly spaced over ``[-pi/4, pi/4]`` inclusive; higher
            values search more finely at proportionally more cost (one
            block re-quantization and reconstruction-error measurement per
            candidate, per pair)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the SmoothQuant-style scale,
            avoiding a divide-by-zero on an all-zero channel
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight and activation
            replaced by scaled-and-pairwise-rotated, INT4-quantized
            versions (plus the original bias, if any); output tensor name
            unchanged. Layers with a non-constant, non-2-D weight, a
            reduction dimension not divisible by ``block_size``, or no
            calibration activation available, are left untouched; a model
            with no matching layer, an odd ``block_size``, or an opset
            older than 21 (INT4's tensor type and ``DequantizeLinear``'s
            ``block_size`` attribute both need opset 21), is returned
            unchanged

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_paroquant_cpp`) -- full parameter parity, no
    functionality gap (see that function's own docstring, and
    ``onnxsim/paroquant_entry.h``; unlike :func:`onnxsim.apply_spinquant`'s
    own eigendecomposition-algorithm divergence, the per-pair Givens angle
    search here has no RNG/LAPACK-equivalent choice to diverge on, so
    fitted angles are expected to agree closely with this function's own
    former pure-Python implementation).
    """
    from onnxsim.onnx_simplifier import apply_paroquant_cpp

    return apply_paroquant_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        block_size=block_size,
        alpha=alpha,
        num_angle_steps=num_angle_steps,
        epsilon=epsilon,
        providers=providers,
    )
