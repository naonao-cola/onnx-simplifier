"""GPTVQ (Van Baalen et al., 2024, "GPTVQ: The Blessing of Dimensionality
for LLM Quantization", https://arxiv.org/abs/2402.15319). onnxsim ports the
*algorithm*, not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.aqlm` (GPTVQ's own
reference implementation quantizes live PyTorch weights via a
calibration-aware optimizer, with no ONNX export path).

GPTVQ's own idea is a genuine combination of two techniques already in
onnxsim, each covering only one half of it:

- :mod:`onnxsim.gptq` quantizes a layer's input channels **one at a time**,
  propagating each channel's rounding error into every not-yet-quantized
  channel via the layer's own (calibration-data-derived) Hessian -- but its
  codebook is always a fixed, uniform integer grid (a *scalar* scheme):
  every element is corrected independently, never jointly.
- :mod:`onnxsim.aqlm`/:mod:`onnxsim.kmeans_quantization` quantize (groups
  of) weight elements against a codebook **fit to the weight's own
  values** -- a *vector* scheme, richer than a uniform grid -- but neither
  uses any calibration data or Hessian: every group/element is assigned to
  its nearest codebook entry independently, with no error compensation
  between groups at all.

GPTVQ's contribution is doing both at once: quantize small groups of
``vector_dim`` consecutive input-channel columns against a codebook fit to
the *whole layer's* weight values (like :mod:`onnxsim.aqlm`'s single shared
codebook, via the same k-means routine), processing groups left to right
and, after each group is assigned to its nearest codebook entry,
propagating the resulting per-column residual into every not-yet-quantized
column using :mod:`onnxsim.gptq`'s own Cholesky-based Hessian correction
(:func:`onnxsim.gptq._inverse_hessian_cholesky`,
reused unchanged) -- exactly the same per-column identity GPTQ itself
uses, which cancels a column's rounding error's contribution to the
*layer's* squared reconstruction error regardless of how that column's
residual was produced. That identity is what lets a *joint*,
whole-group codebook decision (impossible to express as an independent
per-column rounding rule) still slot into GPTQ's sequential error-feedback
loop unchanged: each of the group's columns gets its own compensation step
using its own actual residual, one column at a time, exactly like GPTQ's
existing loop -- only the *value* each column is corrected towards (a
coordinate of a jointly chosen codebook vector, not an independently
rounded grid point) is new. This is a simpler stand-in for the paper's own
more involved block-Cholesky joint update, chosen for the same
reason :mod:`onnxsim.aqlm`'s greedy residual k-means stands in for AQLM's
own beam-search codebook optimizer: an independently verifiable procedure
built from already-proven pieces, rather than an unverified from-scratch
reproduction of a paper's bespoke joint solver.

Like :mod:`onnxsim.kmeans_quantization`'s single ``Gather``, dequantization
here needs no per-group scale multiply (the codebook already stores
reconstructed weight vectors in the weight's own units): a single
``Gather(codebook, codes, axis=0)`` followed by a ``Reshape`` (folding the
gathered ``[num_groups, vector_dim]`` vectors back into the weight's own
``[N, K]``/``[K, N]`` layout). No custom op or contrib domain, and no
opset requirement beyond ordinary ``Gather``/``Reshape``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.calibration import Tensors
from onnxsim.gptq import _inverse_hessian_cholesky


def _gptvq_quantize_groups(
    w_nk: np.ndarray,
    codebook: np.ndarray,
    h: np.ndarray,
    percdamp: float,
    vector_dim: int,
) -> np.ndarray:
    """Returns, for every row of ``w_nk`` ([N, K], output channel first)
    and every consecutive ``vector_dim``-wide column group, the index into
    ``codebook`` ([num_centroids, vector_dim]) that row's group is
    quantized to. Groups are processed left to right: each group's
    ``[N, vector_dim]`` slice (as already corrected by every
    previously-processed group) is assigned, per row, to its nearest fixed
    codebook entry, then -- one column at a time, in order, exactly
    :mod:`onnxsim.gptq`'s own loop -- that column's resulting residual is
    propagated into every not-yet-quantized column via
    ``h``'s Cholesky-based inverse (see
    :func:`onnxsim.gptq._inverse_hessian_cholesky`). Reusing GPTQ's
    per-column formula here is valid because its derivation only depends on
    the column's own residual (whatever produced it), not on how that
    column's quantized value was chosen -- so a value coming from a joint,
    whole-group codebook lookup propagates through it exactly like GPTQ's
    own independently-rounded scalar would.
    """
    n, k = w_nk.shape
    num_groups = k // vector_dim
    hinv = _inverse_hessian_cholesky(h, percdamp)

    codes = np.zeros((n, num_groups), dtype=np.int64)
    w_work = w_nk.copy()

    for g in range(num_groups):
        col_start = g * vector_dim
        col_end = col_start + vector_dim
        w_group = w_work[:, col_start:col_end]  # [n, vector_dim]
        dist = np.sum(
            (w_group[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2
        )
        assignment = np.argmin(dist, axis=1)  # [n]
        codes[:, g] = assignment
        q_group = codebook[assignment]  # [n, vector_dim]

        for j in range(vector_dim):
            col = col_start + j
            d = hinv[col, col]
            err = (w_work[:, col] - q_group[:, j]) / d
            w_work[:, col] = q_group[:, j]
            if col + 1 < k:
                w_work[:, col + 1 :] -= np.outer(err, hinv[col, col + 1 :])

    return codes


def quantize_weight_only_gptvq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    vector_dim: int = 2,
    num_centroids: int = 256,
    num_iterations: int = 10,
    percdamp: float = 0.01,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    skip_names: Optional["set[str]"] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``vector_dim``) into GPTVQ-style Hessian-compensated vector-codebook
    quantization -- see this module's own docstring for the technique.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a much more representative Hessian than random input).
    :param vector_dim: elements per group (each ``vector_dim``-element
            chunk of consecutive input-channel columns is jointly
            quantized to a single codebook entry)
    :param num_centroids: entries in the layer's own fitted codebook (256
            is a typical choice, matching one byte per stored index)
    :param num_iterations: Lloyd's-algorithm iterations fitting the
            codebook (see :func:`onnxsim.aqlm._fit_kmeans_codebook`)
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :mod:`onnxsim.gptq`'s own default -- keeps the inversion
            numerically stable when calibration data doesn't fully
            activate (or correlates too tightly across) a layer's input
            channels
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for the codebook's
            k-means initialization (a fresh ``numpy.random.Generator`` is
            derived once and advanced per matched layer, in graph node
            order, so results are deterministic and reproducible for a
            given model and seed)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Gather(Codebook, Codes, axis=0)`` reshaped back to the
            weight's own shape, feeding the original MatMul/Gemm node;
            layers with a non-constant, non-2-D, or non-``vector_dim``-
            divisible weight are left untouched

    The pure-Python k-means-plus-Hessian-correction implementation above
    has been retired in favor of the verified C++ port -- this is now a
    thin alias for :func:`onnxsim.apply_gptvq_cpp`
    (``onnxsim/gptvq_entry.cpp``'s own ``ApplyGptvq``), forwarding every
    argument unchanged. **Behavior change from earlier onnxsim
    versions:** the C++ port fits its k-means codebook via a partial
    Fisher-Yates sample over ``std::mt19937_64`` with an independent
    per-node RNG derivation, not this module's own
    ``numpy.random.Generator.choice`` sequenced across matches -- the
    same permanent divergence :func:`onnxsim.apply_quarot`/
    :func:`onnxsim.apply_quarot_cpp` already document for their own
    random rotation. A given ``seed`` therefore no longer derives the
    same codebook this function produced before being aliased; the
    Hessian-compensated group-correction math itself is unchanged (see
    tests/test_gptvq_cpp.py's own cross-check). ``_gptvq_quantize_groups``
    stays in this module (imported by tests/test_gptvq_cpp.py); only the
    entry point is aliased. Imported lazily (inside the function body,
    not at module scope) to avoid a circular import:
    ``onnxsim.onnx_simplifier`` already imports from this module, so
    importing it back at module load time here would deadlock the import
    machinery.
    """
    from onnxsim.onnx_simplifier import apply_gptvq_cpp

    return apply_gptvq_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        vector_dim=vector_dim,
        num_centroids=num_centroids,
        num_iterations=num_iterations,
        percdamp=percdamp,
        providers=providers,
        skip_names=skip_names,
    )
