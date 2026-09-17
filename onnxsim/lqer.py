"""LQER (Zhang et al., 2024, "LQER: Low-Rank Quantization Error
Reconstruction for LLMs", https://arxiv.org/abs/2402.02446). onnxsim ports
the algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.low_rank_compensation`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`.

:mod:`onnxsim.low_rank_compensation` (ZeroQuant-V2's LoRC) already adds a
low-rank correction to a quantized layer's leftover error matrix
``residual = float_weight - dequantized_weight``, computed via that
matrix's own plain SVD -- no calibration data needed, but also blind to
which parts of ``residual`` actually matter: the Eckart-Young theorem's
"best rank-r approximation" is only best in an *unweighted*
Frobenius-norm sense, treating every input channel's contribution to
``residual`` as equally important. LQER's own contribution is exactly
this: since what actually reaches the model's output is ``X @ residual``
(not ``residual`` on its own), an input channel the real calibration data
drives with much larger activation energy than another deserves much more
weight in the low-rank fit, not equal weight.

**The technique**: for a candidate layer's ``residual`` (``[K, N]``,
ready for ``X @ residual``) and its own calibration-measured per-input-
channel activation RMS ``s`` (``[K]``, ``s_k = sqrt(mean_over_calibration(
X[:, k]^2))``), LQER row-scales the residual by ``s`` before taking the
SVD (``R' = diag(s) @ residual``), keeps the top ``r`` singular
components of ``R'``, and *un*-scales the result back
(``residual_approx = diag(1/s) @ R'_r``) before folding it into the same
``B @ A`` two-matmul correction :mod:`onnxsim.low_rank_compensation`
already uses. This is a weighted low-rank approximation minimizing
``sum_k s_k^2 * ||residual[k, :] - residual_approx[k, :]||^2`` --
exactly the AWQ-style "some channels matter more, weight by their own
activation energy" idea (:mod:`onnxsim.awq`'s own per-channel scaling),
applied here to *error compensation* rather than to the quantization
grid itself. Row-channels the calibration data never actually drives (an
all-zero or near-zero ``s_k``) contribute almost nothing to the weighted
objective, so LQER's fit spends its limited rank budget on the channels
that actually move the layer's real output, which
:mod:`onnxsim.low_rank_compensation`'s plain SVD cannot distinguish from
any other channel.

Everything else -- candidate matching (reusing
:mod:`onnxsim.adaround`'s ``_find_int4_matmul_candidates``/
``_node_outputs``), the resulting graph rewrite (two extra ``MatMul``
nodes plus an ``Add``, the original INT4 weight/scale left untouched) --
is identical to :mod:`onnxsim.low_rank_compensation`; only how ``B``/``A``
are computed differs. Falling back to an unweighted SVD (LoRC's own
formula) for any candidate whose input activation was never observed
during calibration keeps this a strict superset, never a regression,
of :mod:`onnxsim.low_rank_compensation`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_lqer_cpp


def weighted_low_rank_correction(
    residual_kn: np.ndarray,
    channel_weights: Optional[np.ndarray],
    rank: int,
    eps: float = 1e-6,
) -> "tuple[np.ndarray, np.ndarray]":
    """The core LQER math, standalone and directly testable: a rank-``r``
    approximation of ``residual_kn`` (``[K, N]``) minimizing
    ``sum_k channel_weights[k]^2 * ||residual[k, :] - approx[k, :]||^2``
    instead of plain (unweighted) Frobenius error -- see this module's own
    docstring for the derivation (row-scale by ``channel_weights`` before
    the SVD, un-scale the result back afterward).

    :param residual_kn: ``[K, N]`` matrix to approximate
    :param channel_weights: ``[K]`` non-negative per-row weight (typically
            a per-input-channel activation RMS); ``None`` falls back to
            :mod:`onnxsim.low_rank_compensation`'s own plain, unweighted
            SVD (every channel weighted equally)
    :param rank: target rank ``r`` (**not** clamped to ``min(K, N)`` here --
            callers needing that do it themselves, matching
            :mod:`onnxsim.low_rank_compensation`'s own contract)
    :param eps: floor applied to ``channel_weights`` before dividing by it
    :returns: ``(b_kn, a_rn)`` with ``b_kn @ a_rn`` the rank-``r``
            approximation, shapes ``[K, r]``/``[r, N]``, float32
    """
    if channel_weights is None:
        u, sv, vt = np.linalg.svd(residual_kn, full_matrices=False)
        b_kn = u[:, :rank] * sv[np.newaxis, :rank]
        a_rn = vt[:rank, :]
        return b_kn.astype(np.float32), a_rn.astype(np.float32)

    s = np.maximum(channel_weights, eps)
    weighted = residual_kn * s[:, np.newaxis]
    u, sv, vt = np.linalg.svd(weighted, full_matrices=False)
    b_kn = (u[:, :rank] * sv[np.newaxis, :rank]) / s[:, np.newaxis]
    a_rn = vt[:rank, :]
    return b_kn.astype(np.float32), a_rn.astype(np.float32)


def apply_lqer(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    rank: int = 8,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> onnx.ModelProto:
    """Adds an activation-weighted rank-``r`` low-rank correction to every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model`` --
    see this module's own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param rank: the correction's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer)
    :param calibration_data: representative input batches used to measure
            each layer's own per-input-channel activation RMS -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param eps: floor applied to each per-channel RMS before dividing by it,
            avoiding a division blow-up on a channel calibration barely
            drives
    :returns: ``quantized_model`` with every matched layer's output summed
            with a new rank-``r`` activation-weighted correction term (two
            chained ``MatMul`` nodes plus an ``Add``); the layer's own
            existing INT4 weight and scale are left completely untouched.
            A candidate whose activation was never observed during
            calibration (or isn't a plain 2-D ``[batch, K]`` shape) falls
            back to :mod:`onnxsim.low_rank_compensation`'s own unweighted
            SVD.

    Delegates to :func:`onnxsim.apply_lqer_cpp` (the verified C++ port,
    which has full parameter parity with this function -- see
    ``lqer_entry.h`` for its own scope note); this pure-Python name is
    kept only for backward compatibility with existing callers.
    """
    return apply_lqer_cpp(
        float_model,
        quantized_model,
        rank=rank,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        providers=providers,
        eps=eps,
    )
