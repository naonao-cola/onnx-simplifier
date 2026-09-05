"""KBVQ-MoE (Xu et al., 2026, ICLR 2026, "KBVQ-MoE: KLT-guided SVD with
Bias-Corrected Vector Quantization for MoE Large Language Models",
https://arxiv.org/abs/2602.11184). onnxsim ports the algorithm -- the
paper's own reference code (https://github.com/xuzukang/kbvq_moe) trains
against live PyTorch MoE modules, with no ONNX export path, the same
rationale as :mod:`onnxsim.gptq`/:mod:`onnxsim.awq`.

A Mixture-of-Experts layer routes the *same kind* of token to many
structurally-similar experts, so a router group's own ``E`` expert weight
matrices end up highly cross-expert redundant: much of what any one
expert's weight encodes, some linear combination of the others already
encodes too. Every MoE-aware or vector-quantization scheme already in
onnxsim leaves that redundancy completely unexploited:

- :mod:`onnxsim.moequant`'s own contribution is a **calibration
  methodology** for MoE's sparse, data-dependent routing (Affinity-Guided
  Quantization + Expert-Balanced Self-Sampling feeding an ordinary
  per-expert :mod:`onnxsim.gptq` pass) -- it does not change *how* an
  expert's weight is represented relative to its siblings at all; each
  expert is still quantized as if it were an unrelated standalone layer.
- :mod:`onnxsim.kmeans_quantization`'s own codebook is fit *per layer*,
  independently -- run on each expert of a router group one at a time
  (exactly what this module's own baseline comparison in its test suite
  does), it re-fits, and re-pays the storage cost of, a fresh codebook
  for every expert's own copy of the group's shared structure.
- :mod:`onnxsim.low_rank_compensation`'s own SVD is of a single quantized
  layer's own error matrix (``float_weight - dequantized_weight``),
  fit and applied strictly *within* that one layer -- it never looks at,
  or shares a basis across, any other layer's weight at all.
- :mod:`onnxsim.bias_correction`'s own per-channel correction targets one
  ordinary Conv/Gemm/MatMul layer's own quantization-induced output mean
  shift. It has no notion of several already-lossy layers being summed
  together afterwards -- exactly what a token's top-``k`` selected
  experts' own outputs are, in every ``com.microsoft::MoE`` node this
  module (and :mod:`onnxsim.moequant`) targets -- so it cannot see, or
  correct for, quantization bias that MoE's own gated aggregation
  amplifies across several experts at once.

KBVQ-MoE's own contribution is a genuinely different **weight
representation** for a router group's experts, not a different
calibration recipe or a single-layer correction:

- **KLT-guided shared basis.** Flatten each of a router group's ``E``
  expert weight matrices (``fc1_experts_weights``/``fc2_experts_weights``,
  handled independently) into one length-``D`` vector, stack them into an
  ``[E, D]`` matrix, and take the Karhunen-Loeve Transform (KLT, i.e. PCA)
  of that stack: the top-``rank`` eigenvectors of the ``E`` experts' own
  ``D x D`` covariance matrix, computed here via one compact SVD of the
  *centered* ``[E, D]`` stack itself (:func:`_klt_basis`) -- numerically
  the same eigenvectors a direct eigendecomposition of the ``D x D``
  covariance would give (right singular vectors of centered data are that
  matrix's own eigenvectors), but never forming the far larger ``D x D``
  matrix explicitly, the same closed-form-SVD style
  :mod:`onnxsim.low_rank_compensation` already uses. This basis is fit
  **once per router group** and shared by construction -- every expert's
  own dominant, cross-expert-redundant component is represented by the
  same handful of basis vectors, not re-encoded from scratch per expert.
- **Per-expert residual vector quantization.** Each expert's own
  projection onto the shared basis (``shared_e = mean + coeff_e @
  basis``) is subtracted from its float weight, and only the
  *residual* -- whatever the shared basis does not already capture -- is
  vector-quantized with an ordinary k-means codebook, reusing
  :mod:`onnxsim.kmeans_quantization`'s own Lloyd's-algorithm fitting
  routine (:func:`onnxsim.kmeans_quantization._kmeans_1d`) directly,
  applied to each expert's own flattened residual instead of its raw
  weight.

Composing these two pieces is the whole point: the shared component
(the expensive-to-represent-per-expert part, since it is what makes the
experts *alike*) is paid for once per router group, while VQ -- already
the tightest-fitting codebook representation onnxsim has for a
single tensor's own value distribution -- is spent only on each expert's
own genuinely idiosyncratic remainder. See this module's own test suite
for a direct, numerical demonstration that this beats running
:mod:`onnxsim.kmeans_quantization` independently per expert at a matched
total bit budget, on a router group constructed to have real shared
structure.

Scope: like :mod:`onnxsim.moequant`, this targets ``com.microsoft::MoE``
nodes matched by :func:`onnxsim.pruning._match_moe_producer` (reused,
unmodified, via :func:`onnxsim.pruning._find_moe_chains`), restricted to
FLOAT32 ``fc1_experts_weights``/``fc2_experts_weights``. Also like
:mod:`onnxsim.moequant`, the result is **simulated ("fake") quantization**:
each expert's weight is overwritten in place with its dequantized
(shared-basis-reconstruction plus dequantized-residual) value, so the
graph keeps its original ``MoE`` node, dtype, and shapes -- no new
storage-packed op. The paper's own **channel-wise affine output bias
correction** -- compensating for how MoE's gated aggregation amplifies
several already-lossy experts' own individual quantization bias into a
larger aggregate bias at the ``MoE`` node's own output, a distinct
correction target from :mod:`onnxsim.bias_correction`'s own per-layer one
(see above) -- is deliberately left out of this module's own scope, the
same kind of documented simplification :mod:`onnxsim.moequant`'s own
docstring makes for a real ``QMoE`` rewrite: a natural, self-contained
follow-up built the same way :func:`onnxsim.bias_correction.correct_bias`
already measures a per-channel correction from calibration data, applied
at the ``MoE`` node's own output instead of a plain layer's.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.kmeans_quantization import _kmeans_1d
from onnxsim.pruning import _find_moe_chains


def _klt_basis(stack_ed: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """The KLT (Karhunen-Loeve Transform, i.e. PCA) basis shared across the
    ``E`` rows of ``stack_ed`` (``[E, D]``, ``D`` = one flattened expert
    weight's own element count): the top-``rank`` eigenvectors of the ``E``
    experts' own ``[D, D]`` covariance matrix. See this module's own
    docstring for why a compact SVD of the centered ``[E, D]`` stack gives
    exactly that without ever forming the ``[D, D]`` matrix.

    Returns ``(mean [D], basis [r, D])`` with ``r = min(rank, E, D)``
    (``rank`` clamped the same way :func:`onnxsim.low_rank_compensation.
    apply_low_rank_compensation`'s own ``rank`` is). ``r == 0`` (a
    non-positive ``rank``) returns an empty basis -- every expert's own
    "shared" reconstruction is then just the group mean.
    """
    mean = stack_ed.mean(axis=0)
    centered = stack_ed - mean
    r = max(0, min(rank, stack_ed.shape[0], stack_ed.shape[1]))
    if r == 0:
        return mean, np.zeros((0, stack_ed.shape[1]), dtype=stack_ed.dtype)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:r]


def _kbvq_reconstruct(
    w_e_d: np.ndarray, rank: int, bits: int, kmeans_iters: int, seed: int
) -> np.ndarray:
    """KBVQ-MoE's own shared-basis-plus-per-expert-residual-codebook
    reconstruction of one ``fc1``/``fc2`` expert-weight tensor, flattened
    to ``[E, D]``. See this module's own docstring for the two-piece
    algorithm. Returns the dequantized ``[E, D]`` reconstruction.
    """
    num_experts = w_e_d.shape[0]
    mean, basis = _klt_basis(w_e_d, rank)
    coeff = (w_e_d - mean) @ basis.T  # [E, r]
    shared = mean + coeff @ basis  # [E, D] -- this group's shared component

    num_codes = 2**bits
    reconstructed = np.empty_like(w_e_d)
    for e in range(num_experts):
        residual = w_e_d[e] - shared[e]
        centroids, assignments = _kmeans_1d(residual, num_codes, kmeans_iters, seed)
        reconstructed[e] = shared[e] + centroids[assignments]
    return reconstructed


def apply_kbvq_moe(
    model: Union[str, onnx.ModelProto],
    rank: int = 4,
    bits: int = 4,
    kmeans_iters: int = 20,
    seed: int = 0,
) -> onnx.ModelProto:
    """KBVQ-MoE's shared-KLT-basis-plus-per-expert-residual-VQ (simulated)
    quantization of every matched ``com.microsoft::MoE`` node's per-expert
    ``fc1_experts_weights``/``fc2_experts_weights``. See this module's own
    docstring for the technique and its exact scope (FLOAT32 only,
    simulated quantization, matcher shared with
    :func:`onnxsim.moequant.apply_moequant`). Needs no calibration data:
    like :func:`onnxsim.low_rank_compensation.apply_low_rank_compensation`,
    the shared basis and per-expert residual codebooks are both computed
    directly from the weight tensors themselves.

    :param model: onnx ModelProto object or file path
    :param rank: the shared KLT basis's own rank (clamped to
            ``min(rank, num_experts, D)`` per ``fc1``/``fc2`` tensor, where
            ``D`` is that tensor's own per-expert element count); larger
            values let the shared basis capture more of the group's
            cross-expert structure, at the cost of a proportionally larger
            basis to store once per router group.
    :param bits: each expert's own residual codebook size is ``2**bits``
            (default 4, matching :func:`onnxsim.kmeans_quantization.
            quantize_weight_only_kmeans`'s own default)
    :param kmeans_iters: maximum Lloyd's-algorithm iterations per expert's
            own residual codebook fit, passed straight through to
            :func:`onnxsim.kmeans_quantization._kmeans_1d`
    :param seed: seed for :func:`onnxsim.kmeans_quantization._kmeans_1d`'s
            own random fallback centroid samples
    :returns: ``model`` with every matched, FLOAT32 MoE node's per-expert
            ``fc1``/``fc2`` weight overwritten in place with its
            shared-basis-plus-dequantized-residual reconstruction. A
            router group whose ``fc1``/``fc2`` tensor is not FLOAT32 is
            left untouched, the same restriction
            :func:`onnxsim.moequant.apply_moequant` applies.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    chains = _find_moe_chains(model.graph)
    if not chains:
        return model

    result = onnx.ModelProto()
    result.CopyFrom(model)
    initializer_map = {t.name: t for t in result.graph.initializer}
    touched: "set[str]" = set()

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w}
        if weight_names & touched:
            continue  # a shared/tied initializer another MoE node already quantized
        touched |= weight_names

        for w_name in (chain.fc1_w, chain.fc2_w):
            init = initializer_map[w_name]
            if init.data_type != onnx.TensorProto.FLOAT:
                continue  # FLOAT16/BFLOAT16 experts are out of scope -- see docstring

            w = onnx.numpy_helper.to_array(init).astype(np.float64)
            shape = w.shape
            flat = w.reshape(shape[0], -1)
            reconstructed = _kbvq_reconstruct(flat, rank, bits, kmeans_iters, seed)
            new_w = reconstructed.reshape(shape).astype(np.float32)
            init.CopyFrom(onnx.numpy_helper.from_array(new_w, name=w_name))

    return result
