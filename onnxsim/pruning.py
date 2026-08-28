"""Post-training weight pruning for MatMul/vanilla-Gemm (and, for structured
pruning, Conv) layers.

Surveying the pruning literature against what onnxsim can actually act on
(an exported ONNX graph, no training loop, no gradients, usually no labels)
narrows the field a lot. Most well-known pruning *tools* --
``torch.nn.utils.prune``, NNI's pruning API, Neural Magic's SparseML, Intel
Neural Compressor's pruning API -- assume a live framework model mid-training
or at least a fine-tuning loop to recover accuracy after each pruning step
(iterative magnitude pruning / the Lottery Ticket Hypothesis, movement
pruning, "pattern lock" pruning, ...). That is the same reason onnxsim's
existing weight-only quantization stack (:mod:`onnxsim.gptq`,
:mod:`onnxsim.awq`, ...) reimplements each technique's *algorithm* against
raw ONNX MatMul/Gemm weights rather than depending on those libraries
directly: they operate one level up, on a model object onnxsim never has.

*Structured* pruning (removing whole channels/filters, e.g. Torch-Pruning,
NNI's L1/L2 filter pruning, network slimming, or the expert-intermediate-
channel/Mamba-state pruning inside NVIDIA's "Iterative Puzzle" compression
pipeline for hybrid MoE LLMs, https://arxiv.org/abs/2607.04371) is a
fundamentally bigger project than the rest of this module for two separate
reasons, and this module only takes on one of them. It *does* change tensor
shapes, which ripples through every downstream consumer of the pruned
dimension -- real graph surgery, not the self-contained per-layer weight
rewrite every other ``apply_*``/``quantize_*`` pass in onnxsim is. That part
:func:`apply_structured_pruning` takes on, but deliberately only for the
narrowest topology where the surgery is unambiguous: a single MatMul/Gemm
or ordinary (``group=1``) Conv whose output feeds, through a chain of
shape-preserving elementwise ops (activations, and for MatMul/Gemm also a
bias/scale add/mul) with no other consumer anywhere along that chain, into
exactly one downstream layer of the same family whose reduction/input-
channel dimension matches.
Any residual/skip connection, multi-consumer fan-out, or branch (all of
which need real dependency-graph analysis -- what Torch-Pruning's DepGraph
does in general) is left untouched rather than guessed at. The other part
of the paper's pipeline -- an architecture *search* over what to prune,
alternated with knowledge-distillation/RL recovery afterwards -- needs a
training loop onnxsim does not have and is not in scope here at all; this
is a single, static, no-retraining structural cut, closer in spirit to Li
et al.'s L2-norm filter pruning (below) than to anything iterative.

What *does* fit that mold, and needs no retraining loop: post-training
*unstructured* (or semi-structured N:M) pruning, à la magnitude pruning
(Han et al., 2015, "Learning both Weights and Connections for Efficient
Neural Networks", https://arxiv.org/abs/1506.02626) and, for the
calibrated variant, Wanda (Sun et al., 2023, "A Simple and Effective
Pruning Approach for Large Language Models",
https://arxiv.org/abs/2306.11695 -- the pruning analogue of this module's
neighbors :mod:`onnxsim.awq`/:mod:`onnxsim.smoothquant`: a single forward
pass over calibration data, no weight update, no backward pass at all).
Both zero out individual weight entries and leave every tensor's shape
exactly as it was, so -- like every ``quantize_weight_only_*`` pass here --
the result is a plain ONNX model, correct by construction (a MatMul/Gemm
with some zeroed entries computes the same op, just with less nonzero
data), that a runtime with sparse-kernel support (or a later, separate
dense-to-sparse repacking step) can exploit for speed.

:func:`apply_magnitude_pruning` uses ``|W|`` as the importance metric and
needs no calibration data at all -- the simple, data-free baseline.
:func:`apply_wanda_pruning` weights that by each input feature's activation
norm over calibration data (``|W_ij| * ||X_j||_2``), which -- per the
Wanda paper -- better protects weights that multiply high-magnitude
activations even when the weight itself is individually small, the same
class of outlier-activation effect that motivates :mod:`onnxsim.smoothquant`.

Both support two sparsity patterns, chosen per invocation:

- unstructured: for every output row (comparison group), the lowest-
  importance entries are zeroed until that row reaches the target
  ``sparsity`` fraction.
- semi-structured N:M (e.g. ``n=2, m=4`` -- NVIDIA Ampere's 2:4 structured
  sparsity, the pattern Wanda's own paper evaluates most): within every
  consecutive group of ``m`` input-channel entries in a row, only the
  ``n`` highest-importance survive.

:func:`weight_sparsity` reports the fraction of exact-zero entries across
every matched layer's weight, as a quick way to confirm a pruning call
reached its target (or to measure an already-sparse model).

:func:`apply_structured_pruning` actually removes channels (real shape
reduction, real FLOP/parameter reduction on any runtime, no sparse-kernel
support needed) from every producer -> consumer chain it can prove safe to
cut, per output-channel L2-norm importance (Li et al., 2017, "Pruning
Filters for Efficient ConvNets", https://arxiv.org/abs/1608.08710) -- for a
MatMul/Gemm chain, that criterion is a transplant from Conv filters to
output channels (the same one :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning` already made for Han et al./Wanda's element-wise
criteria); for a Conv chain it is the paper's own original setting, applied
directly: each output filter's full ``[in_channels, kH, kW]`` kernel is
flattened and ranked by its own L2 norm. Conv support is deliberately
narrower than the MatMul/Gemm path: only ordinary (``group=1``) 2-D
``Conv`` producers/consumers are matched, joined by unary activations alone
-- no per-channel ``Add``/``Mul`` scale-or-bias op, since a real Conv
already carries any bias in its own optional third input, and
``BatchNormalization`` is expected to already be fused into the preceding
Conv's weight by the time this pass runs (onnxsim's own default
optimization does exactly that, see ``fuse_bn_into_conv``), so a raw
per-channel affine between two Convs isn't a shape this pass special-cases.
A *general* grouped Conv (``group`` neither 1 nor equal to its channel
count) is left untouched entirely: its output and input channels aren't
independent per-index the way this pass's single producer/consumer cut
assumes, and safely cutting a shared group's channels needs real
group-aware bookkeeping this pass does not attempt (the same boundary
attention-head pruning below draws around GroupQueryAttention). The
*depthwise* special case (``group == in_channels == out_channels``,
weight ``[C, 1, kH, kW]``) is different: with one filter per channel and
no cross-channel mixing at all, output channel ``i`` depends only on
input channel ``i``, so a depthwise Conv sitting between a chain's real
producer and real consumer needs no independent importance of its own --
the chain walk (:func:`_walk_to_conv_consumer`) crosses it transparently,
like one more shape-preserving activation hop, carrying whatever
channel-index set survives upstream straight through unchanged, while
still slicing that depthwise layer's own weight/bias by the same indices
and shrinking its ``group`` attribute to match. This is exactly the
``Conv(1x1, group=1) -> DepthwiseConv(3x3, group=C) -> Conv(1x1,
group=1)`` "inverted residual" block MobileNet/EfficientNet-style
efficient CNN backbones use throughout, so it's worth the special case; a
depthwise Conv is never itself matched as a producer or consumer (see
:func:`_match_conv_producer`/:func:`_match_conv_consumer`), only ever a
transparent hop between two real ``group=1`` Conv boundaries -- one
sitting last before a graph output or an unhandled branch simply ends the
chain unmatched, same as any other topology this pass declines to guess
at.
:func:`apply_structured_wanda_pruning` is the calibrated upgrade of that
same technique -- ``||W_row||_2 * ||X||_2`` per channel instead of weight
magnitude alone -- exactly the same relationship Wanda has to plain
magnitude pruning, transplanted from individual weights (or, for Conv,
whole filters) to whole channels. Because either changes shapes, the
result is unconditionally irreversible and, unlike a retrained pipeline,
has no distillation/RL step to recover whatever accuracy the cut costs --
evaluate the result before shipping it, the same caution any lossy onnxsim
pass deserves.

:func:`apply_sparsegpt_pruning` is a third, more accurate way to reach an
unstructured or N:M pattern (alongside magnitude and Wanda pruning above):
SparseGPT (Frantar & Alistarh, 2023, "SparseGPT: Massive Language Models
Can Be Accurately Pruned in One-Shot", https://arxiv.org/abs/2301.00774) --
the pruning sibling of :mod:`onnxsim.gptq`, from the same authors, reusing
the exact same machinery (:func:`onnxsim.gptq._inverse_hessian_cholesky`'s
Cholesky-factored inverse Hessian, and the same left-to-right,
error-propagating column processing) but pruning each column to a mask
instead of quantizing it to a grid point. Where magnitude/Wanda pick a
mask once from a static (weight- or weight-times-activation-) importance
score and stop, SparseGPT computes each column's OBS-style saliency score
``w_ij^2 / Hinv_jj^2`` from calibration data, then -- after masking a
column -- propagates the resulting reconstruction error into every
not-yet-processed column via the same Hessian-based correction GPTQ uses
for quantization error, so later columns compensate for earlier ones'
removal instead of every column being scored independently against the
original, uncorrected weights. This reliably beats magnitude/Wanda at the
same sparsity, at the cost of needing calibration data (there is no
data-free variant, unlike magnitude vs. Wanda) and being noticeably more
expensive per layer (one Cholesky factorization plus a sequential,
Hessian-propagating pass over every column, rather than one static
element-wise score). Ported directly from the reference implementation's
``fasterprune`` (https://github.com/IST-DASLab/sparsegpt), including one
behavior that's otherwise a departure from every other function in this
module: for *unstructured* sparsity, the reference selects one threshold
per ``proc_block_size``-wide column block, shared across every output row
in that block, rather than :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning`'s per-row threshold -- faithfully reproduced
here rather than "corrected" to match, since the point of this function is
to reproduce SparseGPT specifically. N:M pruning is unaffected (it is
already per-row in the reference too, and matches this module's own
``n``/``m`` convention exactly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _inverse_hessian_cholesky
from onnxsim.smoothquant import _match_matmul_like


def _validate_pattern(sparsity: float, n: Optional[int], m: Optional[int]) -> None:
    if (n is None) != (m is None):
        raise ValueError("n and m must be given together (N:M pruning) or not at all")
    if n is not None and m is not None:
        if not (0 < n <= m):
            raise ValueError(f"require 0 < n <= m, got n={n}, m={m}")
    elif not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")


def _sparsity_mask(importance: np.ndarray, sparsity: float) -> np.ndarray:
    # Per-row (per-output-channel) threshold, matching Wanda's own
    # per-output comparison group rather than a single global threshold --
    # a layer with output-channel-dependent weight/activation scale would
    # otherwise have some rows pruned to nothing and others left untouched.
    rows, cols = importance.shape
    keep = max(1, round(cols * (1.0 - sparsity)))
    if keep >= cols:
        return np.ones((rows, cols), dtype=bool)
    order = np.argsort(importance, axis=1)
    drop = order[:, : cols - keep]
    mask = np.ones((rows, cols), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    return mask


def _nm_mask(importance: np.ndarray, n: int, m: int) -> np.ndarray:
    """Row-wise N:M mask: within every consecutive group of ``m`` columns,
    keeps only the ``n`` highest-importance entries. A trailing partial
    group (fewer than ``m`` columns) keeps a proportional share (rounded,
    at least 1) instead of raising on a non-multiple-of-``m`` width.
    """
    rows, cols = importance.shape
    mask = np.ones((rows, cols), dtype=bool)
    full_cols = (cols // m) * m
    if full_cols:
        groups = importance[:, :full_cols].reshape(rows, full_cols // m, m)
        order = np.argsort(groups, axis=2)
        drop = order[:, :, : m - n]
        group_mask = np.ones_like(groups, dtype=bool)
        np.put_along_axis(group_mask, drop, False, axis=2)
        mask[:, :full_cols] = group_mask.reshape(rows, full_cols)
    tail = cols - full_cols
    if tail:
        keep = min(tail, max(1, round(n * tail / m)))
        tail_importance = importance[:, full_cols:]
        order = np.argsort(tail_importance, axis=1)
        drop = order[:, : tail - keep]
        tail_mask = np.ones((rows, tail), dtype=bool)
        np.put_along_axis(tail_mask, drop, False, axis=1)
        mask[:, full_cols:] = tail_mask
    return mask


def _candidates(graph: onnx.GraphProto):
    initializer_map = {t.name: t for t in graph.initializer}
    out = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        out.append((node, x_name, w_name, weight_transposed))
    return out


def _prune_weight(
    w_init: onnx.TensorProto, weight_transposed: bool, importance_of_nk
) -> None:
    w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
    dim0, dim1 = w.shape
    w_nk = w if weight_transposed else w.T  # [N, K], output channel first
    mask = importance_of_nk(w_nk)
    w_pruned_nk = np.where(mask, w_nk, 0.0)
    w_new = w_pruned_nk if weight_transposed else w_pruned_nk.T
    w_new = w_new.reshape(dim0, dim1).astype(np.float32)
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def apply_magnitude_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
) -> onnx.ModelProto:
    """Zeros the least-magnitude entries of every MatMul/vanilla-Gemm
    layer's constant 2-D float32 weight -- the data-free pruning baseline
    (Han et al., 2015). See this module's own docstring for how importance
    is grouped and why structured (shape-changing) pruning isn't offered.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each row's entries to zero,
            ignored when ``n``/``m`` are given
    :param n: keep the ``n`` highest-magnitude entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :returns: ``model`` with every matched layer's weight zeroed in place
            to the target pattern; layers with a non-constant or non-2-D
            weight are left untouched
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializer_map = {t.name: t for t in out.graph.initializer}

    for _, _, w_name, weight_transposed in _candidates(out.graph):
        w_init = initializer_map[w_name]

        def importance_of_nk(w_nk, n=n, m=m, sparsity=sparsity):
            importance = np.abs(w_nk)
            return (
                _nm_mask(importance, n, m)
                if n is not None
                else _sparsity_mask(importance, sparsity)
            )

        _prune_weight(w_init, weight_transposed, importance_of_nk)

    return out


def apply_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Wanda pruning (Sun et al., 2023): zeros the least-important entries
    of every MatMul/vanilla-Gemm layer's constant 2-D float32 weight, using
    ``|W_ij| * ||X_j||_2`` (weight magnitude times its input channel's
    activation norm over calibration data) as the importance metric instead
    of plain ``|W|``. See this module's own docstring for the technique and
    :func:`apply_magnitude_pruning` for the calibration-free baseline this
    upgrades.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation norm on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each row's entries to zero,
            ignored when ``n``/``m`` are given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every entry of an all-zero channel tying at
            exactly-zero importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight zeroed in place
            to the target pattern; layers with a non-constant, non-2-D
            weight, or whose activation input isn't a plain 2-D tensor
            matching the weight's reduction dimension, fall back to plain
            magnitude pruning (no activation norm was ever observed)
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}

    candidates = _candidates(graph)
    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            s = np.square(x).sum(axis=0)
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + x.shape[0]

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }

    for _, x_name, w_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        norm = act_norm.get(x_name)

        def importance_of_nk(w_nk, norm=norm, n=n, m=m, sparsity=sparsity):
            if norm is None or norm.shape[0] != w_nk.shape[1]:
                importance = np.abs(w_nk)  # fall back to plain magnitude
            else:
                importance = np.abs(w_nk) * np.maximum(norm, epsilon)[np.newaxis, :]
            return (
                _nm_mask(importance, n, m)
                if n is not None
                else _sparsity_mask(importance, sparsity)
            )

        _prune_weight(w_init, weight_transposed, importance_of_nk)

    return out


def weight_sparsity(model: Union[str, onnx.ModelProto]) -> float:
    """Fraction of exact-zero entries across every matched MatMul/vanilla-
    Gemm layer's constant 2-D float32 weight -- a quick way to confirm a
    pruning call reached its target, or to measure an already-sparse model.
    Returns ``0.0`` if no matching layer is present.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    zeros = 0
    total = 0
    initializer_map = {t.name: t for t in model.graph.initializer}
    for _, _, w_name, _ in _candidates(model.graph):
        w = onnx.numpy_helper.to_array(initializer_map[w_name])
        zeros += int(np.count_nonzero(w == 0))
        total += w.size

    return zeros / total if total else 0.0


# --- SparseGPT ----------------------------------------------------------


def _sparsegpt_prune_columns(
    w_nk: np.ndarray,
    h: np.ndarray,
    sparsity: float,
    n: Optional[int],
    m: Optional[int],
    percdamp: float,
    proc_block_size: int,
) -> np.ndarray:
    """Returns SparseGPT-pruned values for ``w_nk`` ([N, K], output channel
    first), a direct port of the reference implementation's own
    ``fasterprune`` (https://github.com/IST-DASLab/sparsegpt/blob/master/
    sparsegpt.py). Unlike :func:`_prune_weight`'s ``importance_of_nk``
    callbacks, this returns fully-formed replacement values, not a mask --
    every *kept* entry may also change, having accumulated Hessian-based
    compensation for every *pruned* entry processed before it.
    """
    n_rows, k = w_nk.shape
    diag = np.arange(k)
    dead = h[diag, diag] == 0.0

    w_work = w_nk.copy()
    w_work[:, dead] = 0.0
    w_pruned = np.zeros_like(w_work)

    if n is None and sparsity <= 0.0:
        return w_nk.copy()  # true no-op, rather than the reference's own
        # "always drop the single lowest-scoring entry" edge case at
        # sparsity == 0.0 -- matching every other apply_*_pruning function
        # in this module, all of which treat sparsity=0.0 as a no-op.

    hinv = _inverse_hessian_cholesky(h, percdamp)

    for i1 in range(0, k, proc_block_size):
        i2 = min(i1 + proc_block_size, k)
        count = i2 - i1
        w1 = w_work[:, i1:i2].copy()
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        hinv1_diag = np.diag(hinv1)

        if n is None:
            score = np.square(w1) / np.square(hinv1_diag)[np.newaxis, :]
            thresh = np.sort(score.reshape(-1))[int(score.size * sparsity)]
            mask1 = score <= thresh
        else:
            mask1 = np.zeros_like(w1, dtype=bool)

        for i in range(count):
            if n is not None and m is not None and i % m == 0:
                group_end = min(i + m, count)
                group_score = (
                    np.square(w1[:, i:group_end])
                    / np.square(hinv1_diag[i:group_end])[np.newaxis, :]
                )
                prune_count = min(group_end - i, m - n)
                mask1[:, i:group_end] = False
                if prune_count > 0:
                    drop_local = np.argsort(group_score, axis=1)[:, :prune_count]
                    np.put_along_axis(mask1[:, i:group_end], drop_local, True, axis=1)

            w_col = w1[:, i]
            d = hinv1_diag[i]
            q_col = np.where(mask1[:, i], 0.0, w_col)
            w_pruned[:, i1 + i] = q_col

            err = (w_col - q_col) / d
            err1[:, i] = err
            if i + 1 < count:
                w1[:, i + 1 :] -= np.outer(err, hinv1[i, i + 1 :])

        if i2 < k:
            w_work[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return w_pruned


def apply_sparsegpt_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """SparseGPT (Frantar & Alistarh, 2023): zeros the least-important
    entries of every MatMul/vanilla-Gemm layer's constant 2-D float32
    weight, the same unstructured-or-N:M patterns
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning` offer, but
    -- unlike either -- using a sequential, Hessian-error-compensating
    algorithm ported from GPTQ (:mod:`onnxsim.gptq`, same authors, same
    Cholesky-factored inverse Hessian) rather than a one-shot static
    importance score. See this module's own docstring for the technique,
    including the one deliberate departure from every other function here:
    for unstructured sparsity, the pruning threshold is shared across every
    output row within each ``proc_block_size``-wide column block (the
    reference implementation's own behavior), not chosen per row.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of entries to zero (shared per column
            block, not per row -- see above), ignored when ``n``/``m`` are
            given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4, per-row exactly
            as :func:`apply_magnitude_pruning`). Must be given together
            with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :func:`onnxsim.apply_gptq`'s own default
    :param proc_block_size: column-processing block size -- both the
            lazy-update granularity (how many columns' errors accumulate
            locally before a full cross-block update, matching
            :func:`onnxsim.apply_gptq`'s ``proc_block_size``) and, for
            unstructured sparsity only, the width each shared per-block
            threshold is computed over
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight rewritten in
            place to the target pattern -- every surviving entry may also
            change value, having accumulated compensation for entries
            pruned before it; a layer with no observed 2-D calibration
            activation (dead input, or every batch's activation isn't
            plain 2-D/higher-rank-with-a-trailing-feature-axis) is left
            completely untouched -- unlike Wanda, there is no data-free
            fallback for a technique whose entire mechanism is the Hessian
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}

    candidates = _candidates(graph)
    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 2:
                continue
            activations[name].append(x.reshape(-1, x.shape[-1]))

    for _, x_name, w_name, weight_transposed in candidates:
        acts = activations[x_name]
        if not acts:
            continue
        x = np.concatenate(acts, axis=0)

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        if x.shape[1] != w_nk.shape[1]:
            continue

        h = x.T @ x
        w_pruned_nk = _sparsegpt_prune_columns(
            w_nk, h, sparsity, n, m, percdamp, proc_block_size
        )

        w_new = w_pruned_nk if weight_transposed else w_pruned_nk.T
        w_new = w_new.reshape(dim0, dim1).astype(np.float32)
        w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))

    return out


# --- Structured (channel) pruning -------------------------------------------

# Shape-preserving, channel-order-preserving elementwise ops that may sit
# between a producer and consumer without blocking the chain: unary
# activations (single input, single output, no other operand to worry
# about) and Add/Mul against a constant per-channel bias/scale.
_UNARY_PASS_THROUGH = {
    "Relu",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Sigmoid",
    "Tanh",
    "Softplus",
    "Softsign",
    "Gelu",
    "HardSigmoid",
    "Mish",
    "Identity",
    "Cast",
}
_BINARY_CHANNEL_OPS = {"Add", "Mul"}
_MAX_CHAIN_HOPS = 8

_ConsumerMatch = Tuple[onnx.NodeProto, str, bool]  # (node, weight, weight_transposed)


@dataclass(frozen=True)
class _Producer:
    node: onnx.NodeProto
    weight: str
    weight_transposed: bool
    bias: Optional[str]
    # Activation nodes between this producer's raw output and the point it
    # combines with another producer (a gated pair only -- see
    # :func:`_find_gated_chains`; empty for a plain single-producer chain).
    pre_ops: Tuple[onnx.NodeProto, ...] = ()
    # True for a Conv producer: `weight_transposed` is meaningless then
    # (Conv's ``[out_channels, in_channels, kH, kW]`` weight layout is
    # fixed), and output channels always live on axis 0.
    is_conv: bool = False


@dataclass(frozen=True)
class _ConvPassThrough:
    """A depthwise Conv (``group == in_channels == out_channels``) the chain
    walk crossed transparently between a Conv chain's real producer and real
    consumer. A depthwise Conv mixes no channels at all -- output channel
    ``i`` depends only on input channel ``i`` -- so it needs no independent
    importance of its own the way a producer/consumer boundary does; it is
    carried on the matched :class:`_Chain` purely so :func:`_apply_chains`
    can slice its own ``[C, 1, kH, kW]`` weight (and bias, if present) by
    the *same* `keep` index set as the chain's real producer, and update its
    ``group`` attribute to the new channel count. See
    :func:`_walk_to_conv_consumer`.
    """

    node: onnx.NodeProto
    weight: str
    bias: Optional[str]


@dataclass(frozen=True)
class _Chain:
    # One producer for a plain chain; two for a gated (elementwise-product)
    # pair, where both branches must agree on which channels survive.
    producers: Tuple[_Producer, ...]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    n_channels: int
    # True for a Conv consumer: input channels always live on axis 1 of its
    # ``[out_channels, in_channels, kH, kW]`` weight, regardless of
    # `consumer_weight_transposed` (unused then).
    consumer_is_conv: bool = False
    # Depthwise Conv hops the chain walk crossed transparently between the
    # real producer and the real consumer (Conv chains only -- see
    # :class:`_ConvPassThrough`; always empty for a MatMul/Gemm chain).
    conv_pass_through: Tuple[_ConvPassThrough, ...] = ()


def _consumers_of(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _match_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, bool, Optional[str], int]]:
    """If `node` is a MatMul/vanilla-Gemm with a constant 2-D float32
    weight (and, for Gemm, either no bias or a constant one), returns
    ``(weight_name, weight_transposed, bias_name_or_None, n_channels)``.
    """
    match = _match_matmul_like(node)
    if match is None:
        return None
    _, w_name, weight_transposed = match
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 2
    ):
        return None
    bias_name = None
    if node.op_type == "Gemm" and len(node.input) == 3:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    n_channels = w_init.dims[0] if weight_transposed else w_init.dims[1]
    return w_name, weight_transposed, bias_name, n_channels


def _walk_to_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
) -> Tuple[Optional[_ConsumerMatch], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]:
    """From tensor `start`, walks forward through shape-preserving
    elementwise ops (an activation, or an Add/Mul against a constant
    per-channel bias/scale) with no other consumer anywhere along the way,
    until a MatMul/vanilla-Gemm consumer is found whose reduction
    dimension matches `n_channels`. Returns ``(None, ())`` if the walk
    runs out of hops, hits a branch, or never reaches such a consumer.
    """
    chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
    consumer = None
    cur = start
    for _hop in range(max_hops):
        candidates = consumers_of.get(cur, [])
        if len(candidates) != 1:
            break
        nxt = candidates[0]

        cm = _match_matmul_like(nxt)
        if cm is not None and cm[0] == cur:
            _, cw_name, c_weight_transposed = cm
            cw_init = initializer_map.get(cw_name)
            if (
                cw_init is not None
                and cw_init.data_type == onnx.TensorProto.FLOAT
                and len(cw_init.dims) == 2
            ):
                k = cw_init.dims[1] if c_weight_transposed else cw_init.dims[0]
                if k == n_channels:
                    consumer = (nxt, cw_name, c_weight_transposed)
            break

        const_name: Optional[str] = None
        if (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            pass
        elif (
            nxt.op_type in _BINARY_CHANNEL_OPS
            and len(nxt.input) == 2
            and cur in nxt.input
            and len(nxt.output) == 1
        ):
            other = nxt.input[1] if nxt.input[0] == cur else nxt.input[0]
            const_init = initializer_map.get(other)
            if (
                const_init is not None
                and const_init.data_type == onnx.TensorProto.FLOAT
                and list(const_init.dims)
                and const_init.dims[-1] == n_channels
                and int(np.prod(const_init.dims)) == n_channels
            ):
                const_name = other
            else:
                break
        else:
            break

        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            break
        chain_ops.append((nxt, const_name))
        cur = out2

    return consumer, tuple(chain_ops)


def _find_chains(graph: onnx.GraphProto) -> List[_Chain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        # Safe to reshape only if exactly one node reads it and it isn't
        # itself something the caller observes (a graph output).
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is None:
            continue
        w_name, weight_transposed, bias_name, n_channels = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_Producer(node, w_name, weight_transposed, bias_name),),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_channels,
            )
        )
    return chains


def _conv_group(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "group":
            return attr.i
    return 1  # ONNX default


def _match_conv_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int]]:
    """If `node` is an ordinary (``group=1``) 2-D ``Conv`` with a constant
    4-D float32 ``[out_channels, in_channels, kH, kW]`` weight (and, if
    present, a constant bias), returns
    ``(weight_name, bias_name_or_None, out_channels)``. A grouped or
    depthwise Conv (``group != 1``) never matches: its output and input
    channels aren't independent per-index the way this pass's single
    producer/consumer cut assumes -- true here too for the depthwise case,
    even though a depthwise Conv *is* given a narrower exception elsewhere
    in this pass, as a transparent pass-through hop the chain walk may
    cross between two real producer/consumer boundaries (see
    :func:`_match_depthwise_conv_pass_through`,
    :func:`_walk_to_conv_consumer`) -- it is never itself matched as a
    producer.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
        or _conv_group(node) != 1
    ):
        return None
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name, w_init.dims[0]


def _match_conv_consumer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, int]]:
    """If `node` is an ordinary (``group=1``) 2-D ``Conv`` with a constant
    4-D float32 weight, returns ``(weight_name, in_channels)``. Like
    :func:`_match_conv_producer`, a depthwise Conv never matches here
    either -- it's only ever a transparent pass-through hop the chain walk
    crosses en route to a *real* ``group=1`` consumer, never a consumer
    itself (see :func:`_match_depthwise_conv_pass_through`).
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
        or _conv_group(node) != 1
    ):
        return None
    return w_name, w_init.dims[1]


def _match_depthwise_conv_pass_through(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    n_channels: int,
) -> Optional[Tuple[str, Optional[str]]]:
    """If `node` is a depthwise 2-D ``Conv`` (``group == in_channels ==
    out_channels == n_channels``) with a constant ``[n_channels, 1, kH,
    kW]`` float32 weight (and, if present, a constant bias), returns
    ``(weight_name, bias_name_or_None)``. A depthwise Conv mixes no channels
    at all -- output channel ``i`` depends only on input channel ``i`` --
    unlike a general grouped Conv (``group`` neither 1 nor `n_channels`),
    which is not matched here and stays out of scope for this pass entirely
    (see :func:`_match_conv_producer`/:func:`_match_conv_consumer`'s own
    docstrings): only in the depthwise case is every output channel tied
    1:1 to the same-index input channel, which is what lets the chain walk
    (:func:`_walk_to_conv_consumer`) treat it as a transparent pass-through
    hop -- carrying whatever channel-index set survives upstream straight
    through, unchanged -- rather than a producer or consumer of its own.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
        or w_init.dims[0] != n_channels
        or w_init.dims[1] != 1
        or _conv_group(node) != n_channels
    ):
        return None
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if b_init is None or b_init.data_type != onnx.TensorProto.FLOAT:
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name


def _walk_to_conv_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
) -> Tuple[
    Optional[Tuple[onnx.NodeProto, str]],
    Tuple[Tuple[onnx.NodeProto, None], ...],
    Tuple[_ConvPassThrough, ...],
]:
    """The Conv analogue of :func:`_walk_to_consumer`: from tensor `start`,
    walks forward through unary shape-preserving activations (see
    `_UNARY_PASS_THROUGH`) and depthwise Conv hops (see
    :func:`_match_depthwise_conv_pass_through` -- transparent to the
    channel-index mapping, but each still needs its own weight/bias sliced
    and its ``group`` attribute updated, so they're returned separately as
    `conv_pass_through` rather than folded into `chain_ops`) with no other
    consumer anywhere along the way, until an ordinary (``group=1``) Conv
    consumer is found whose input channel count matches `n_channels`. A
    depthwise Conv is only ever a transparent hop, never a match for the
    consumer role itself -- one sitting last before a graph output or a
    branch simply ends the walk with no consumer found, same as any other
    unmatched topology. Unlike the MatMul/Gemm walk, no per-channel
    ``Add``/``Mul`` op is recognized -- see this module's own docstring for
    why that's out of scope for Conv chains.
    """
    chain_ops: List[Tuple[onnx.NodeProto, None]] = []
    conv_pass_through: List[_ConvPassThrough] = []
    consumer = None
    cur = start
    for _hop in range(max_hops):
        candidates = consumers_of.get(cur, [])
        if len(candidates) != 1:
            break
        nxt = candidates[0]

        if nxt.op_type == "Conv" and nxt.input[0] == cur:
            depthwise = _match_depthwise_conv_pass_through(
                nxt, initializer_map, n_channels
            )
            if depthwise is not None:
                out2 = nxt.output[0]
                if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
                    break
                dw_weight, dw_bias = depthwise
                conv_pass_through.append(_ConvPassThrough(nxt, dw_weight, dw_bias))
                cur = out2
                continue

            match = _match_conv_consumer(nxt, initializer_map)
            if match is not None and match[1] == n_channels:
                consumer = (nxt, match[0])
            break

        if not (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            break

        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            break
        chain_ops.append((nxt, None))
        cur = out2

    return consumer, tuple(chain_ops), tuple(conv_pass_through)


def _find_conv_chains(graph: onnx.GraphProto) -> List[_Chain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_conv_producer(node, initializer_map)
        if info is None:
            continue
        w_name, bias_name, n_channels = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops, conv_pass_through = _walk_to_conv_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_Producer(node, w_name, False, bias_name, is_conv=True),),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=False,
                n_channels=n_channels,
                consumer_is_conv=True,
                conv_pass_through=conv_pass_through,
            )
        )
    return chains


def _trace_gate_producer_backward(
    tensor_name: str,
    node_by_output: Dict[str, onnx.NodeProto],
    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    max_hops: int,
) -> Optional[
    Tuple[
        Tuple[onnx.NodeProto, str, bool, Optional[str], int], Tuple[onnx.NodeProto, ...]
    ]
]:
    """Walks backward from `tensor_name` through unary activation ops
    (Sigmoid, Gelu, ...) until it resolves to a matmul-like producer's raw
    output -- the mirror image of :func:`_walk_to_consumer`'s forward walk,
    used to recognize a gate branch's own activation (e.g. SwiGLU's
    ``silu(gate)`` when exported as separate Sigmoid/Mul-by-a-second-
    operand rather than a single node -- see :func:`_find_gated_chains`).
    Every tensor walked through, `tensor_name` itself included, must have
    exactly one consumer and not be a graph output: the same safety bar
    the forward walk holds every intermediate tensor to.
    """
    pre_ops: List[onnx.NodeProto] = []
    cur = tensor_name
    for _ in range(max_hops):
        if len(consumers_of.get(cur, [])) != 1 or cur in graph_outputs:
            return None
        if cur in producer_infos:
            return producer_infos[cur], tuple(reversed(pre_ops))
        producer_node = node_by_output.get(cur)
        if producer_node is None:
            return None
        if not (
            producer_node.op_type in _UNARY_PASS_THROUGH
            and len(producer_node.input) == 1
            and len(producer_node.output) == 1
        ):
            return None
        pre_ops.append(producer_node)
        cur = producer_node.input[0]
    return None


def _find_gated_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds gated FFN blocks -- SwiGLU/GeGLU-style ``down(act(gate(x)) *
    up(x))``, the FFN architecture most current LLMs use (Llama, Mistral,
    Qwen, Gemma, ...) -- that :func:`_find_chains` cannot see at all,
    because it only ever follows a *single* producer's output. Two
    matmul-like producers (gate and up) whose outputs, each optionally
    through its own activation, combine via one of:

    - a plain elementwise ``Mul`` of two non-constant operands (covers an
      unactivated GLU, or any activation expressed as ordinary unary ops
      -- e.g. GeGLU's ``Gelu``); or
    - ONNX's native fused ``SwiGLU(a, b[, alpha]) = swish(a) * b`` node
      (opset 28+), whose swish lives entirely inside the op, so ``a``/``b``
      must be the two producers' raw outputs with nothing in between,

    with no other consumer anywhere along either branch or at the combine
    point, into exactly one downstream MatMul/vanilla-Gemm's reduction
    dimension, are pruned together: both branches must drop the *same*
    output-channel indices, since they're about to be multiplied
    elementwise. A gate activation decomposed into more than one node
    (e.g. SiLU exported as the self-referencing ``x * Sigmoid(x)`` rather
    than a single ``Sigmoid``/native ``Swish``) isn't recognized -- that
    block is safely left untouched, not guessed at.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}
    node_by_output = {out: node for node in graph.node for out in node.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]] = {}
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is not None:
            w_name, weight_transposed, bias_name, n_channels = info
            producer_infos[node.output[0]] = (
                node,
                w_name,
                weight_transposed,
                bias_name,
                n_channels,
            )

    def _producer(info, pre_ops) -> _Producer:
        node, w_name, weight_transposed, bias_name, _n = info
        return _Producer(node, w_name, weight_transposed, bias_name, pre_ops)

    chains: List[_Chain] = []
    for node in graph.node:
        if node.op_type == "Mul" and len(node.input) == 2 and len(node.output) == 1:
            a_name, b_name = node.input
            if (
                a_name == b_name
                or a_name in initializer_map
                or b_name in initializer_map
            ):
                continue
            trace_a = _trace_gate_producer_backward(
                a_name,
                node_by_output,
                producer_infos,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            trace_b = _trace_gate_producer_backward(
                b_name,
                node_by_output,
                producer_infos,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            if trace_a is None or trace_b is None:
                continue
            info_a, pre_a = trace_a
            info_b, pre_b = trace_b
        elif (
            node.op_type == "SwiGLU" and len(node.input) == 2 and len(node.output) == 1
        ):
            a_name, b_name = node.input
            if a_name in initializer_map or b_name in initializer_map:
                continue
            if not (_is_internal(a_name) and _is_internal(b_name)):
                continue
            info_a_lookup = producer_infos.get(a_name)
            info_b_lookup = producer_infos.get(b_name)
            if info_a_lookup is None or info_b_lookup is None:
                continue
            info_a, pre_a = info_a_lookup, ()
            info_b, pre_b = info_b_lookup, ()
        else:
            continue

        node_a, n_a = info_a[0], info_a[4]
        node_b, n_b = info_b[0], info_b[4]
        if node_a is node_b or n_a != n_b:
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, n_a, _MAX_CHAIN_HOPS
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_producer(info_a, pre_a), _producer(info_b, pre_b)),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_a,
            )
        )
    return chains


def _slice_producer_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    keep: np.ndarray,
    is_conv: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    if is_conv:
        # [out_channels, in_channels, kH, kW]: output channel is always axis 0.
        w_new = w[keep, ...]
    else:
        # [N, K] storage (transB=1): output channel is axis 0. [K, N]
        # storage (the common case): output channel is axis 1.
        w_new = w[keep, :] if weight_transposed else w[:, keep]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_consumer_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    keep: np.ndarray,
    is_conv: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    if is_conv:
        # [out_channels, in_channels, kH, kW]: input channel is always axis 1.
        w_new = w[:, keep, ...]
    else:
        # [N, K] storage (transB=1): reduction dim is axis 1. [K, N] storage:
        # reduction dim is axis 0.
        w_new = w[:, keep] if weight_transposed else w[keep, :]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_last_axis(init: onnx.TensorProto, keep: np.ndarray) -> None:
    arr = onnx.numpy_helper.to_array(init)
    new = np.take(arr, keep, axis=-1)
    init.CopyFrom(onnx.numpy_helper.from_array(new, name=init.name))


def _plain_structured_importance(
    chain: _Chain, w_arrays_nk: List[np.ndarray]
) -> np.ndarray:
    # Combined (root-sum-square) importance across every producer in this
    # chain: for a plain chain this is just that producer's own L2 norm;
    # for a gated pair, both branches must agree on which channels survive,
    # so their per-channel norms are combined first.
    squared_norm = np.zeros(chain.n_channels, dtype=np.float64)
    for w_nk in w_arrays_nk:
        squared_norm += np.square(np.linalg.norm(w_nk, axis=1))
    return np.sqrt(squared_norm)


def _apply_chains(
    graph: onnx.GraphProto,
    chains: List[_Chain],
    sparsity: float,
    compute_importance,
) -> None:
    """Shared body for :func:`apply_structured_pruning` and
    :func:`apply_structured_wanda_pruning`: resolves cross-chain touched-role
    conflicts, computes each surviving chain's target channel count, calls
    ``compute_importance(chain, w_arrays_nk) -> np.ndarray[n_channels]`` for
    the ranking, and performs the actual slicing plus stale ``value_info``
    cleanup. Mutates ``graph`` in place.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    # A weight legitimately plays both roles across two different chains --
    # e.g. the middle layer of a 3-layer MLP is the *consumer* of the first
    # chain (its reduction/input axis gets pruned) and the *producer* of the
    # second (its own output axis gets pruned), two independent axes of the
    # same tensor. Only collapse when the *same role* is claimed twice (a
    # tied/shared weight), tracked separately per role; bias/scale constants
    # only ever play one role, so a single shared set is enough for those.
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    const_touched: Set[str] = set()
    conv_hop_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        producer_weights = {p.weight for p in chain.producers}
        if len(producer_weights) != len(chain.producers):
            continue  # degenerate (a gated pair naming the same weight twice)

        conv_hop_weights = {h.weight for h in chain.conv_pass_through}
        if len(conv_hop_weights) != len(chain.conv_pass_through):
            continue  # degenerate (the same depthwise weight named twice)

        consts = {p.bias for p in chain.producers if p.bias is not None}
        consts.update(
            const_name for _, const_name in chain.chain_ops if const_name is not None
        )
        if (
            (producer_weights & producer_touched)
            or chain.consumer_weight in consumer_touched
            or (consts & const_touched)
            or (conv_hop_weights & conv_hop_touched)
        ):
            continue  # a shared/tied initializer another chain already resized

        n = chain.n_channels
        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        w_arrays_nk = []
        for p in chain.producers:
            w = onnx.numpy_helper.to_array(initializer_map[p.weight]).astype(np.float64)
            if p.is_conv:
                w_nk = w.reshape(w.shape[0], -1)  # [out_channels, in_channels*kH*kW]
            else:
                w_nk = w if p.weight_transposed else w.T  # [N, K]
            w_arrays_nk.append(w_nk)
        importance = compute_importance(chain, w_arrays_nk)
        keep = np.sort(np.argsort(-importance)[:keep_count])

        for p in chain.producers:
            _slice_producer_weight(
                initializer_map[p.weight], p.weight_transposed, keep, is_conv=p.is_conv
            )
            if p.bias is not None:
                _slice_last_axis(initializer_map[p.bias], keep)
        for _, const_name in chain.chain_ops:
            if const_name is not None:
                _slice_last_axis(initializer_map[const_name], keep)
        for hop in chain.conv_pass_through:
            # Same `keep` index set as the real producer -- a depthwise
            # Conv's own channel i is exactly upstream channel i, so its
            # weight (output-channel axis 0, like any Conv producer) and
            # bias slice identically, and `group` (== in_channels ==
            # out_channels for a depthwise Conv) drops to the new count
            # right alongside them.
            _slice_producer_weight(
                initializer_map[hop.weight], False, keep, is_conv=True
            )
            if hop.bias is not None:
                _slice_last_axis(initializer_map[hop.bias], keep)
            found_group = False
            for attr in hop.node.attribute:
                if attr.name == "group":
                    attr.i = keep_count
                    found_group = True
                    break
            if not found_group:
                hop.node.attribute.append(
                    onnx.helper.make_attribute("group", keep_count)
                )
        _slice_consumer_weight(
            initializer_map[chain.consumer_weight],
            chain.consumer_weight_transposed,
            keep,
            is_conv=chain.consumer_is_conv,
        )

        producer_touched.update(producer_weights)
        consumer_touched.add(chain.consumer_weight)
        const_touched.update(consts)
        conv_hop_touched.update(conv_hop_weights)
        for p in chain.producers:
            stale_value_info.add(p.node.output[0])
            stale_value_info.update(pre_op.output[0] for pre_op in p.pre_ops)
        stale_value_info.update(
            chain_node.output[0] for chain_node, _ in chain.chain_ops
        )
        stale_value_info.update(hop.node.output[0] for hop in chain.conv_pass_through)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)


def apply_structured_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """Removes whole output channels from MatMul/vanilla-Gemm layers --
    real structural pruning (smaller weight tensors, smaller matmuls on any
    runtime, not just one with sparse-kernel support), as opposed to
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s value-only
    zeroing. See this module's own docstring for the technique, its L2-norm
    importance metric, and why it's restricted to an unambiguous single
    producer -> consumer topology rather than general dependency-graph
    pruning. :func:`apply_structured_wanda_pruning` is the calibrated
    upgrade of this same technique, exactly as :func:`apply_wanda_pruning`
    is to :func:`apply_magnitude_pruning`.

    For every MatMul/vanilla-Gemm node (the "producer") whose output feeds,
    through zero or more shape-preserving elementwise ops (an activation,
    or an Add/Mul against a constant per-channel bias/scale) with no other
    consumer anywhere along that path, into exactly one downstream
    MatMul/vanilla-Gemm's reduction dimension (the "consumer"): ranks the
    producer's output channels by L2 norm of their own weight row, drops
    the lowest-``sparsity``-fraction of them, and removes the corresponding
    rows/columns from the producer's weight (and bias, if it has a constant
    one) and every intermediate per-channel constant, and the matching
    columns/rows from the consumer's weight -- a shape change that leaves
    the two layers' composition mathematically unaffected for every
    surviving channel.

    The same cut applies to ordinary (``group=1``) 2-D ``Conv`` producer ->
    consumer pairs -- each output filter's whole ``[in_channels, kH, kW]``
    kernel ranked by its own L2 norm, exactly Li et al.'s original filter-
    pruning criterion -- joined by unary activations and/or depthwise Conv
    hops (``group == in_channels == out_channels``: one filter per channel,
    no cross-channel mixing, so it's crossed transparently -- its own
    weight/bias sliced by the producer's channel indices and its ``group``
    attribute shrunk to match, but it contributes no importance of its own
    and can't itself be the producer or consumer -- see this module's own
    docstring). No per-channel Add/Mul between two Convs (a Conv already
    carries its own bias, and ``BatchNormalization`` is expected to already
    be fused into the preceding Conv by the time this pass runs). A
    *general* grouped Conv (``group`` neither 1 nor its channel count) is
    left untouched.

    Also handles the gated FFN pattern most current LLMs use in place of a
    plain two-layer MLP (SwiGLU/GeGLU: ``down(act(gate(x)) * up(x))``, see
    :func:`_find_gated_chains`) -- two producers (gate and up) combined by
    an elementwise product feed one consumer; both branches are ranked by
    combined (root-sum-square) importance and pruned to the *same*
    surviving channel indices, since they're about to be multiplied. This
    gated form is MatMul/Gemm-only -- Conv chains don't take part in it.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched producer's output
            channels to remove (at least one channel is always kept)
    :returns: ``model`` with every matched chain's tensors resized in
            place; anything not matching that exact topology (branching,
            a non-constant bias, a consumer whose reduction dimension
            doesn't line up, ...) is left completely untouched
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_chains(graph) + _find_gated_chains(graph) + _find_conv_chains(graph)
    if chains:
        _apply_chains(graph, chains, sparsity, _plain_structured_importance)

    return out


def apply_structured_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """The calibrated upgrade of :func:`apply_structured_pruning`, exactly
    as :func:`apply_wanda_pruning` is to :func:`apply_magnitude_pruning`:
    same real structural channel removal, same topology matching (a single
    producer or a gated pair -> zero or more shape-preserving elementwise
    ops and, for a Conv chain, depthwise Conv hops -> one consumer,
    MatMul/Gemm or Conv, see :func:`apply_structured_pruning`'s own
    docstring) including the same depthwise-Conv pass-through sliced by the
    producer's channel indices alone -- it contributes no activation norm
    of its own to the ranking either, being transparent to the chain's
    channel-index mapping just as it is to plain L2-norm importance -- but
    each chain's
    output channels are ranked by ``||W_row||_2 * ||X||_2`` -- L2 norm of
    that channel's own weight row (or, for Conv, whole filter), times the
    L2 norm of the *activation* actually flowing through that channel over
    calibration data (captured right where the chain feeds into its
    consumer, reduced over every axis but the channel one -- the last axis
    for a MatMul/Gemm consumer, axis 1 of ``[N, C, H, W]`` for a Conv
    consumer) -- instead of weight magnitude alone. This is the same
    protection Wanda's element-wise metric gives unstructured pruning,
    transplanted to whole channels: a channel whose weight is individually
    unremarkable but which gates a consistently high-magnitude activation
    is kept over one with a larger weight norm but a near-dead activation.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            chain's consumer-side activation norm on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched chain's output
            channels to remove (at least one channel is always kept)
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every channel of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched chain's tensors resized in
            place; anything not matching that exact topology falls back to
            :func:`apply_structured_pruning`'s plain L2-norm ranking if no
            matching activation was ever observed for that chain's consumer
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_chains(graph) + _find_gated_chains(graph) + _find_conv_chains(graph)
    if not chains:
        return out

    # The channel axis of the activation feeding each chain's consumer: a
    # MatMul/Gemm's reduction dimension is its input's last axis, while a
    # Conv's input channel dimension is always axis 1 of [N, C, H, W]. Two
    # chains can't disagree on a shared probe name -- a tensor has exactly
    # one producer node, so it feeds one consumer type.
    channel_axis: Dict[str, int] = {
        chain.consumer_node.input[0]: (1 if chain.consumer_is_conv else -1)
        for chain in chains
    }
    probe_names = sorted(channel_axis)
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            axis = channel_axis[name]
            axis = axis if axis >= 0 else x.ndim + axis
            if axis < 0 or axis >= x.ndim:
                continue
            # Sum of squares over every axis but the channel one -- correct
            # for any activation rank, not just the 2-D case.
            reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
            s = np.square(x).sum(axis=reduce_axes) if reduce_axes else np.square(x)
            cnt = int(np.prod(x.shape, dtype=np.int64)) // x.shape[axis]
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + cnt

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }

    def _wanda_structured_importance(
        chain: _Chain, w_arrays_nk: List[np.ndarray]
    ) -> np.ndarray:
        base = _plain_structured_importance(chain, w_arrays_nk)
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.n_channels:
            return base  # no matching activation observed -- fall back to |W|
        return base * np.maximum(norm, epsilon)

    _apply_chains(graph, chains, sparsity, _wanda_structured_importance)
    return out


# --- Attention-head pruning -----------------------------------------------

# The ``com.microsoft`` domain contrib op onnxsim's own `fuse_attention`
# optimizer pass (onnxsim/passes/fuse_attention.h) fuses a decomposed
# multi-head self-attention block into: a single merged QKV weight/bias
# ([hidden_size, Nq+Nk+Nv] / [Nq+Nk+Nv]) plus `num_heads`/`qkv_hidden_sizes`
# attributes. GroupQueryAttention (unequal Q/KV head counts, separate,
# un-merged Q/K/V weights -- see fuse_gqa.h) and the plain `ai.onnx`
# `Attention` op (opset 23+, a different schema) are both out of scope here,
# the same kind of narrower-than-general-case boundary Conv-group pruning
# above draws: pruning a *shared* KV head out from under some, but not all,
# of the query heads mapped to it needs real group-aware bookkeeping this
# function does not attempt.
_ATTENTION_DOMAIN = "com.microsoft"


@dataclass(frozen=True)
class _AttentionChain:
    node: onnx.NodeProto
    weight: str
    bias: Optional[str]
    num_heads: int
    nq: int
    nk: int
    nv: int
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool


def _match_attention_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int, int, int, int]]:
    """If `node` is a ``com.microsoft::Attention`` node with a constant 2-D
    float32 merged QKV weight ``[K, Nq+Nk+Nv]`` (and, if present, a
    constant 1-D float32 merged bias), returns
    ``(weight_name, bias_name_or_None, num_heads, Nq, Nk, Nv)``.
    """
    if node.domain != _ATTENTION_DOMAIN or node.op_type != "Attention":
        return None
    if len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 2
    ):
        return None
    total_n = w_init.dims[1]

    bias_name = None
    if len(node.input) >= 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if (
            b_init is None
            or b_init.data_type != onnx.TensorProto.FLOAT
            or list(b_init.dims) != [total_n]
        ):
            return None

    num_heads = None
    qkv_hidden_sizes: Optional[List[int]] = None
    for attr in node.attribute:
        if attr.name == "num_heads":
            num_heads = attr.i
        elif attr.name == "qkv_hidden_sizes":
            qkv_hidden_sizes = list(attr.ints)
    if not num_heads or num_heads <= 0:
        return None

    if qkv_hidden_sizes is not None:
        if len(qkv_hidden_sizes) != 3:
            return None
        nq, nk, nv = qkv_hidden_sizes
    else:
        # Schema default: Q/K/V evenly split the merged width.
        if total_n % 3 != 0:
            return None
        nq = nk = nv = total_n // 3
    if (
        nq <= 0
        or nk <= 0
        or nv <= 0
        or nq + nk + nv != total_n
        or nq % num_heads
        or nk % num_heads
        or nv % num_heads
    ):
        return None

    return w_name, bias_name, num_heads, nq, nk, nv


def _reshape_last_dim(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[int]:
    """If `node` is a ``Reshape`` whose target-shape input is a constant
    int64 tensor, returns its last entry (or ``None`` if that entry is a
    wildcard/inferred ``-1`` or ``0``, or the shape can't be read at all).
    """
    if node.op_type != "Reshape" or len(node.input) != 2:
        return None
    shape_init = initializer_map.get(node.input[1])
    if shape_init is None or shape_init.data_type != onnx.TensorProto.INT64:
        return None
    dims = onnx.numpy_helper.to_array(shape_init)
    if dims.size == 0:
        return None
    last = int(dims[-1])
    return last if last > 0 else None


def _walk_to_attention_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    nv: int,
) -> Tuple[Optional[_ConsumerMatch], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]:
    """From `Attention`'s raw (V-hidden-size-wide) output tensor `start`,
    optionally through a single ``Reshape`` hop whose target shape's last
    entry is provably still `nv` (the shape onnxsim's own `fuse_attention`
    pass always appends, reusing the original ``ctx`` reshape's own target
    -- see fuse_attention.h's own doc comment; a hand-authored or
    differently-sourced graph is still handled the same way as long as it
    matches this same shape), to a MatMul/vanilla-Gemm consumer (the output
    projection) whose reduction dimension matches `nv`. Declines (``None``)
    on anything else -- a branch, an activation, a mismatched Reshape --
    rather than guessing. When a Reshape hop is matched, its second (shape)
    input must be single-use too -- the caller overwrites that constant's
    last entry to the post-pruning `nv` in place, which would corrupt any
    other reader of the same tensor.
    """
    candidates = consumers_of.get(start, [])
    if len(candidates) != 1:
        return None, ()
    node = candidates[0]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...] = ()
    cur = start

    if node.op_type == "Reshape" and node.input[:1] == [cur]:
        last_dim = _reshape_last_dim(node, initializer_map)
        if last_dim != nv:
            return None, ()
        shape_name = node.input[1]
        if len(consumers_of.get(shape_name, [])) != 1:
            return None, ()  # shared shape constant -- mutating it isn't safe
        out_name = node.output[0]
        if len(consumers_of.get(out_name, [])) != 1 or out_name in graph_outputs:
            return None, ()
        chain_ops = ((node, shape_name),)
        cur = out_name
        node = consumers_of[cur][0]

    cm = _match_matmul_like(node)
    if cm is None or cm[0] != cur:
        return None, chain_ops
    _, cw_name, c_weight_transposed = cm
    cw_init = initializer_map.get(cw_name)
    if (
        cw_init is None
        or cw_init.data_type != onnx.TensorProto.FLOAT
        or len(cw_init.dims) != 2
    ):
        return None, chain_ops
    k = cw_init.dims[1] if c_weight_transposed else cw_init.dims[0]
    if k != nv:
        return None, chain_ops
    return (node, cw_name, c_weight_transposed), chain_ops


def _find_attention_chains(graph: onnx.GraphProto) -> List[_AttentionChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_attention_producer(node, initializer_map)
        if info is None:
            continue
        w_name, bias_name, num_heads, nq, nk, nv = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_attention_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, nv
        )
        if consumer is None:
            continue

        chains.append(
            _AttentionChain(
                node=node,
                weight=w_name,
                bias=bias_name,
                num_heads=num_heads,
                nq=nq,
                nk=nk,
                nv=nv,
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
            )
        )
    return chains


def _plain_attention_head_importance(
    chain: _AttentionChain,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    dq: int,
    dk: int,
    dv: int,
) -> np.ndarray:
    # Combined (Frobenius-norm) importance of each head's full Q+K+V
    # weight block -- the Li et al. filter-norm criterion this module uses
    # everywhere else, applied to a whole head's block of columns (across
    # every input row) at once instead of a single output channel/filter.
    importance = np.zeros(chain.num_heads, dtype=np.float64)
    for h in range(chain.num_heads):
        block = np.concatenate(
            [
                wq[:, h * dq : (h + 1) * dq],
                wk[:, h * dk : (h + 1) * dk],
                wv[:, h * dv : (h + 1) * dv],
            ],
            axis=1,
        )
        importance[h] = np.linalg.norm(block)
    return importance


def _head_column_indices(keep_heads: np.ndarray, head_size: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(h * head_size, (h + 1) * head_size) for h in keep_heads]
    )


def _apply_attention_chains(
    graph: onnx.GraphProto,
    chains: List[_AttentionChain],
    sparsity: float,
    compute_importance,
) -> None:
    """Shared body for :func:`apply_attention_head_pruning` and
    :func:`apply_attention_head_wanda_pruning`, mirroring
    :func:`_apply_chains`'s own shape (touched-role bookkeeping, keep-count
    computation, a ``compute_importance`` callback for the ranking) but at
    whole-head granularity: every dropped head removes a *contiguous*
    ``head_size``-wide column block from each of Q/K/V (and the matching
    row block from the consumer), not an arbitrary top-k column subset.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        if (
            chain.weight in producer_touched
            or chain.consumer_weight in consumer_touched
        ):
            continue

        h = chain.num_heads
        keep_count = max(1, h - round(h * sparsity))
        if keep_count >= h:
            continue

        dq, dk, dv = chain.nq // h, chain.nk // h, chain.nv // h
        w_init = initializer_map[chain.weight]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)  # [K, Nq+Nk+Nv]
        wq = w[:, : chain.nq]
        wk = w[:, chain.nq : chain.nq + chain.nk]
        wv = w[:, chain.nq + chain.nk :]

        importance = compute_importance(chain, wq, wk, wv, dq, dk, dv)
        keep_heads = np.sort(np.argsort(-importance)[:keep_count])

        q_idx = _head_column_indices(keep_heads, dq)
        k_idx = _head_column_indices(keep_heads, dk) + chain.nq
        v_idx_local = _head_column_indices(keep_heads, dv)
        v_idx = v_idx_local + chain.nq + chain.nk
        all_idx = np.concatenate([q_idx, k_idx, v_idx])

        w_arr = onnx.numpy_helper.to_array(w_init)
        w_init.CopyFrom(
            onnx.numpy_helper.from_array(w_arr[:, all_idx], name=w_init.name)
        )
        if chain.bias is not None:
            _slice_last_axis(initializer_map[chain.bias], all_idx)

        found_qkv = False
        for attr in chain.node.attribute:
            if attr.name == "num_heads":
                attr.i = keep_count
            elif attr.name == "qkv_hidden_sizes":
                found_qkv = True
                del attr.ints[:]
                attr.ints.extend([keep_count * dq, keep_count * dk, keep_count * dv])
        if not found_qkv:
            chain.node.attribute.append(
                onnx.helper.make_attribute(
                    "qkv_hidden_sizes",
                    [keep_count * dq, keep_count * dk, keep_count * dv],
                )
            )

        _slice_consumer_weight(
            initializer_map[chain.consumer_weight],
            chain.consumer_weight_transposed,
            v_idx_local,
        )

        for _, shape_name in chain.chain_ops:
            if shape_name is not None:
                shape_init = initializer_map[shape_name]
                dims = onnx.numpy_helper.to_array(shape_init).copy()
                dims[-1] = keep_count * dv
                shape_init.CopyFrom(
                    onnx.numpy_helper.from_array(dims, name=shape_init.name)
                )

        producer_touched.add(chain.weight)
        consumer_touched.add(chain.consumer_weight)
        stale_value_info.add(chain.node.output[0])
        stale_value_info.update(op.output[0] for op, _ in chain.chain_ops)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)


def apply_attention_head_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """Removes whole attention heads from every ``com.microsoft::Attention``
    node (the fused multi-head self-attention block onnxsim's own
    ``fuse_attention`` optimizer pass produces, see this module's own
    docstring) whose output feeds, optionally through a single shape-
    preserving ``Reshape``, exactly one downstream MatMul/vanilla-Gemm's
    reduction dimension (the output projection) -- the attention analogue
    of :func:`apply_structured_pruning`, at head instead of single-channel
    granularity.

    For each matched block: ranks every head by the combined Frobenius
    norm of its own ``[hidden_size, head_size]`` Q, K, and V weight
    columns, drops the lowest-``sparsity``-fraction of heads (at least one
    head is always kept), and removes the corresponding column blocks from
    the merged QKV weight (and bias, if present), decrementing
    ``num_heads``/``qkv_hidden_sizes`` accordingly, and the matching row
    block from the output projection's weight -- mathematically unaffected
    for every surviving head, the same guarantee
    :func:`apply_structured_pruning` gives per channel.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched block's heads to
            remove (at least one head is always kept)
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology (a
            non-constant weight, GroupQueryAttention's separate-weights
            shape, a consumer whose reduction dimension doesn't line up,
            ...) is left completely untouched
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_attention_chains(graph)
    if chains:
        _apply_attention_chains(
            graph, chains, sparsity, _plain_attention_head_importance
        )

    return out


def apply_attention_head_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """The calibrated upgrade of :func:`apply_attention_head_pruning`,
    exactly as :func:`apply_structured_wanda_pruning` is to
    :func:`apply_structured_pruning`: same real head removal, same
    topology matching, but each head's importance is
    ``||W_head||_F * ||X_head||_2`` -- the plain Frobenius-norm score times
    the combined (root-sum-square) activation norm of that head's own
    slice of the *output projection's* input, captured over calibration
    data -- instead of weight magnitude alone.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            block's output-projection-side activation norm on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched block's heads to
            remove (at least one head is always kept)
    :param epsilon: floor applied to the accumulated per-head activation
            norm, avoiding every head of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology falls back to
            :func:`apply_attention_head_pruning`'s plain Frobenius-norm
            ranking if no matching activation was ever observed for that
            block's consumer
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_attention_chains(graph)
    if not chains:
        return out

    probe_names = sorted({chain.consumer_node.input[0] for chain in chains})
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 1:
                continue
            reduce_axes = tuple(range(x.ndim - 1))
            s = np.square(x).sum(axis=reduce_axes) if reduce_axes else np.square(x)
            cnt = int(np.prod(x.shape[:-1], dtype=np.int64)) if x.ndim > 1 else 1
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + cnt

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }

    def _wanda_attention_head_importance(chain, wq, wk, wv, dq, dk, dv):
        base = _plain_attention_head_importance(chain, wq, wk, wv, dq, dk, dv)
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.nv:
            return base  # no matching activation observed -- fall back to plain
        act_head = np.array(
            [
                np.linalg.norm(norm[h * dv : (h + 1) * dv])
                for h in range(chain.num_heads)
            ]
        )
        return base * np.maximum(act_head, epsilon)

    _apply_attention_chains(graph, chains, sparsity, _wanda_attention_head_importance)
    return out
