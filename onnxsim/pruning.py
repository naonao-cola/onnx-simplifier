"""Post-training weight pruning for MatMul/vanilla-Gemm layers.

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
whose output feeds, through a chain of shape-preserving elementwise ops
(activations, a bias/scale add/mul) with no other consumer anywhere along
that chain, into exactly one downstream MatMul/Gemm's reduction dimension.
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
Filters for Efficient ConvNets", https://arxiv.org/abs/1608.08710, adapted
here from Conv filters to MatMul/Gemm output channels -- the same
transplant :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`
already made for Han et al./Wanda's element-wise criteria). Because this
changes shapes, it is unconditionally irreversible and, unlike a retrained
pipeline, has no distillation/RL step to recover whatever accuracy the cut
costs -- evaluate the result before shipping it, the same caution any
lossy onnxsim pass deserves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
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


@dataclass(frozen=True)
class _Chain:
    producer: onnx.NodeProto
    producer_weight: str
    producer_weight_transposed: bool
    producer_bias: Optional[str]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_weight: str
    consumer_weight_transposed: bool
    n_channels: int


def _consumers_of(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


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
        match = _match_matmul_like(node)
        if match is None:
            continue
        _, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        bias_name = None
        if node.op_type == "Gemm" and len(node.input) == 3:
            bias_name = node.input[2]
            if bias_name not in initializer_map:
                continue  # non-constant bias -- can't safely prune it

        n_channels = w_init.dims[0] if weight_transposed else w_init.dims[1]

        chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
        consumer = None
        cur = out_name
        for _hop in range(_MAX_CHAIN_HOPS):
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
            if not _is_internal(out2):
                break
            chain_ops.append((nxt, const_name))
            cur = out2

        if consumer is None:
            continue

        chains.append(
            _Chain(
                producer=node,
                producer_weight=w_name,
                producer_weight_transposed=weight_transposed,
                producer_bias=bias_name,
                chain_ops=tuple(chain_ops),
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_channels,
            )
        )
    return chains


def _slice_producer_weight(
    w_init: onnx.TensorProto, weight_transposed: bool, keep: np.ndarray
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    # [N, K] storage (transB=1): output channel is axis 0. [K, N] storage
    # (the common case): output channel is axis 1.
    w_new = w[keep, :] if weight_transposed else w[:, keep]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_consumer_weight(
    w_init: onnx.TensorProto, weight_transposed: bool, keep: np.ndarray
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    # [N, K] storage (transB=1): reduction dim is axis 1. [K, N] storage:
    # reduction dim is axis 0.
    w_new = w[:, keep] if weight_transposed else w[keep, :]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_last_axis(init: onnx.TensorProto, keep: np.ndarray) -> None:
    arr = onnx.numpy_helper.to_array(init)
    new = np.take(arr, keep, axis=-1)
    init.CopyFrom(onnx.numpy_helper.from_array(new, name=init.name))


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
    pruning.

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

    chains = _find_chains(graph)
    if not chains:
        return out

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
    stale_value_info: Set[str] = set()

    for chain in chains:
        consts = {chain.producer_bias} if chain.producer_bias is not None else set()
        consts.update(
            const_name for _, const_name in chain.chain_ops if const_name is not None
        )
        if (
            chain.producer_weight in producer_touched
            or chain.consumer_weight in consumer_touched
            or (consts & const_touched)
        ):
            continue  # a shared/tied initializer another chain already resized

        n = chain.n_channels
        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        w_init = initializer_map[chain.producer_weight]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if chain.producer_weight_transposed else w.T  # [N, K]
        importance = np.linalg.norm(w_nk, axis=1)
        keep = np.sort(np.argsort(-importance)[:keep_count])

        _slice_producer_weight(w_init, chain.producer_weight_transposed, keep)
        if chain.producer_bias is not None:
            _slice_last_axis(initializer_map[chain.producer_bias], keep)
        for _, const_name in chain.chain_ops:
            if const_name is not None:
                _slice_last_axis(initializer_map[const_name], keep)
        _slice_consumer_weight(
            initializer_map[chain.consumer_weight],
            chain.consumer_weight_transposed,
            keep,
        )

        producer_touched.add(chain.producer_weight)
        consumer_touched.add(chain.consumer_weight)
        const_touched.update(consts)
        stale_value_info.add(chain.producer.output[0])
        stale_value_info.update(node.output[0] for node, _ in chain.chain_ops)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)

    return out
