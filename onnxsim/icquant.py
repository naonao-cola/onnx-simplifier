"""ICQuant (Li, Hanna, Fragouli, Diggavi, 2025, "ICQuant: Index Coding
enables Low-bit LLM Quantization", https://arxiv.org/abs/2505.00850, COLM
2025; code at https://github.com/Avery-xl/ICQuant). onnxsim ports the
algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.spinquant` (ICQuant's
own reference implementation quantizes live PyTorch weights, with no ONNX
export path).

Like :mod:`onnxsim.owq` and :mod:`onnxsim.spqr`, this module separates a
small fraction of "salient" weight elements from the rest of each
quantization group so the group's own scale is no longer inflated by them,
then quantizes the remaining elements to a tighter grid. **ICQuant's own
distinguishing contribution is not a new way to decide which elements are
outliers** -- this module reuses the same ordinary per-group top-magnitude
criterion :mod:`onnxsim.spqr` already uses (any of this repo's other
magnitude/Hessian-based salience criteria would work just as well here).
ICQuant's contribution is a **cheaper way to communicate which positions
were chosen** once decided:

* :mod:`onnxsim.owq` stores an explicit list of the chosen column indices
  (``k * ceil(log2(K))`` bits for ``k`` chosen columns out of ``K``).
* :mod:`onnxsim.spqr` stores an explicit ``[row, col]`` pair per outlier
  (again a plain index list, ``num_outliers * (ceil(log2(N)) +
  ceil(log2(K)))`` bits).

Both are, information-theoretically, using far more bits than necessary: a
naive **bitmask** of ``n`` bits (one per group element, marking outlier or
not) already overspends -- it can represent ``2^n`` distinct
outlier/non-outlier patterns when there are only ``C(n, k)`` patterns with
exactly ``k`` outliers, and a plain index list overspends similarly by
picking an ordering-sensitive encoding. ICQuant's own idea, borrowed from
**index coding** in information theory, is to instead number the ``C(n,
k)`` possible ``k``-of-``n`` outlier subsets ``0, 1, ..., C(n, k) - 1`` via
the classical **combinatorial number system** ("combinadics" -- see
:func:`_combinadic_rank`/:func:`_combinadic_unrank` below) and store a
single integer rank in that range: ``ceil(log2(C(n, k)))`` bits, versus a
naive scheme's ``n`` bits (bitmask) or ``k * ceil(log2(n))`` bits (index
list). For the paper's own typical setting -- groups of ``n = 32`` with
``k = 1`` or ``2`` outliers -- this is roughly 5 or 9 bits per group
(``0.16`` or ``0.28`` bits/element) versus a naive bitmask's 32 bits/group
(``1.0`` bit/element): see :func:`icquant_metadata_bits` and this module's
own tests for the exact numbers, matching the paper's reported ~0.3
bits/element overhead to shrink the quantization range the same way a
naive ~1-bit/element scheme would.

**What this module actually emits.** ONNX has no combinatorial-decoding
op, and none of onnxsim's other quantizers decode anything at runtime
either -- they all bake plain, already-resolved indices into the graph as
ordinary initializers. So here too: at quantization time (in Python), each
group's chosen outlier positions are combinadic-encoded into one rank
integer *and immediately decoded back* via :func:`_combinadic_unrank`
(exercising the actual encode/decode round trip this module's docstring
claims, not just asserting it), and it is that decoded, explicit index
list which is baked into the emitted graph's ``ScatterND`` indices --
exactly the same graph shape :mod:`onnxsim.spqr` already uses to
reconstruct its own sparse correction:

    Before:
      Y = MatMul(X, W) [+ bias]                  -- W constant, [K, N], float32

    After:
      Wq  = <int4, per-(group, row) symmetric, outlier positions excluded
             from each group's own scale>
      Ws  = <float32, [K/group_size, N]>
      Wdq = DequantizeLinear(Wq, Ws, axis=0, block_size=group_size)  -- float32
      Wreconstructed = ScatterND(Wdq, outlier_indices, outlier_values)
      Y = MatMul(X, Wreconstructed) [+ bias]

Unlike :mod:`onnxsim.spqr` (which ``Add``s an exact *residual* on top of
the dequantized value, because SpQR's outliers can be anywhere in the
matrix and its correction must cancel whatever the block-quantized value
there happened to round to), ICQuant's outlier value is written directly
in full precision at each chosen position, so a plain overwrite
(``ScatterND``'s default, non-accumulating semantics) already reconstructs
it exactly -- no residual arithmetic needed.

Deliberately not ported: ICQuant's own incoherence-processing preprocessing
step (a random orthogonal/Hadamard rotation applied before outlier
selection, in the same family as :mod:`onnxsim.quip_sharp`'s own
incoherence processing) -- out of scope here so this module can isolate
and demonstrate ICQuant's own distinguishing contribution (the index-coding
metadata scheme) cleanly on top of plain magnitude-based outlier selection,
the same scope boundary :mod:`onnxsim.owq`/:mod:`onnxsim.spqr` draw around
their own outlier-selection criteria.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _combinadic_rank(combo: Sequence[int], n: int) -> int:
    """Ranks a ``k``-element subset of ``{0, ..., n - 1}`` to its unique
    integer in ``[0, C(n, k))`` via the combinatorial number system: for
    ``combo`` sorted descending ``c_k > c_{k-1} > ... > c_1``, the rank is
    ``sum_i C(c_i, i)`` (1-indexed from the top element down to 1). This is
    the standard "combinadic" bijection between ``k``-of-``n`` subsets and
    ``[0, C(n, k))`` -- see :func:`_combinadic_unrank` for its inverse.

    :param combo: the chosen element indices (any order, each in
            ``[0, n)``, no duplicates)
    :param n: size of the ground set the subset was chosen from
    :returns: the subset's unique rank in ``[0, C(n, len(combo)))``
    """
    k = len(combo)
    combo_desc = sorted(combo, reverse=True)
    return sum(math.comb(c, k - i) for i, c in enumerate(combo_desc))


def _combinadic_unrank(rank: int, k: int, n: int) -> List[int]:
    """Inverse of :func:`_combinadic_rank`: recovers the ``k``-element
    subset of ``{0, ..., n - 1}`` with the given combinadic ``rank``, via
    the classical greedy digit-by-digit decomposition (find the largest
    ``c`` with ``C(c, i) <= remaining_rank``, for ``i`` from ``k`` down to
    ``1``).

    :param rank: a combinadic rank in ``[0, C(n, k))``
    :param k: subset size
    :param n: size of the ground set
    :returns: the subset, as a list of ``k`` indices sorted ascending
    """
    combo = []
    remaining = rank
    for i in range(k, 0, -1):
        c = i - 1
        while math.comb(c + 1, i) <= remaining:
            c += 1
        combo.append(c)
        remaining -= math.comb(c, i)
    return sorted(combo)


def icquant_metadata_bits(group_size: int, num_outliers: int) -> dict:
    """Compares the per-group metadata cost of ICQuant's combinadic
    encoding against the two naive alternatives it replaces (see this
    module's own docstring), for a group of ``group_size`` elements with
    exactly ``num_outliers`` outliers.

    :returns: a dict with ``combinadic_bits`` (``ceil(log2(C(group_size,
            num_outliers)))``, or ``0`` when ``num_outliers`` is 0),
            ``bitmask_bits`` (``group_size``), ``index_list_bits``
            (``num_outliers * ceil(log2(group_size))``), and each scheme's
            per-element overhead (the above divided by ``group_size``)
            under ``*_bits_per_element`` keys
    """
    if num_outliers <= 0:
        combinadic_bits = 0
    else:
        combinadic_bits = max(
            1, math.ceil(math.log2(math.comb(group_size, num_outliers)))
        )
    bitmask_bits = group_size
    index_list_bits = (
        num_outliers * max(1, math.ceil(math.log2(group_size)))
        if num_outliers > 0
        else 0
    )
    return {
        "combinadic_bits": combinadic_bits,
        "bitmask_bits": bitmask_bits,
        "index_list_bits": index_list_bits,
        "combinadic_bits_per_element": combinadic_bits / group_size,
        "bitmask_bits_per_element": bitmask_bits / group_size,
        "index_list_bits_per_element": index_list_bits / group_size,
    }


def quantize_weight_only_icquant(
    model: Union[str, onnx.ModelProto],
    group_size: int = 32,
    num_outliers: int = 1,
) -> onnx.ModelProto:
    """Applies ICQuant-style outlier-aware block-wise INT4 quantization
    (see this module's own docstring) to every MatMul/vanilla-Gemm layer
    with a constant 2-D float32 weight whose reduction dimension ``K`` is
    divisible by ``group_size``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param group_size: elements per quantization group along ``K``,
            matching :func:`onnxsim.quantize_weight_only_spqr`'s own
            ``block_size`` granularity -- the paper's own typical choice is
            32
    :param num_outliers: number of largest-magnitude elements excluded
            from each group's own scale computation and stored at full
            precision instead, communicated via one combinadic rank per
            group (see this module's own docstring); the paper's own
            typical choice is 1 or 2. ``0`` degenerates to plain
            group-wise INT4 quantization with no outlier handling.
    :returns: ``model`` with every matched layer's weight replaced by
            group-wise INT4 codes plus exact per-group outlier values
            reconstructed via ``ScatterND`` (see the module docstring's
            diagram); output tensor name unchanged. Layers with a
            non-constant, non-2-D weight, a reduction dimension not
            divisible by ``group_size``, or ``num_outliers >=
            group_size``, are left untouched; a model with no matching
            layer, or an opset older than 21 (INT4's tensor type and
            ``DequantizeLinear``'s ``block_size`` attribute both need
            opset 21), is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21):
        return model
    if num_outliers < 0:
        raise ValueError("num_outliers must be >= 0")

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    nodes = list(graph.node)
    candidates = []
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, bias_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, bias_name, weight_transposed))

    if not candidates:
        return out

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n_rows, k = w_nk.shape
        if k % group_size != 0 or num_outliers >= group_size or n_rows == 0 or k == 0:
            continue

        num_blocks = k // group_size
        blocks = w_nk.reshape(n_rows, num_blocks, group_size)
        abs_blocks = np.abs(blocks)
        mask = np.ones((n_rows, num_blocks, group_size), dtype=bool)

        outlier_rows: List[int] = []
        outlier_cols: List[int] = []
        outlier_values: List[float] = []

        if num_outliers > 0:
            # Vectorized top-`num_outliers` selection per group, then a
            # per-group combinadic encode/decode round trip -- see this
            # module's own docstring for why the decode step (not just the
            # rank) is what actually gets baked into the graph below.
            top_unsorted = np.argpartition(-abs_blocks, num_outliers - 1, axis=2)[
                :, :, :num_outliers
            ]
            for r in range(n_rows):
                for b in range(num_blocks):
                    combo = sorted(int(i) for i in top_unsorted[r, b])
                    rank = _combinadic_rank(combo, group_size)
                    decoded = _combinadic_unrank(rank, num_outliers, group_size)
                    assert decoded == combo
                    for pos in decoded:
                        mask[r, b, pos] = False
                        outlier_rows.append(r)
                        outlier_cols.append(b * group_size + pos)
                        outlier_values.append(float(blocks[r, b, pos]))

        abs_masked = np.where(mask, abs_blocks, 0.0)
        scale_blocks = np.maximum(abs_masked.max(axis=2), 1e-12) / 7.0
        scale_full = np.repeat(scale_blocks, group_size, axis=1)  # [N, K]

        codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)

        prefix = f"{w_name}_icquant"
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N]
        scale_kn = scale_blocks.T.astype(np.float32)  # [K/group_size, N]

        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        codes_tensor = onnx.TensorProto()
        codes_tensor.name = codes_name
        codes_tensor.data_type = onnx.TensorProto.INT4
        codes_tensor.dims.extend([k, n_rows])
        codes_tensor.raw_data = _pack_int4(codes_kn)
        graph.initializer.append(codes_tensor)

        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_kn, name=scale_name)
        )

        new_nodes: List[onnx.NodeProto] = []

        def _new(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"{prefix}_{out_suffix}", taken_names)
            n_ = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{out_suffix}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n_)
            return out_name

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=group_size,
        )

        if outlier_values:
            # [K, N]-layout indices, matching codes_kn/scale_kn's own
            # transposed storage: index[i] = [k_pos, n_pos].
            outlier_indices_kn = np.stack(
                [np.asarray(outlier_cols), np.asarray(outlier_rows)], axis=1
            )

            indices_name = _unique_name(f"{prefix}_outlier_indices", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    outlier_indices_kn.astype(np.int64), name=indices_name
                )
            )
            values_name = _unique_name(f"{prefix}_outlier_values", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(outlier_values, dtype=np.float32), name=values_name
                )
            )
            w_reconstructed = _new(
                "ScatterND",
                [w_dequant, indices_name, values_name],
                "w_reconstructed",
            )
        else:
            w_reconstructed = w_dequant

        core = _new("MatMul", [x_name, w_reconstructed], "core")

        old_output = node.output[0]
        if bias_name is not None:
            final = onnx.helper.make_node(
                "Add",
                [core, bias_name],
                [old_output],
                name=_unique_name(f"{prefix}_bias_add_node", taken_names),
            )
        else:
            final = onnx.helper.make_node(
                "Identity",
                [core],
                [old_output],
                name=_unique_name(f"{prefix}_identity_node", taken_names),
            )
        new_nodes.append(final)

        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
