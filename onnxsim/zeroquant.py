"""ZeroQuant (Yao, Aminabadi, Zhang, Wu, Li, He, 2022, "ZeroQuant: Efficient
and Affordable Post-Training Quantization for Large-Scale Transformers",
https://arxiv.org/abs/2206.01861). onnxsim ports the algorithm, not any
framework's code, per the same rationale as :mod:`onnxsim.quarot`/
:mod:`onnxsim.duquant`/:mod:`onnxsim.quip_sharp`.

**What's already in onnxsim, and what ZeroQuant actually adds.** This
repository already has both halves of ZeroQuant's quantization scheme in
isolation, each ported for its own reasons well before this module existed:

- **Group-wise INT8 weight quantization** -- one symmetric scale per
  ``(K-group, output channel)`` instead of one scale per whole tensor or
  per output channel only -- is exactly
  :func:`onnxsim.quantize_weight_only_int8_block` (see
  ``docs/int8-block-quantization.md``). That pass already reproduces
  ZeroQuant's weight side faithfully; this module does not reimplement it.
- **Per-token dynamic INT8 activation quantization** -- a fresh
  ``scale = max(|x|, axis=-1) / 127`` computed from each token's own row,
  at graph-run time, no calibration data -- is a pattern this repo already
  uses repeatedly (:mod:`onnxsim.quarot`, :mod:`onnxsim.duquant`,
  :mod:`onnxsim.attention_quantization`, and
  :mod:`onnxsim.kv_cache_quantization`'s Value-style rewrite). **But** every
  one of those existing uses immediately dequantizes back to float32 right
  after quantizing (a round-trip that models the *precision loss* of
  quantizing, for INT4 weight/activation schemes that keep the actual
  matmul running in float) -- none of them feed the quantized activation
  into a true integer ``MatMulInteger``. And onnxsim's one existing
  *integer*-executing activation path, :func:`onnxsim.quantize_dynamic`
  (``onnxsim/passes/dynamic_quantize_matmul.h``), uses standard ONNX
  ``DynamicQuantizeLinear`` -- which computes **one scale for the entire
  input tensor**, not one per row/token, despite "dynamic" in the name.
  So genuine per-token dynamic quantization feeding a *real* integer matmul
  does not exist anywhere in onnxsim yet.

ZeroQuant's real, non-redundant contribution here is therefore not either
piece alone -- it is **pairing them**: group-wise INT8 weight quantization
*and* per-token dynamic INT8 activation quantization, applied *together* to
the same layer, executed as genuine ``int8 x int8`` integer matmuls (not
simulated in float), because that specific fine-grained combination is what
the paper identifies as the hardware-friendly sweet spot for W8A8
transformer inference -- coarser than this (per-tensor activation, as
:func:`onnxsim.quantize_dynamic` does; or per-output-channel-only weight, as
:func:`onnxsim.quantize_weight_only` does) loses more accuracy than
necessary at the same INT8 bit width, while finer approaches (calibrated
per-channel activation ranges, or the paper's own separate INT4-weight/INT8-
activation variant) need either calibration data or asymmetric bit widths
this module does not add.

    Before:
      Y = MatMul(X, W) [+ bias]      -- W constant, [K, N], float32

    After (conceptually; see "Why grouped MatMulInteger" for the actual
    node-level construction):
      Xq = round_to_nearest_int8_per_token(X)          -- computed at runtime
      Wq, Ws = per_group_symmetric_int8(W)             -- computed once, here
      Acc = sum over K-groups g of MatMulInteger(Xq[:, group g], Wq[group g])
      Y = Acc * Xscale * Ws [+ bias]                    -- dequantize

**Why grouped ``MatMulInteger`` instead of one call, and why the activation
is symmetric.** Standard ONNX's ``MatMulInteger`` schema documents a
per-row zero point on ``A`` and a per-column zero point on ``B`` -- which
would appear to be exactly what a per-token activation scale and a
per-group weight scale need, in one call. Two separate obstacles rule that
out:

1. A single ``MatMulInteger`` call always contracts the *entire* ``K``
   dimension into one integer accumulator, so a weight scale that varies
   partway through ``K`` (this module's whole point) cannot be applied
   after the fact -- the different groups' products are already summed
   together by the time the op returns. This module therefore slices both
   the quantized activation and the quantized weight into
   ``block_size``-wide groups along ``K`` and runs one real
   ``MatMulInteger`` per group, combining the groups' dequantized partial
   sums in float32 afterward -- more nodes per layer than any single-call
   rewrite elsewhere in this codebase, but the only way to get a real (not
   simulated) integer matmul out of standard ONNX ops for this specific
   combination of granularities. Each group's own accumulation is small
   enough (``block_size`` terms) that int32 overflow is a non-issue in
   practice (see ``_MAX_SAFE_GROUP_SIZE`` below), unlike
   :func:`onnxsim.quantize_dynamic`'s own accumulator-overflow guard on the
   full, ungrouped reduction depth.
2. Empirically (checked directly against ``onnxruntime`` while building
   this module, not assumed from the spec text alone): ONNX Runtime's own
   CPU ``MatMulInteger`` kernel rejects a genuine per-row ``a_zero_point``
   at run time (``IsScalarOr1ElementVector(a_zero_point) was false``) even
   though the ONNX operator *schema* documents that shape as valid -- the
   schema's per-row zero point is, in practice, unimplemented on the one
   execution provider onnxsim's own tests (and every other onnxsim
   quantizer's tests) run against. So this module quantizes the activation
   **symmetrically** (``scale = max(|x|, axis=-1) / 127``, zero point
   always exactly ``0``) instead of the asymmetric,
   ``DynamicQuantizeLinear``-style scheme its own per-token *scale*
   formula would otherwise suggest -- the same symmetric convention every
   other per-token dynamic scale in this repo already uses
   (:mod:`onnxsim.quarot`, :mod:`onnxsim.duquant`,
   :mod:`onnxsim.attention_quantization`,
   :mod:`onnxsim.kv_cache_quantization`'s Value-style rewrite), just now
   feeding a real ``MatMulInteger`` instead of a simulated round-trip. With
   zero point fixed at a compile-time constant ``0`` (never a per-row
   tensor), ``a_zero_point``/``b_zero_point`` are omitted entirely (their
   documented default), which sidesteps the unimplemented shape
   completely -- confirmed directly with a standalone ``int8 x int8``
   ``MatMulInteger`` model run through ``onnxruntime.InferenceSession``
   during development. Only the *scale* varies per token; that multiply
   happens entirely outside ``MatMulInteger`` (in this module's own ``Mul``
   node, against the dequantized float accumulator), so per-token
   granularity is fully preserved -- it is only the zero point that had to
   give.

**Explicitly out of scope: layer-by-layer knowledge distillation (LKD).**
The ZeroQuant paper's other contribution is a training loop that
compensates deeper-layer quantization error by distilling each quantized
layer against its own original-precision output, one layer at a time. That
needs a training loop over a framework-native model with gradient
computation -- fundamentally incompatible with onnxsim's whole
architecture of stateless graph rewriting on an existing ONNX protobuf, the
same boundary ``docs/nncf-comparison-future-work.md``'s own "Explicitly out
of scope" section draws for quantization-aware training generally. LKD is
not reproduced here, nor anywhere else in onnxsim.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.quip_sharp import _match_matmul_like

# int32 accumulation over a single K-group can't overflow until
# block_size * 255 (uint8 activation range) * 127 (int8 weight range)
# exceeds INT32_MAX -- about 66,311. Mirrors
# passes/quantize_matmul_common.h's IsSafeInt32ReductionDepth, applied per
# group instead of over the whole (ungrouped) reduction depth.
_MAX_SAFE_GROUP_SIZE = (2**31 - 1) // (255 * 127)


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _quantize_weight_groupwise_int8(w_kn: np.ndarray, block_size: int, epsilon: float):
    """Symmetric INT8 quantization of a ``[K, N]`` weight, one scale per
    ``(block_size``-wide K-group, output column``)`` -- the same granularity
    :func:`onnxsim.quantize_weight_only_int8_block` uses. Returns
    ``(wq_kn int8 [K, N], scale_gn float32 [K // block_size, N])``. Caller
    must ensure ``K % block_size == 0``.
    """
    k, n = w_kn.shape
    num_groups = k // block_size
    blocks = w_kn.reshape(num_groups, block_size, n).astype(np.float64)
    scale = np.max(np.abs(blocks), axis=1) / 127.0  # [num_groups, N]
    scale = np.maximum(scale, epsilon)
    wq = np.clip(np.round(blocks / scale[:, np.newaxis, :]), -127, 127)
    wq = wq.reshape(k, n).astype(np.int8)
    return wq, scale.astype(np.float32)


def apply_zeroquant(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """Applies ZeroQuant-style W8A8 quantization -- group-wise INT8 weight
    quantization paired with per-token dynamic INT8 activation
    quantization, executed as real ``int8 x int8`` integer matmuls -- to
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight
    whose reduction dimension ``K`` is divisible by ``block_size``. See
    this module's own docstring for exactly what's reused from existing
    onnxsim passes vs. novel here. Needs no calibration data: the weight's
    per-group scales come from the weight's own static values, and the
    activation's per-token scale is computed fresh at graph-run time from
    that token's own values, symmetrically (``scale = max(|x|) / 127``,
    zero point fixed at ``0`` -- see this module's own docstring, point 2,
    for why the activation is symmetric rather than following
    ``DynamicQuantizeLinear``'s asymmetric convention).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per weight quantization group along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int8_block`'s own
            default. Must divide ``K`` evenly for a layer to be quantized,
            and must not exceed ``_MAX_SAFE_GROUP_SIZE`` (~66,311) -- the
            point past which a single group's own int32
            ``MatMulInteger`` accumulator could overflow in the worst case.
    :param epsilon: floor applied to a weight group's own max-abs value
            (and, at graph-run time, a token's own quantization range)
            before using it as a scale, avoiding a divide-by-zero on an
            all-zero group or token
    :returns: ``model`` with every matched layer's weight and activation
            replaced by a group-wise/per-token INT8 ``MatMulInteger``
            pipeline (plus the original bias, if any); output tensor name
            unchanged. Layers with a non-constant, non-2-D weight, or a
            reduction dimension not divisible by ``block_size``, are left
            untouched. A model whose opset is older than 18 (this module's
            per-token activation scale needs ``ReduceMax``'s
            ``axes``-as-input form, and equal-sized ``Split`` needs its
            ``num_outputs`` attribute -- both opset 18), or with
            ``block_size`` not a positive divisor of some matched layer's
            ``K`` at all, is returned with that layer (or, if no layer
            qualifies and/or the opset gate fails, the whole model)
            unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if block_size <= 0 or block_size > _MAX_SAFE_GROUP_SIZE:
        return model
    if not _has_min_opset(model, 18):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    candidates = []
    for node in graph.node:
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
        w_shape = w_init.dims
        k = w_shape[1] if weight_transposed else w_shape[0]
        if k % block_size != 0:
            continue
        candidates.append((node, x_name, w_name, bias_name, weight_transposed))

    if not candidates:
        return out

    minus_one_name = _unique_name("zq_minus_one", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([-1], dtype=np.int64), name=minus_one_name
        )
    )
    last_axis_name = _unique_name("zq_last_axis", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([-1], dtype=np.int64), name=last_axis_name
        )
    )
    i127_name = _unique_name("zq_127", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(127.0, dtype=np.float32), name=i127_name)
    )
    i127_min_name = _unique_name("zq_neg127", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(-127.0, dtype=np.float32), name=i127_min_name
        )
    )
    eps_name = _unique_name("zq_eps", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
    )
    zero_1d_name = _unique_name("zq_zero_1d", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name=zero_1d_name)
    )

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_kn = w.T if weight_transposed else w  # [K, N]
        k, n = w_kn.shape
        if k % block_size != 0:
            continue
        num_groups = k // block_size

        wq_kn, scale_gn = _quantize_weight_groupwise_int8(w_kn, block_size, epsilon)

        prefix = f"{w_name}_zeroquant"
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

        # --- Flatten X to 2-D [M, K] (M = every leading dim collapsed) so
        # the per-token quantization below always sees a plain matrix
        # regardless of X's original rank (e.g. [batch, seq, K]).
        orig_shape = _new("Shape", [x_name], "orig_shape")
        k_dim = _new("Gather", [orig_shape, last_axis_name], "k_dim", axis=0)
        flat_shape = _new("Concat", [minus_one_name, k_dim], "flat_shape", axis=0)
        x2d = _new("Reshape", [x_name, flat_shape], "x2d")

        # --- Per-token (per-row) dynamic INT8 activation quantization,
        # symmetric (scale = max(|x|) / 127, zero point fixed at 0) -- see
        # this module's own docstring, point 2, for why symmetric rather
        # than DynamicQuantizeLinear's asymmetric convention. No
        # calibration data: each token's own scale is computed fresh here,
        # at graph-run time, from that token's own values.
        x_abs = _new("Abs", [x2d], "x_abs")
        x_max_abs = _new("ReduceMax", [x_abs, last_axis_name], "x_max_abs", keepdims=1)
        x_safe_max_abs = _new("Max", [x_max_abs, eps_name], "x_safe_max_abs")
        x_scale = _new("Div", [x_safe_max_abs, i127_name], "x_scale")  # [M, 1]
        x_scaled = _new("Div", [x2d, x_scale], "x_scaled")
        x_rounded = _new("Round", [x_scaled], "x_rounded")
        x_clipped = _new("Clip", [x_rounded, i127_min_name, i127_name], "x_clipped")
        xq_2d = _new("Cast", [x_clipped], "xq_2d", to=onnx.TensorProto.INT8)

        # --- Group-wise INT8 weight (computed once, here, from W's static
        # values) and one real MatMulInteger per K-group.
        xq_groups = [
            _unique_name(f"{prefix}_xq_group{g}", taken_names)
            for g in range(num_groups)
        ]
        split_node = onnx.helper.make_node(
            "Split",
            [xq_2d],
            xq_groups,
            name=_unique_name(f"{prefix}_split_node", taken_names),
            axis=1,
            num_outputs=num_groups,
        )
        new_nodes.append(split_node)

        group_terms = []
        for g in range(num_groups):
            wq_g_name = _unique_name(f"{prefix}_wq_group{g}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.ascontiguousarray(wq_kn[g * block_size : (g + 1) * block_size]),
                    name=wq_g_name,
                )
            )
            ws_g_name = _unique_name(f"{prefix}_ws_group{g}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(scale_gn[g], name=ws_g_name)
            )

            # a_zero_point/b_zero_point both omitted (default 0): both
            # operands are quantized symmetrically, and ONNX Runtime's own
            # MatMulInteger CPU kernel rejects a genuine per-row zero point
            # anyway (see this module's own docstring, point 2).
            acc_g = _new("MatMulInteger", [xq_groups[g], wq_g_name], f"acc_group{g}")
            acc_g_f = _new(
                "Cast", [acc_g], f"acc_group{g}_f", to=onnx.TensorProto.FLOAT
            )
            scaled_g = _new("Mul", [acc_g_f, ws_g_name], f"scaled_group{g}")
            group_terms.append(scaled_g)

        unscaled_sum = (
            group_terms[0]
            if num_groups == 1
            else _new("Sum", group_terms, "unscaled_sum")
        )
        y2d_unbiased = _new("Mul", [unscaled_sum, x_scale], "y2d_unbiased")

        if bias_name is not None:
            y2d = _new("Add", [y2d_unbiased, bias_name], "y2d")
        else:
            y2d = y2d_unbiased

        # --- Restore the original leading (batch/sequence) dims, with the
        # last dim now N instead of K.
        n_const_name = _unique_name(f"{prefix}_n_const", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([n], dtype=np.int64), name=n_const_name
            )
        )
        lead_shape = _new(
            "Slice", [orig_shape, zero_1d_name, last_axis_name], "lead_shape"
        )
        out_shape = _new("Concat", [lead_shape, n_const_name], "out_shape", axis=0)

        old_output = node.output[0]
        final = onnx.helper.make_node(
            "Reshape",
            [y2d, out_shape],
            [old_output],
            name=_unique_name(f"{prefix}_reshape_out_node", taken_names),
        )
        new_nodes.append(final)

        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
