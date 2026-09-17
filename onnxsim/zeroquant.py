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

from typing import Union

import numpy as np
import onnx

from onnxsim.onnx_simplifier import apply_zeroquant_cpp

# int32 accumulation over a single K-group can't overflow until
# block_size * 255 (uint8 activation range) * 127 (int8 weight range)
# exceeds INT32_MAX -- about 66,311. Mirrors
# passes/quantize_matmul_common.h's IsSafeInt32ReductionDepth, applied per
# group instead of over the whole (ungrouped) reduction depth. Still used
# by tests/test_zeroquant_cpp.py to exercise the C++ port's own block_size
# upper bound.
_MAX_SAFE_GROUP_SIZE = (2**31 - 1) // (255 * 127)


def _quantize_weight_groupwise_int8(w_kn: np.ndarray, block_size: int, epsilon: float):
    """Symmetric INT8 quantization of a ``[K, N]`` weight, one scale per
    ``(block_size``-wide K-group, output column``)`` -- the same granularity
    :func:`onnxsim.quantize_weight_only_int8_block` uses. Returns
    ``(wq_kn int8 [K, N], scale_gn float32 [K // block_size, N])``. Caller
    must ensure ``K % block_size == 0``.

    Kept as a reference oracle for ``tests/test_zeroquant_cpp.py`` (which
    cross-checks the C++ port's own weight-quantization math against it);
    the actual graph rewrite now happens entirely in the C++ port
    (``passes/zeroquant.h``), which this module's own
    :func:`apply_zeroquant` delegates to.
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

    Delegates to the verified C++ port (:func:`onnxsim.apply_zeroquant_cpp`,
    ``passes/zeroquant.h``), which builds the exact same node-for-node
    graph rewrite.

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
    return apply_zeroquant_cpp(model, block_size=block_size, epsilon=epsilon)
