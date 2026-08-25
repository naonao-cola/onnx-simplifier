"""Static quantization-precision analysis for MatMul/Gemm/Conv/Attention.

Given a model's constant weights and each node's shape hyperparameters --
the reduction depth K for MatMul/Gemm, Cin/groups * kernel-volume for Conv,
num_heads/head_dim for Attention -- this estimates whether INT8 quantization
(the scheme ``onnxsim.quantize_dynamic``/``quantize_static`` apply, see
docs/dynamic-quantization.md) is numerically safe, without running the model
or needing calibration data. Four independent questions, all answerable
from static graph info alone:

1. Accumulator overflow (MatMul/Gemm/Conv). onnxsim's INT8 weight
   quantization is symmetric and per-channel, scaled so a channel's
   largest-magnitude element always quantizes to +-127 -- so the worst-case
   *quantized* value is fixed by the scheme, not by the actual weight data.
   Paired with a uint8 activation (the full range DynamicQuantizeLinear /
   QuantizeLinear can produce), the worst-case accumulated value is bounded
   by ``reduction_depth * 127 * 255``. Once that exceeds INT32_MAX, an int32
   accumulator (MatMulInteger's, or a hypothetical ConvInteger's) can wrap
   around. This bound is exact and depends only on the reduction depth, which
   is why ``onnxsim.quantize_dynamic`` itself refuses to quantize a node past
   it (see ``passes/quantize_matmul_common.h``'s ``IsSafeInt32ReductionDepth``,
   which this module's threshold matches).

2. Effective resolution / outlier risk (MatMul/Gemm/Conv). *Within* the safe
   range, actual weight data matters: since a channel's scale is set by its
   single largest-magnitude element, a channel with a few extreme outliers
   wastes most of its 8 bits on values the bulk of the channel's weights
   never approach. This is estimated from each channel's
   ``max(|w|) / median(|w|)`` ratio: an 8-bit symmetric quantizer has 127
   positive levels, so once that ratio exceeds 127 the channel's *typical*
   (median-magnitude) weight rounds to within one quantization step of zero --
   effectively losing it.

3. float32-cast exactness (MatMul/Gemm/Conv) -- a separate effect from (1),
   not a correctness bound. Even a node that clears the int32-overflow check
   still has its accumulator run through ``Cast<float>(Acc)`` before
   dequantization; float32's 24-bit mantissa represents integers exactly only
   up to 2**24, so once ``reduction_depth * 127 * 255`` passes that (a much
   lower bar than INT32_MAX -- around 518 reduction terms), the cast rounds
   to the nearest representable float32. This is ordinary floating-point
   rounding, not overflow, and at ~2**-24 relative it is negligible next to
   INT8 quantization's own ~1/127 (~0.8%) relative error -- reported so
   "int32-safe" is never read as "exact end-to-end".

4. Activation-range provenance (MatMul/Gemm/Conv). These compute-dominant
   ops never run in isolation -- their activation input is almost always the
   output of another op, and a few common ones (``Sigmoid``, ``Tanh``,
   ``HardSigmoid``, ``Softmax``, and ``Clip`` with constant bounds) have an
   output range that is fixed by the op itself, for *any* input -- not a
   property of the data, so no calibration run is needed to know it. This is
   a distinct claim from (1)-(3): it does NOT tighten the accumulator-overflow
   bound (``DynamicQuantizeLinear`` rescales to the observed run's actual
   min/max regardless of the op's theoretical range, so a near-uniform
   Softmax output still spreads across most of uint8's range -- the
   worst-case bound in (1) already accounts for that and is unaffected). What
   it DOES mean: such a tensor could be quantized with a single, fixed,
   analytically-derived scale -- no calibration dataset (unlike an arbitrary
   activation, which onnxsim.quantize_static needs calibration data for) and
   no runtime ``DynamicQuantizeLinear`` overhead (unlike
   onnxsim.quantize_dynamic's current scheme) -- reported as
   ``activation_range``/``activation_producer_op`` when recognized.

Attention has no constant weight in the MatMul sense (Q/K/V are runtime
activations), so it gets a different, advisory-only estimate: the pre-softmax
QK^T dot product's magnitude grows with head_dim, which is exactly why
attention scales by ``1 / sqrt(head_dim)`` (Vaswani et al., 2017) -- this
reports that expected scale against the node's actual ``scale`` attribute (or
ai.onnx opset 23 Attention's own default) so a mismatch that would leave
softmax's input un-normalized is visible statically.

This module is read-only: it never modifies the model. The actual analysis
runs in C++ (``onnxsim/precision_estimator.h``/``.cpp``) -- the same
implementation the WASM converter UI's "Check quantization risk" button
calls into (``scripts/convertmodel/interface.cpp``) -- so the algorithm
exists in exactly one place. This module is a thin wrapper that reconstructs
the dataclasses below from the C++ extension's result; see that C++ module's
own doc comment for the implementation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import onnx

# INT32_MAX // (127 * 255) -- see this module's docstring, point 1. Kept as a
# literal formula (not a hardcoded constant) so it stays obviously in sync
# with passes/quantize_matmul_common.h's MaxSafeInt32ReductionDepth(), which
# onnxsim.quantize_dynamic actually enforces.
MAX_SAFE_INT32_REDUCTION_DEPTH = (2**31 - 1) // (127 * 255)

# An 8-bit symmetric quantizer has floor(127) positive levels (scale =
# max|w| / 127); a channel's max(|w|) / median(|w|) ratio past this leaves
# its median-magnitude weight within one quantization step of zero.
OUTLIER_RATIO_THRESHOLD = 127.0

# Beyond MAX_SAFE_INT32_REDUCTION_DEPTH, the int32 accumulator itself can
# wrap around -- a correctness bug. This second, much smaller threshold marks
# a different, non-overflowing effect: even when the accumulator is safe,
# the graph's ``Cast<float>(Acc)`` step (see docs/dynamic-quantization.md)
# converts that int32 sum to float32, whose 24-bit mantissa represents
# integers exactly only up to 2**24. Past this many reduction terms, the
# worst-case accumulated value can exceed 2**24 and the cast rounds it to the
# nearest representable float32 -- ordinary floating-point error, not a
# correctness bug, and, at ~2**-24 relative, negligible next to INT8
# quantization's own ~1/127 (~0.8%) relative error by design. Included so
# "int32-safe" is never conflated with "exact end-to-end".
MAX_EXACT_FLOAT32_REDUCTION_DEPTH = (2**24) // (127 * 255)


@dataclass
class MatMulGemmPrecisionEstimate:
    node_name: str
    op_type: str  # "MatMul" or "Gemm"
    reduction_depth: int
    num_channels: int
    int32_accumulator_safe: bool
    float32_cast_exact: bool  # False just means routine float rounding, not a bug
    max_outlier_ratio: (
        float  # max over channels of max|w| / median(|w|); nan if unknown
    )
    outlier_risk: bool
    activation_producer_op: Optional[str]  # e.g. "Sigmoid"; None if not recognized
    activation_range: Optional[Tuple[float, float]]  # analytically-known (lo, hi)
    recommendation: str


@dataclass
class ConvPrecisionEstimate:
    node_name: str
    reduction_depth: int  # (Cin / groups) * kernel volume
    num_channels: int  # Cout
    int32_accumulator_safe: bool
    float32_cast_exact: bool
    max_outlier_ratio: float
    outlier_risk: bool
    activation_producer_op: Optional[str]
    activation_range: Optional[Tuple[float, float]]
    recommendation: str


@dataclass
class AttentionPrecisionEstimate:
    node_name: str
    num_query_heads: Optional[int]
    num_kv_heads: Optional[int]
    head_dim: Optional[int]
    default_scale: Optional[float]
    actual_scale: Optional[float]
    scale_matches_default: Optional[bool]
    recommendation: str


PrecisionEstimate = Union[
    MatMulGemmPrecisionEstimate, ConvPrecisionEstimate, AttentionPrecisionEstimate
]


@dataclass
class ModelQuantizationEstimate:
    """A single, whole-model rollup of :func:`estimate_quantization_precision`'s
    per-node estimates -- see :func:`estimate_model_quantization_drop`."""

    total_nodes_analyzed: int
    unsafe_nodes: List[str]  # node names failing the int32-accumulator-safe check
    outlier_risk_nodes: List[str]  # node names flagged outlier_risk=True
    worst_outlier_ratio: float  # max over all analyzed nodes; nan if none had one
    estimated_relative_error: float  # see docstring below; nan if any unsafe_nodes
    risk_level: str  # "unsafe" | "degraded" | "safe"
    per_node: List[PrecisionEstimate]


def _weight_estimate_from_tuple(t) -> Union[MatMulGemmPrecisionEstimate, ConvPrecisionEstimate]:
    (
        node_name,
        op_type,
        reduction_depth,
        num_channels,
        int32_accumulator_safe,
        float32_cast_exact,
        max_outlier_ratio,
        outlier_risk,
        activation_producer_op,
        activation_range_lo,
        activation_range_hi,
        recommendation,
    ) = t
    activation_range = (
        (activation_range_lo, activation_range_hi)
        if activation_range_lo is not None
        else None
    )
    if op_type == "Conv":
        return ConvPrecisionEstimate(
            node_name=node_name,
            reduction_depth=reduction_depth,
            num_channels=num_channels,
            int32_accumulator_safe=int32_accumulator_safe,
            float32_cast_exact=float32_cast_exact,
            max_outlier_ratio=max_outlier_ratio,
            outlier_risk=outlier_risk,
            activation_producer_op=activation_producer_op,
            activation_range=activation_range,
            recommendation=recommendation,
        )
    return MatMulGemmPrecisionEstimate(
        node_name=node_name,
        op_type=op_type,
        reduction_depth=reduction_depth,
        num_channels=num_channels,
        int32_accumulator_safe=int32_accumulator_safe,
        float32_cast_exact=float32_cast_exact,
        max_outlier_ratio=max_outlier_ratio,
        outlier_risk=outlier_risk,
        activation_producer_op=activation_producer_op,
        activation_range=activation_range,
        recommendation=recommendation,
    )


def _attention_estimate_from_tuple(t) -> AttentionPrecisionEstimate:
    (
        node_name,
        num_query_heads,
        num_kv_heads,
        head_dim,
        default_scale,
        actual_scale,
        scale_matches_default,
        recommendation,
    ) = t
    return AttentionPrecisionEstimate(
        node_name=node_name,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        default_scale=default_scale,
        actual_scale=actual_scale,
        scale_matches_default=scale_matches_default,
        recommendation=recommendation,
    )


def _call_estimator(model: onnx.ModelProto):
    # Imported lazily so importing ``onnxsim.precision_estimator`` never forces
    # the compiled extension at module load, and cannot form an import cycle
    # with the extension -- mirrors onnxsim.model_info's own lazy import of
    # the same extension module.
    from onnxsim import onnxsim_cpp2py_export as _C

    return _C._estimate_model_quantization_drop(model.SerializeToString())


def estimate_quantization_precision(model: onnx.ModelProto) -> List[PrecisionEstimate]:
    """Per-node INT8-quantization precision estimates for `model`.

    Walks every MatMul, Gemm, Conv, and (ai.onnx opset 23+) Attention node in
    the top-level graph (subgraphs -- e.g. inside If/Loop/Scan -- are not
    descended into) and returns one estimate per node whose weight is a
    top-level graph initializer (nodes whose weight comes from a Constant
    node, a subgraph, or isn't a float MatMul-like/Conv shape are skipped).
    This never modifies `model`.
    """
    _, _, _, _, _, _, weight_tuples, attention_tuples = _call_estimator(model)
    estimates: List[PrecisionEstimate] = [
        _weight_estimate_from_tuple(t) for t in weight_tuples
    ]
    estimates.extend(_attention_estimate_from_tuple(t) for t in attention_tuples)
    return estimates


def estimate_model_quantization_drop(
    model: onnx.ModelProto,
) -> ModelQuantizationEstimate:
    """Aggregates :func:`estimate_quantization_precision`'s per-node estimates
    into a single whole-model INT8-quantization risk summary and an
    *estimated* relative-error figure -- purely from the model's static
    weights and shapes, no execution or calibration data needed. For an
    actual, data-driven measurement of a specific quantized model's real
    accuracy drop, see :func:`onnxsim.accuracy.measure_accuracy_drop` instead
    -- this function is the fast, no-data pre-check; that one is the ground
    truth.

    ``risk_level``:

    - ``"unsafe"``: at least one MatMul/Gemm/Conv node's reduction depth
      exceeds the int32-accumulator-safe bound (see this module's docstring,
      point 1) -- a real correctness bug, not a precision tradeoff.
      ``estimated_relative_error`` is ``nan`` in this case: an overflowing
      accumulator's output isn't bounded by any small error term, so no
      single number would honestly describe it.
    - ``"degraded"``: no unsafe nodes, but at least one node has a
      ``max(|w|) / median(|w|)`` outlier ratio past
      :data:`OUTLIER_RATIO_THRESHOLD` -- INT8 quantization is safe but loses
      meaningful resolution on that node's typical-magnitude weights.
    - ``"safe"``: neither of the above.

    ``estimated_relative_error`` (only meaningful when ``risk_level`` is not
    ``"unsafe"``): a per-node relative RMS quantization-noise estimate, from
    the standard uniform-quantizer noise model -- a symmetric INT8 quantizer
    with step ``Delta = max(|w|) / 127`` has quantization error uniformly
    distributed in ``[-Delta/2, Delta/2]``, whose RMS is ``Delta / sqrt(12)``.
    Relative to a channel's *typical* (median-magnitude, not peak-magnitude)
    weight, that RMS error scales by the channel's own outlier ratio ``r =
    max(|w|) / median(|w|)`` (``r = 1`` when no outlier ratio was computable,
    e.g. a channel with fewer than two nonzero weights):

        per_node_relative_error = r / (127 * sqrt(12))

    Whole-model ``estimated_relative_error`` combines every analyzed
    MatMul/Gemm/Conv node's ``per_node_relative_error`` in root-sum-square --
    the standard back-of-envelope combination for independent noise sources
    -- as a **heuristic**, not a guarantee: it assumes each node's
    quantization error behaves as an independent random perturbation that
    neither compounds multiplicatively through the network's depth nor
    cancels out, which real networks only approximate. Treat this as a
    relative ranking/screening signal (worse models get a bigger number),
    not a certified error bound -- :func:`onnxsim.accuracy.measure_accuracy_drop`
    is the tool for an actual bound on a specific model and dataset.
    Attention nodes are excluded from this sum (no weight-quantization error
    term applies to them in this scheme -- see this module's docstring).
    """
    (
        total_nodes_analyzed,
        unsafe_nodes,
        outlier_risk_nodes,
        worst_outlier_ratio,
        estimated_relative_error,
        risk_level,
        weight_tuples,
        attention_tuples,
    ) = _call_estimator(model)
    per_node: List[PrecisionEstimate] = [
        _weight_estimate_from_tuple(t) for t in weight_tuples
    ]
    per_node.extend(_attention_estimate_from_tuple(t) for t in attention_tuples)

    return ModelQuantizationEstimate(
        total_nodes_analyzed=total_nodes_analyzed,
        unsafe_nodes=list(unsafe_nodes),
        outlier_risk_nodes=list(outlier_risk_nodes),
        worst_outlier_ratio=worst_outlier_ratio,
        estimated_relative_error=estimated_relative_error,
        risk_level=risk_level,
        per_node=per_node,
    )
