"""Static quantization-precision analysis for MatMul/Gemm/Conv/Attention.

Given a model's constant weights and each node's shape hyperparameters --
the reduction depth K for MatMul/Gemm, Cin/groups * kernel-volume for Conv,
num_heads/head_dim for Attention -- this estimates whether INT8 quantization
(the scheme ``onnxsim.quantize_dynamic``/``quantize_static`` apply, see
docs/dynamic-quantization.md) is numerically safe, without running the model
or needing calibration data. Three independent questions, all answerable
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

Attention has no constant weight in the MatMul sense (Q/K/V are runtime
activations), so it gets a different, advisory-only estimate: the pre-softmax
QK^T dot product's magnitude grows with head_dim, which is exactly why
attention scales by ``1 / sqrt(head_dim)`` (Vaswani et al., 2017) -- this
reports that expected scale against the node's actual ``scale`` attribute (or
ai.onnx opset 23 Attention's own default) so a mismatch that would leave
softmax's input un-normalized is visible statically.

This module is read-only: it never modifies the model and has no C++/pybind
component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

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
    max_outlier_ratio: float  # max over channels of max|w| / median(|w|); nan if unknown
    outlier_risk: bool
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


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _get_attr_float(node: onnx.NodeProto, name: str) -> Optional[float]:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return None


def _channel_outlier_ratio(abs_weights_for_channel) -> float:
    """max(|w|) / median(|w|) over one channel's weights, ignoring zeros.

    Zeros are excluded because a channel legitimately containing many exact
    zeros (e.g. after pruning) would otherwise report a spuriously enormous
    ratio driven by sparsity, not by outlier magnitude among the weights that
    actually carry information. Returns ``nan`` when fewer than two nonzero
    weights remain (the ratio isn't meaningful).
    """
    nonzero = abs_weights_for_channel[abs_weights_for_channel > 0]
    if nonzero.size < 2:
        return math.nan
    peak = float(nonzero.max())
    med = float(np.median(nonzero))
    if med == 0.0:
        return math.nan
    return peak / med


def _recommendation(safe: bool, float32_exact: bool, outlier_risk: bool) -> str:
    if not safe:
        return (
            "reduction depth exceeds the int32-safe bound: onnxsim.quantize_dynamic "
            "already skips this node; INT8 with a wider (int64) accumulator, or "
            "splitting the reduction (blockwise quantization), would be needed"
        )
    notes = []
    if not float32_exact:
        notes.append(
            "the accumulator's Cast<float> is not bit-exact past this reduction "
            "depth (float32's 24-bit mantissa), though the rounding involved "
            "(~2**-24 relative) is far below INT8's own ~1/127 quantization error "
            "and does not change the recommendation"
        )
    if outlier_risk:
        notes.append(
            "a channel's outliers dominate its scale: per-group quantization or "
            "INT16 would preserve more resolution for this node's typical-magnitude "
            "weights"
        )
    if not notes:
        return "INT8 (onnxsim.quantize_dynamic's scheme) looks safe and well-resolved"
    return "int32-safe, but " + "; ".join(notes)


def _estimate_matmul_gemm(
    node: onnx.NodeProto, initializers: Dict[str, onnx.TensorProto]
) -> Optional[MatMulGemmPrecisionEstimate]:
    if node.op_type not in ("MatMul", "Gemm"):
        return None
    if len(node.input) < 2 or node.input[1] not in initializers:
        return None
    w = numpy_helper.to_array(initializers[node.input[1]])
    if w.dtype.kind != "f" or w.ndim != 2:
        return None

    transposed = node.op_type == "Gemm" and _get_attr_int(node, "transB", 0) != 0
    if node.op_type == "Gemm" and (
        _get_attr_int(node, "transA", 0) != 0
        or _get_attr_float(node, "alpha") not in (None, 1.0)
    ):
        return None

    k, n = (w.shape[1], w.shape[0]) if transposed else (w.shape[0], w.shape[1])
    channel_axis = 0 if transposed else 1
    abs_w = abs(w)

    ratios = [
        _channel_outlier_ratio(abs_w.take(c, axis=channel_axis)) for c in range(n)
    ]
    finite_ratios = [r for r in ratios if not math.isnan(r)]
    max_ratio = max(finite_ratios) if finite_ratios else math.nan
    safe = k <= MAX_SAFE_INT32_REDUCTION_DEPTH
    float32_exact = k <= MAX_EXACT_FLOAT32_REDUCTION_DEPTH
    outlier_risk = bool(finite_ratios) and max_ratio > OUTLIER_RATIO_THRESHOLD

    return MatMulGemmPrecisionEstimate(
        node_name=node.name or f"{node.op_type}({node.input[1]})",
        op_type=node.op_type,
        reduction_depth=int(k),
        num_channels=int(n),
        int32_accumulator_safe=safe,
        float32_cast_exact=float32_exact,
        max_outlier_ratio=max_ratio,
        outlier_risk=outlier_risk,
        recommendation=_recommendation(safe, float32_exact, outlier_risk),
    )


def _estimate_conv(
    node: onnx.NodeProto, initializers: Dict[str, onnx.TensorProto]
) -> Optional[ConvPrecisionEstimate]:
    if node.op_type != "Conv":
        return None
    if len(node.input) < 2 or node.input[1] not in initializers:
        return None
    w = numpy_helper.to_array(initializers[node.input[1]])
    if w.dtype.kind != "f" or w.ndim < 3:
        return None

    cout = w.shape[0]
    k = int(w[0].size)  # (Cin / groups) * kernel volume
    abs_w = abs(w).reshape(cout, -1)

    ratios = [_channel_outlier_ratio(abs_w[c]) for c in range(cout)]
    finite_ratios = [r for r in ratios if not math.isnan(r)]
    max_ratio = max(finite_ratios) if finite_ratios else math.nan
    safe = k <= MAX_SAFE_INT32_REDUCTION_DEPTH
    float32_exact = k <= MAX_EXACT_FLOAT32_REDUCTION_DEPTH
    outlier_risk = bool(finite_ratios) and max_ratio > OUTLIER_RATIO_THRESHOLD

    return ConvPrecisionEstimate(
        node_name=node.name or f"Conv({node.input[1]})",
        reduction_depth=k,
        num_channels=int(cout),
        int32_accumulator_safe=safe,
        float32_cast_exact=float32_exact,
        max_outlier_ratio=max_ratio,
        outlier_risk=outlier_risk,
        recommendation=_recommendation(safe, float32_exact, outlier_risk),
    )


def _static_dims(shape_proto: Optional[onnx.TensorShapeProto]) -> Optional[List[Optional[int]]]:
    if shape_proto is None:
        return None
    dims: List[Optional[int]] = []
    for d in shape_proto.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else None)
    return dims


def _collect_static_shapes(model: onnx.ModelProto) -> Dict[str, List[Optional[int]]]:
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes: Dict[str, List[Optional[int]]] = {}
    graph = inferred.graph
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            if vi.type.HasField("tensor_type"):
                dims = _static_dims(vi.type.tensor_type.shape)
                if dims is not None:
                    shapes[vi.name] = dims
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def _estimate_attention(
    node: onnx.NodeProto, shapes: Dict[str, List[Optional[int]]]
) -> Optional[AttentionPrecisionEstimate]:
    if node.op_type != "Attention" or len(node.input) < 3:
        return None
    q_shape = shapes.get(node.input[0])
    k_shape = shapes.get(node.input[1])

    q_heads = _get_attr_int(node, "q_num_heads", 0) or None
    kv_heads = _get_attr_int(node, "kv_num_heads", 0) or None
    head_dim: Optional[int] = None

    if q_shape and len(q_shape) == 4 and q_shape[3] is not None:
        q_heads = q_shape[1] if q_shape[1] is not None else q_heads
        head_dim = q_shape[3]
        if k_shape and len(k_shape) == 4:
            kv_heads = k_shape[1] if k_shape[1] is not None else kv_heads
    elif (
        q_shape
        and k_shape
        and len(q_shape) == 3
        and len(k_shape) == 3
        and kv_heads
        and k_shape[2] is not None
    ):
        head_dim = k_shape[2] // kv_heads

    default_scale = 1.0 / math.sqrt(head_dim) if head_dim else None
    actual_scale = _get_attr_float(node, "scale")
    matches: Optional[bool] = None
    if default_scale is not None and actual_scale is not None:
        matches = math.isclose(default_scale, actual_scale, rel_tol=1e-3)

    if head_dim is None:
        recommendation = (
            "head_dim could not be determined statically (dynamic/unknown shape "
            "and no q_num_heads/kv_num_heads attributes) -- no estimate available"
        )
    elif matches is False:
        recommendation = (
            f"scale={actual_scale!r} attribute does not match the canonical "
            f"1/sqrt(head_dim)={default_scale:.6g}: pre-softmax QK^T logits will "
            "grow unnormalized with head_dim, risking saturation/overflow in a "
            "low-precision (e.g. fp16) softmax"
        )
    else:
        recommendation = (
            "scale normalizes QK^T's head_dim-dependent growth as expected; "
            "no int8 accumulator applies here since Q/K/V are runtime activations, "
            "not a static weight"
        )

    return AttentionPrecisionEstimate(
        node_name=node.name or "Attention",
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        default_scale=default_scale,
        actual_scale=actual_scale,
        scale_matches_default=matches,
        recommendation=recommendation,
    )


def estimate_quantization_precision(model: onnx.ModelProto) -> List[PrecisionEstimate]:
    """Per-node INT8-quantization precision estimates for `model`.

    Walks every MatMul, Gemm, Conv, and (ai.onnx opset 23+) Attention node in
    the top-level graph (subgraphs -- e.g. inside If/Loop/Scan -- are not
    descended into) and returns one estimate per node whose weight is a
    top-level graph initializer (nodes whose weight comes from a Constant
    node, a subgraph, or isn't a float MatMul-like/Conv shape are skipped).
    This never modifies `model`.
    """
    initializers = {init.name: init for init in model.graph.initializer}
    shapes = _collect_static_shapes(model)

    estimates: List[PrecisionEstimate] = []
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            est = _estimate_matmul_gemm(node, initializers)
        elif node.op_type == "Conv":
            est = _estimate_conv(node, initializers)
        elif node.op_type == "Attention":
            est = _estimate_attention(node, shapes)
        else:
            est = None
        if est is not None:
            estimates.append(est)
    return estimates
