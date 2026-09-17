"""Attention computation quantization.

Every other quantizer in onnxsim targets either a weight-bearing
MatMul/Gemm layer (`quantize_weight_only_int4` and everything built on
it -- `apply_spinquant`, `apply_quarot`, `apply_duquant`, ...) or the
KV-cache tensors specifically (:mod:`onnxsim.kv_cache_quantization`).
Nothing quantizes the attention *computation* itself: the
``QK^T`` score matmul, or the ``softmax(QK^T)@V`` value-weighted sum.
Both are pure activation-to-activation matmuls (no constant weight at
all), so none of onnxsim's weight-quantization machinery applies to them.

This module targets the common **decomposed** attention subgraph most
ONNX exports still produce (rather than the newer, opset-23+ fused
``Attention`` operator :mod:`onnxsim.precision_estimator` already has
advisory-only awareness of -- see that module's own docstring, point 4,
which flags Softmax's output range but doesn't act on it):

    scores  = MatMul(Q, Kt)                  -- Kt: K, transposed
    scaled  = Mul(scores, scale)  [optional]  -- e.g. 1/sqrt(head_dim)
    masked  = Add(scaled, mask)   [optional]  -- e.g. causal mask
    probs   = Softmax(masked, axis=-1)
    out     = MatMul(probs, V)

Three tensors get quantized to INT8, each via the technique already
best-suited to what it actually is -- **no calibration data needed for
any of them**:

- **Q and K** (the score matmul's own operands): data-free, per-token
  dynamic INT8 (the same pattern :mod:`onnxsim.quarot`/:mod:`onnxsim.duquant`
  already use for their own activation quantization -- ``scale =
  max(|x|, axis=-1) / 127``, computed fresh at graph-run time, no
  calibration statistics stored).
- **V** (the second matmul's other operand): the same per-token dynamic
  INT8 scheme.
- **The Softmax output itself** (the attention *probabilities*): unlike
  every other activation in this codebase, a Softmax output's range is
  not merely typical -- it is *guaranteed* to lie in ``[0, 1]`` for any
  input at all (the same fact :mod:`onnxsim.precision_estimator`'s own
  docstring already names as "activation-range provenance", point 4, but
  never previously used to actually quantize anything). That makes it the
  one activation in this whole package that can be quantized with a
  **fixed, non-data-dependent** scale -- ``UINT8`` with ``scale = 1/255``,
  ``zero_point = 0`` -- no calibration run, no runtime scale computation,
  just an ordinary round-to-nearest against a constant.

Score computation itself (``MatMul(Q, Kt)``) and the softmax normalization
are left running in float, exactly as SmoothQuant/AWQ leave their own
internal reductions in float -- only the three tensors *crossing* a
matmul boundary are quantized, matching every other onnxsim quantizer's
own convention of touching operands, not recomputing an op's own math in
lower precision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import onnx

from onnxsim.onnx_simplifier import apply_attention_quantization_cpp


def _find_matmul_producer(
    name: str, producer_by_output: Dict[str, onnx.NodeProto], hops_left: int
) -> Optional[onnx.NodeProto]:
    """Walks back from tensor ``name`` through at most ``hops_left``
    Mul/Div/Add nodes (following each one's *first* input only -- the
    scale/mask operand, never the divisor/mask itself) looking for the
    MatMul that produced the raw attention scores. Returns ``None`` if no
    such MatMul is found within the hop budget.
    """
    node = producer_by_output.get(name)
    if node is None:
        return None
    if node.op_type == "MatMul":
        return node
    if hops_left <= 0 or node.op_type not in ("Mul", "Div", "Add"):
        return None
    return _find_matmul_producer(node.input[0], producer_by_output, hops_left - 1)


class _AttentionCandidate:
    def __init__(
        self,
        qk_matmul: onnx.NodeProto,
        softmax: onnx.NodeProto,
        out_matmul: onnx.NodeProto,
    ):
        self.qk_matmul = qk_matmul
        self.softmax = softmax
        self.out_matmul = out_matmul


def _find_attention_candidates(graph: onnx.GraphProto) -> List[_AttentionCandidate]:
    producer_by_output: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            producer_by_output[out] = node

    consumers_by_input: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers_by_input.setdefault(inp, []).append(node)

    candidates = []
    for node in graph.node:
        if node.op_type != "Softmax":
            continue
        qk_matmul = _find_matmul_producer(node.input[0], producer_by_output, 2)
        if qk_matmul is None:
            continue
        softmax_out = node.output[0]
        consumers = consumers_by_input.get(softmax_out, [])
        out_matmul = next(
            (
                c
                for c in consumers
                if c.op_type == "MatMul" and c.input[0] == softmax_out
            ),
            None,
        )
        if out_matmul is None:
            continue
        candidates.append(_AttentionCandidate(qk_matmul, node, out_matmul))
    return candidates


def apply_attention_quantization(
    model: Union[str, onnx.ModelProto],
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """Quantizes every decomposed attention subgraph
    (``MatMul(Q,Kt) -> [Mul/Div] -> [Add] -> Softmax -> MatMul(_,V)``) to
    INT8 -- see this module's own docstring for the technique. Needs no
    calibration data at all: Q/K/V use data-free per-token dynamic
    scales, and the Softmax output uses a fixed scale (its range is
    always ``[0, 1]``, by construction).

    Delegates to :func:`onnxsim.apply_attention_quantization_cpp` (the
    verified C++ port); this pure-Python name is kept only for backward
    compatibility with existing callers.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param epsilon: floor applied to a token's own max-abs Q/K/V value
            before using it as a scale, avoiding a divide-by-zero on an
            all-zero token. The C++ port hardcodes this session's own
            default (``1e-12``); a caller-supplied value that differs from
            that default cannot be honored, since the C++ port exposes no
            way to override it.
    :returns: ``model`` with every matched subgraph's ``Q``, ``Kt``, and
            ``V`` operands, and the Softmax output, replaced by INT8
            round-trip (quantize-then-immediately-dequantize, kept in
            float32 to simulate the precision loss without a true integer
            matmul) versions; the score MatMul and the Softmax
            normalization itself are left running in float. A model with
            no matching subgraph, or an opset older than 18 (``ReduceMax``'s
            ``axes``-as-input form, used for the per-token Q/K/V scale,
            needs opset 18 -- matching :func:`onnxsim.quantize_kv_cache`'s
            own Value-style gate), is returned unchanged
    """
    if epsilon != 1e-12:
        raise NotImplementedError(
            "apply_attention_quantization now delegates to the C++ port, "
            "which hardcodes epsilon=1e-12 and cannot honor a different "
            "value; call with the default epsilon, or use "
            "apply_attention_quantization_cpp directly."
        )
    return apply_attention_quantization_cpp(model)
