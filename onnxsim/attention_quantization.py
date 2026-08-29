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

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


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

    :param model: the original (unquantized) onnx ModelProto or file path
    :param epsilon: floor applied to a token's own max-abs Q/K/V value
            before using it as a scale, avoiding a divide-by-zero on an
            all-zero token
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
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 18):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    taken_names = _all_names(graph)

    candidates = _find_attention_candidates(graph)
    if not candidates:
        return out

    eps_name = _unique_name("attnq_eps", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
    )
    i127_name = _unique_name("attnq_127", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(127.0, dtype=np.float32), name=i127_name)
    )
    i127_min_name = _unique_name("attnq_neg127", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(-127.0, dtype=np.float32), name=i127_min_name
        )
    )
    axes_name = _unique_name("attnq_axes", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name)
    )
    probs_scale_name = _unique_name("attnq_probs_scale", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(1.0 / 255.0, dtype=np.float32), name=probs_scale_name
        )
    )
    probs_max_name = _unique_name("attnq_probs_max", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(255.0, dtype=np.float32), name=probs_max_name
        )
    )
    probs_zero_name = _unique_name("attnq_probs_zero", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array(0.0, dtype=np.float32), name=probs_zero_name
        )
    )

    # Per-target-node lists of nodes to insert immediately *before* that
    # target, keyed by id() (NodeProto isn't hashable-by-value) -- built up
    # per candidate below, then spliced into the graph in a single pass so
    # every new node lands strictly before whichever original node first
    # needs its output, preserving topological order even though a
    # candidate's own new nodes split across two different insertion
    # points (before qk_matmul, and separately before out_matmul, since
    # the probs/V nodes depend on Softmax's own output, produced strictly
    # between those two original nodes).
    pre_insert: Dict[int, List[onnx.NodeProto]] = {}

    def _quantize_per_token_int8(
        x_name: str, tag: str, sink: List[onnx.NodeProto]
    ) -> str:
        def _new(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"attnq_{out_suffix}", taken_names)
            n_ = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"attnq_{out_suffix}_node", taken_names),
                **attrs,
            )
            sink.append(n_)
            return out_name

        # scale = max(reduce_max(abs(x), axis=-1), eps) / 127
        # x_q = clip(round(x / scale), -127, 127); x_dq = x_q * scale
        abs_name = _new("Abs", [x_name], f"{tag}_abs")
        max_name = _new("ReduceMax", [abs_name, axes_name], f"{tag}_max", keepdims=1)
        safe_max_name = _new("Clip", [max_name, eps_name], f"{tag}_safe_max")
        scale_name = _new("Div", [safe_max_name, i127_name], f"{tag}_scale")
        scaled_name = _new("Div", [x_name, scale_name], f"{tag}_scaled")
        rounded_name = _new("Round", [scaled_name], f"{tag}_rounded")
        clipped_name = _new(
            "Clip", [rounded_name, i127_min_name, i127_name], f"{tag}_clipped"
        )
        return _new("Mul", [clipped_name, scale_name], f"{tag}_dequant")

    for i, c in enumerate(candidates):
        qk_sink: List[onnx.NodeProto] = []
        q_dq = _quantize_per_token_int8(c.qk_matmul.input[0], f"q{i}", qk_sink)
        k_dq = _quantize_per_token_int8(c.qk_matmul.input[1], f"k{i}", qk_sink)
        c.qk_matmul.input[0] = q_dq
        c.qk_matmul.input[1] = k_dq
        pre_insert.setdefault(id(c.qk_matmul), []).extend(qk_sink)

        out_sink: List[onnx.NodeProto] = []

        def _new_out(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"attnq_{out_suffix}", taken_names)
            n_ = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"attnq_{out_suffix}_node", taken_names),
                **attrs,
            )
            out_sink.append(n_)
            return out_name

        # Fixed-scale UINT8 quantization of the softmax output: its range
        # is [0, 1] for any input at all, so scale=1/255, zero_point=0
        # needs no data-dependent computation.
        probs_scaled = _new_out(
            "Div", [c.softmax.output[0], probs_scale_name], f"probs{i}_scaled"
        )
        probs_rounded = _new_out("Round", [probs_scaled], f"probs{i}_rounded")
        probs_clipped = _new_out(
            "Clip",
            [probs_rounded, probs_zero_name, probs_max_name],
            f"probs{i}_clipped",
        )
        probs_dq = _new_out(
            "Mul", [probs_clipped, probs_scale_name], f"probs{i}_dequant"
        )

        v_dq = _quantize_per_token_int8(c.out_matmul.input[1], f"v{i}", out_sink)

        c.out_matmul.input[0] = probs_dq
        c.out_matmul.input[1] = v_dq
        pre_insert.setdefault(id(c.out_matmul), []).extend(out_sink)

    rebuilt: List[onnx.NodeProto] = []
    for node in list(graph.node):
        rebuilt.extend(pre_insert.get(id(node), []))
        rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)

    return out
