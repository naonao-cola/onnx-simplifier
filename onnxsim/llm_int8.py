"""LLM.int8() (Dettmers et al., 2022, "LLM.int8(): 8-bit Matrix
Multiplication for Transformers at Scale", https://arxiv.org/abs/2208.07339).
bitsandbytes' original 8-bit scheme (``bnb.nn.Linear8bitLt`` /
``bnb.matmul(..., threshold=6.0)``), distinct from :mod:`onnxsim.nf4`
(bitsandbytes' *4-bit* NF4 codebook) -- onnxsim ports the algorithm, not
that code, per the same rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/
:mod:`onnxsim.hqq`/:mod:`onnxsim.smoothquant` (bitsandbytes quantizes live
PyTorch ``nn.Module``s with no ONNX export path).

:mod:`onnxsim.smoothquant` addresses outlier activation channels by
*migrating* their difficulty into the weight before quantizing everything
uniformly. LLM.int8() takes a different approach: it does not touch any
channel it considers an outlier at all. For a MatMul/Gemm ``Y = X @ W``, it
finds the input-channel columns of ``X`` whose magnitude exceeds a fixed
threshold (the paper's own default, ``6.0``) *anywhere* in the calibration
data -- empirically a small, consistent subset of channels, concentrated in
a few systematic feature dimensions rather than scattered randomly -- and
decomposes the matmul into two independent parts summed together:

- the outlier columns of ``X`` against the matching rows of ``W``, computed
  in plain float32 (exact, no quantization at all)
- every other column/row, quantized to INT8 and computed via
  ``MatMulInteger`` -- vector-wise: one absmax scale per activation *row*
  (computed at runtime, since it depends on that inference's actual input,
  not on calibration statistics) and one absmax scale per weight *output
  channel* (computed once, offline, since the weight is constant)

Because the (usually <1%) outlier channels are excluded from the INT8 part
entirely rather than merely rescaled, the remaining channels' dynamic range
is far tighter and INT8 quantization loses much less precision on them --
the paper's central empirical finding is that this preserves accuracy at
scale where naive full-tensor INT8 quantization degrades badly.

The activation's INT8 half is quantized to *unsigned* 8-bit with a constant
zero-point of 128 (rather than plain signed INT8) purely as an ONNX Runtime
compatibility choice: this repository's own ``dynamic_quantize_matmul``
C++ pass (see ``passes/dynamic_quantize_matmul.h``) established uint8
activation + int8 weight as the ``MatMulInteger`` operand combination this
codebase's test harness runs correctly; this module reuses that same
combination rather than risking an unsupported int8-times-int8 kernel path.
The math is unaffected -- ``(uint8_value - 128)`` recovers the exact
symmetric INT8 code before it is ever multiplied by the weight -- only the
storage dtype differs from the paper's own int8-times-int8 description.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data

# MatMulInteger's operands are uint8 (activation, offset by a zero-point of
# 128) and int8 (weight) -- see this module's own docstring. The worst-case
# per-term product magnitude is therefore 127 * 255 (an int8 code times the
# largest possible uint8-minus-zero-point spread), matching
# quantize_matmul_common.h's own MaxSafeInt32ReductionDepth bound: past this
# many terms, the int32 accumulator could wrap around in the worst case.
_MAX_SAFE_INT32_REDUCTION_DEPTH = (2**31 - 1) // (127 * 255)


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(x_name, w_name, bias_name_or_None,
    weight_transposed)`` or ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[0], node.input[1], None, False
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        bias_name = None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
            bias_name = node.input[2]
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[0], node.input[1], bias_name, weight_transposed
    return None


def apply_llm_int8(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    outlier_threshold: float = 6.0,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Decomposes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight and a plain 2-D activation input into an outlier
    float32 part plus a vector-wise INT8 part, using real calibration
    activations to find each layer's outlier channels. See this module's
    own docstring for the technique.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to find each
            layer's outlier channels on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative outlier search than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param outlier_threshold: an input channel is treated as an outlier if
            its activation magnitude exceeds this anywhere in the
            calibration data (the paper's own default, ``6.0``)
    :param epsilon: floor applied to a zero row/weight-column max-abs value
            before dividing by it, avoiding a divide-by-zero
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer replaced by its
            outlier/INT8 decomposition, output tensor name unchanged;
            layers with a non-constant, non-2-D weight, a non-2-D
            activation, no outlier channels found, every channel found to
            be an outlier, or whose non-outlier reduction depth is unsafe
            for ``MatMulInteger``'s int32 accumulator, are left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    opset_ge_18 = any(
        o.domain in ("", "ai.onnx") and o.version >= 18 for o in out.opset_import
    )
    if not opset_ge_18:
        return out  # ReduceMax's axes-as-input form needs opset >= 18

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

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    act_absmax: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            m = np.abs(x).max(axis=0)
            act_absmax[name] = (
                m if name not in act_absmax else np.maximum(act_absmax[name], m)
            )

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        acts = act_absmax.get(x_name)
        if acts is None:
            continue  # never observed as a plain 2-D tensor; skip

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        k = w_nk.shape[1]
        if acts.shape[0] != k:
            continue  # activation's feature dim doesn't match K; skip

        outlier_idx = np.where(acts > outlier_threshold)[0]
        regular_idx = np.where(acts <= outlier_threshold)[0]
        if outlier_idx.size == 0 or regular_idx.size == 0:
            continue  # nothing to decompose
        if regular_idx.size > _MAX_SAFE_INT32_REDUCTION_DEPTH:
            continue  # MatMulInteger's int32 accumulator could overflow

        w_outlier_kn = w_nk[:, outlier_idx].T.astype(np.float32)  # [no, N]
        w_regular_kn = w_nk[:, regular_idx].T  # [nr, N]
        col_scale = np.maximum(np.abs(w_regular_kn).max(axis=0), epsilon) / 127.0  # [N]
        wq_regular = np.clip(
            np.round(w_regular_kn / col_scale[np.newaxis, :]), -127, 127
        ).astype(np.int8)

        prefix = f"{x_name}_llm_int8"
        const_names = {}
        for suffix, array in (
            ("outlier_idx", outlier_idx.astype(np.int64)),
            ("regular_idx", regular_idx.astype(np.int64)),
            ("w_outlier", w_outlier_kn),
            ("wq_regular", wq_regular),
            ("col_scale", col_scale.astype(np.float32)),
            ("axis1", np.array([1], dtype=np.int64)),
            ("eps", np.array(epsilon, dtype=np.float32)),
            ("c127", np.array(127.0, dtype=np.float32)),
            ("neg127", np.array(-127.0, dtype=np.float32)),
            ("pos127", np.array(127.0, dtype=np.float32)),
            ("offset128", np.array(128.0, dtype=np.float32)),
            ("zp128", np.array(128, dtype=np.uint8)),
        ):
            name = _unique_name(f"{prefix}_{suffix}", taken_names)
            graph.initializer.append(onnx.numpy_helper.from_array(array, name=name))
            const_names[suffix] = name

        new_nodes: List[onnx.NodeProto] = []

        def _new(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"{prefix}_{out_suffix}", taken_names)
            n = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{out_suffix}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n)
            return out_name

        x_outlier = _new(
            "Gather", [x_name, const_names["outlier_idx"]], "x_outlier", axis=1
        )
        x_regular = _new(
            "Gather", [x_name, const_names["regular_idx"]], "x_regular", axis=1
        )
        y_outlier = _new("MatMul", [x_outlier, const_names["w_outlier"]], "y_outlier")

        x_abs = _new("Abs", [x_regular], "x_abs")
        row_max = _new(
            "ReduceMax", [x_abs, const_names["axis1"]], "row_max", keepdims=1
        )
        row_max_safe = _new("Max", [row_max, const_names["eps"]], "row_max_safe")
        row_scale = _new("Div", [row_max_safe, const_names["c127"]], "row_scale")
        x_scaled = _new("Div", [x_regular, row_scale], "x_scaled")
        x_rounded = _new("Round", [x_scaled], "x_rounded")
        x_clipped = _new(
            "Clip",
            [x_rounded, const_names["neg127"], const_names["pos127"]],
            "x_clipped",
        )
        x_shifted = _new("Add", [x_clipped, const_names["offset128"]], "x_shifted")
        xq = _new("Cast", [x_shifted], "xq", to=onnx.TensorProto.UINT8)

        y_int32 = _new(
            "MatMulInteger",
            [xq, const_names["wq_regular"], const_names["zp128"]],
            "y_int32",
        )
        y_int32_f = _new("Cast", [y_int32], "y_int32_f", to=onnx.TensorProto.FLOAT)
        y_partial = _new("Mul", [y_int32_f, row_scale], "y_partial")
        y_regular = _new("Mul", [y_partial, const_names["col_scale"]], "y_regular")

        old_output = node.output[0]
        if bias_name is not None:
            combined = _new("Add", [y_outlier, y_regular], "combined")
            final = onnx.helper.make_node(
                "Add",
                [combined, bias_name],
                [old_output],
                name=_unique_name(f"{prefix}_bias_add_node", taken_names),
            )
        else:
            final = onnx.helper.make_node(
                "Add",
                [y_outlier, y_regular],
                [old_output],
                name=_unique_name(f"{prefix}_combine_node", taken_names),
            )
        new_nodes.append(final)

        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
