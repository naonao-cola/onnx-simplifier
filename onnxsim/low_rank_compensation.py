"""Low-Rank Compensation (LoRC), from ZeroQuant-V2 (Yao et al., 2023,
"ZeroQuant-V2: Exploring Post-training Quantization in LLMs from
Comprehensive Study to Low Rank Compensation",
https://arxiv.org/abs/2303.08302). onnxsim ports the algorithm, not any
framework's code, per the same rationale as :mod:`onnxsim.awq`/
:mod:`onnxsim.gptq` (DeepSpeed's own ZeroQuant implementation quantizes
live PyTorch modules, with no ONNX export path).

Every other refinement pass in onnxsim -- :mod:`onnxsim.adaround`,
:mod:`onnxsim.awq`, :mod:`onnxsim.gptq` -- takes ``quantize_weight_only_int4``'s
output and changes *how the weight itself gets quantized* (which bin each
element rounds to, or a rescaling applied before quantizing). LoRC takes a
different, much simpler angle: leave the existing INT4 quantization
completely alone, and instead directly cancel out however much error it
already has left over. For a quantized layer's own reconstruction error
matrix (``float_weight - dequantized_weight``, exact and already fully
known from the two weights -- no calibration data needed, unlike
adaround/AWQ/GPTQ, which all need real activations), the Eckart-Young
theorem says the best possible rank-``r`` approximation of any matrix
(minimizing Frobenius-norm error, i.e. mean squared error) is given
directly by that matrix's own truncated SVD -- keeping the ``r`` largest
singular values/vectors and discarding the rest. This module computes
that truncated SVD of each matched layer's error matrix once, offline, and
adds the resulting rank-``r`` correction back as two small extra ``MatMul``
nodes summed into the layer's output (``Y = X @ Wq_dequant + (X @ B) @ A``,
``B`` shape ``[K, r]``, ``A`` shape ``[r, N]``) -- cheap relative to the
layer's own ``O(N*K)`` matmul when ``r`` is small (the paper's own
experiments use ``r`` on the order of a few tens), and, being a strict
generalization (``r = min(N, K)`` recovers the exact float weight), a
strictly-improving one: adding more rank can only reduce the compensated
layer's reconstruction error, never increase it.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs
from onnxsim.bias_correction import _all_names, _unique_name


def apply_low_rank_compensation(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    rank: int = 8,
) -> onnx.ModelProto:
    """Adds a rank-``r`` low-rank correction to every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``,
    canceling out that much of its existing quantization error. See this
    module's own docstring for the technique. Needs no calibration data:
    the correction is computed directly from the two weight tensors.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param rank: the correction's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer); larger values recover more of the
            layer's quantization error at the cost of two proportionally
            larger extra ``MatMul`` nodes
    :returns: ``quantized_model`` with every matched layer's output summed
            with a new rank-``r`` correction term (two chained ``MatMul``
            nodes plus an ``Add``); the layer's own existing INT4 weight
            and scale are left completely untouched
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    if not candidates:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    graph = corrected.graph
    q_init = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    q_by_output = _node_outputs(graph)

    for c in candidates:
        wq_init = q_init[c.wq_name]
        codes = onnx.numpy_helper.to_array(wq_init).astype(np.float64)
        ws = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        scale_full = np.repeat(ws, c.block_size, axis=c.axis)
        slicer: List[slice] = [slice(None)] * codes.ndim
        slicer[c.axis] = slice(0, codes.shape[c.axis])
        w_dequant = codes * scale_full[tuple(slicer)]

        w_float = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        residual = w_float - w_dequant  # original storage layout [dim0, dim1]
        # Normalize to [K, N] -- ready for `X @ residual`, matching plain
        # MatMul's own weight orientation -- regardless of whether the
        # node stores its weight transposed.
        residual_kn = residual if not c.weight_transposed else residual.T
        k, n = residual_kn.shape
        r = min(rank, k, n)
        if r <= 0:
            continue

        u, s, vt = np.linalg.svd(residual_kn, full_matrices=False)
        b_kn = (u[:, :r] * s[np.newaxis, :r]).astype(np.float32)  # [K, r]
        a_rn = vt[:r, :].astype(np.float32)  # [r, N]

        qn = q_by_output[c.output_name]
        x_name = c.float_node.input[0]

        prefix = f"{c.output_name}_lorc"
        b_name = _unique_name(f"{prefix}_b", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(b_kn, name=b_name))
        a_name = _unique_name(f"{prefix}_a", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(a_rn, name=a_name))

        old_output = qn.output[0]
        base_name = _unique_name(f"{prefix}_base", taken_names)
        qn.output[0] = base_name

        tmp_name = _unique_name(f"{prefix}_tmp", taken_names)
        tmp_node = onnx.helper.make_node(
            "MatMul",
            [x_name, b_name],
            [tmp_name],
            name=_unique_name(f"{prefix}_matmul1_node", taken_names),
        )
        lowrank_name = _unique_name(f"{prefix}_lowrank", taken_names)
        lowrank_node = onnx.helper.make_node(
            "MatMul",
            [tmp_name, a_name],
            [lowrank_name],
            name=_unique_name(f"{prefix}_matmul2_node", taken_names),
        )
        add_node = onnx.helper.make_node(
            "Add",
            [base_name, lowrank_name],
            [old_output],
            name=_unique_name(f"{prefix}_add_node", taken_names),
        )

        node_idx = next(i for i, nd in enumerate(graph.node) if nd is qn)
        graph.node.insert(node_idx + 1, tmp_node)
        graph.node.insert(node_idx + 2, lowrank_node)
        graph.node.insert(node_idx + 3, add_node)

    return corrected
