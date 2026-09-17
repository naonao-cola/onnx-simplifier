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

from typing import Union

import onnx

from onnxsim.onnx_simplifier import apply_low_rank_compensation_cpp


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

    Delegates to the verified C++ port
    (:func:`onnxsim.apply_low_rank_compensation_cpp`), which shares this
    function's own full parameter set (``rank``) exactly -- no
    compatibility gap. That port's own SVD is a hand-rolled Jacobi
    implementation (no LAPACK dependency), so individual singular
    vectors/values are not expected to match sign-for-sign, but the
    reconstructed rank-``r`` correction is basis- and sign-invariant
    (Eckart-Young uniqueness), so results track this function's own former
    implementation closely.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    return apply_low_rank_compensation_cpp(float_model, quantized_model, rank)
