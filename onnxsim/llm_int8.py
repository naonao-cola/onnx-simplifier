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

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors

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

    This pure-Python implementation has been retired in favor of the
    verified-bit-exact C++ port -- this is now a thin alias for
    :func:`onnxsim.apply_llm_int8_cpp`
    (``onnxsim/llm_int8_entry.cpp``'s own ``ApplyLlmInt8``), forwarding
    every argument unchanged. Exact (bit-for-bit) parity was verified
    against this function's own pre-alias implementation across
    MatMul/Gemm/transB-Gemm/biased-Gemm, banker's-rounding ties, the
    outlier-threshold boundary, multi-batch calibration, every skip shape
    (no/all outliers, non-constant weight, rank-3 activation, pre-18
    opset, empty calibration), and thresholds around 6.0 -- see
    tests/test_llm_int8_cpp.py -- before this alias was made. Imported
    lazily (inside the function body, not at module scope) to avoid a
    circular import: ``onnxsim.onnx_simplifier`` already imports from
    this module, so importing it back at module load time here would
    deadlock the import machinery.
    """
    from onnxsim.onnx_simplifier import apply_llm_int8_cpp

    return apply_llm_int8_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        outlier_threshold=outlier_threshold,
        epsilon=epsilon,
        providers=providers,
    )
