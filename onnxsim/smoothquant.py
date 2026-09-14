"""SmoothQuant (Xiao et al., 2022, "SmoothQuant: Accurate and Efficient
Post-Training Quantization for Large Language Models",
https://arxiv.org/abs/2211.10438). The technique behind ``llm-compressor``'s
``SmoothQuantModifier`` -- onnxsim ports the *algorithm*, not that code, per
the same rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.hqq`
(``llm-compressor`` quantizes live PyTorch models with no ONNX export path).

Every weight-only technique already in onnxsim (``quantize_weight_only_int4``
and everything built on it, plus :mod:`onnxsim.hqq`, :mod:`onnxsim.nf4`)
leaves activations in float -- only the weight is ever quantized. Quantizing
*activations* too (W8A8, as ``quantize_static``/``quantize_qoperator_gemm``
already do) hits a problem weight-only quantization never sees: in
transformer activations, a handful of feature channels can be an order of
magnitude larger than the rest, consistently, across tokens and inputs. A
single per-tensor (or even per-token) activation quantization range has to
cover those outlier channels, so it wastes most of its resolution on the
sea of ordinary channels below them. Weights, by contrast, are comparatively
flat and easy to quantize.

SmoothQuant's insight: this difficulty can be *migrated* rather than
endured. For a MatMul/Gemm ``Y = X @ W``, scaling one input channel of ``X``
down by ``s_j`` and the matching row of ``W`` up by the same ``s_j`` leaves
``Y`` mathematically unchanged (``(X / s) @ (W * s) == X @ W``) while
moving quantization difficulty between the two operands -- a large ``s_j``
shrinks that channel's activation range (easier to quantize) at the cost of
expanding the corresponding weight row's range (harder, but weights have
range to spare). Per-channel, the paper sets

    s_j = max(|X_j|) ** alpha / max(|W_j|) ** (1 - alpha)

where ``X_j``/``W_j`` are channel ``j``'s values across the calibration set
and the reduction dimension respectively, and ``alpha`` (the paper's
"migration strength", typically ``0.5``) trades off how much of the
difficulty moves: ``alpha -> 1`` pushes activations to a uniform, trivially
quantizable range at the weight's expense; ``alpha -> 0`` does the reverse.
Unlike :mod:`onnxsim.awq` (which grid-searches its own analogous per-channel
scale against measured INT4 reconstruction error), SmoothQuant's ``alpha``
is not searched here -- it is a single fixed global hyperparameter, matching
the paper's own practice of picking one value per model family from a light
validation sweep, not optimizing it per layer.

This module only performs the *migration* (:func:`apply_smoothquant`
returns a float model, provably equivalent to the input up to floating-point
rounding -- no quantization happens here at all): it multiplies every
matched MatMul/Gemm's weight columns by ``s`` in place, and inserts a new
``Mul`` node dividing that layer's activation input by the same ``s`` right
before it. The result is meant to be fed to a W8A8 quantizer afterwards
(e.g. :func:`onnxsim.quantize_static`/:func:`onnxsim.quantize_qoperator_gemm`)
-- exactly how the SmoothQuant paper's own reference implementation is used,
as a pre-conditioning pass ahead of a separate, ordinary PTQ quantizer.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(x_name, w_name, weight_transposed)`` or
    ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[0], node.input[1], False
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
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[0], node.input[1], weight_transposed
    return None


def apply_smoothquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Migrates activation quantization difficulty into the weight for
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight and a
    plain 2-D activation input, using real calibration activations. See
    this module's own docstring for the technique. Returns a float model --
    pass the result to a W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`)
    to actually quantize it.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation range on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative migration than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength (paper's own default, ``0.5``,
            splits difficulty evenly on a log scale; ``1.0`` pushes it
            entirely onto the weight, ``0.0`` entirely onto the activation)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing ``s``, avoiding a divide-by-zero
            on an all-zero channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight columns rescaled
            by ``s`` in place and a new ``Mul`` node inserted before it
            applying ``1 / s`` to its activation input; layers with a
            non-constant, non-2-D weight, or whose activation input isn't a
            plain 2-D tensor matching the weight's reduction dimension, are
            left untouched

    This pure-Python implementation has been retired in favor of the
    verified-bit-exact C++ port -- this is now a thin alias for
    :func:`onnxsim.apply_smoothquant_cpp`
    (``onnxsim/smoothquant_entry.cpp``'s own ``ApplySmoothQuant``),
    forwarding every argument unchanged. Exact (bit-for-bit) parity was
    verified against this function's own pre-alias implementation across
    MatMul/Gemm/transB-Gemm/biased-Gemm, shared activations, outlier and
    all-zero channels, empty calibration, and alpha in {0, 0.25, 0.5,
    0.75, 1} -- see tests/test_smoothquant_cpp.py -- before this alias was
    made. Imported lazily (inside the function body, not at module scope)
    to avoid a circular import: ``onnxsim.onnx_simplifier`` already
    imports from this module, so importing it back at module load time here
    would deadlock the import machinery.
    """
    from onnxsim.onnx_simplifier import apply_smoothquant_cpp

    return apply_smoothquant_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        alpha=alpha,
        epsilon=epsilon,
        providers=providers,
    )
