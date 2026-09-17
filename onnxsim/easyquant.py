"""EasyQuant (Wu, Judd, Isaev, Micikevicius, 2020, "EasyQuant: Post-training
Quantization via Scale Optimization", https://arxiv.org/abs/2006.16669).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (EasyQuant's own
reference implementation quantizes live framework tensors with no ONNX
export path).

Every scale-calibration routine already in onnxsim (:func:`onnxsim.
calibrate`'s ``"minmax"``/``"entropy"``/``"mse"`` methods) picks a
quantization range from **one tensor's own observed distribution alone** --
a histogram, or a simple min/max -- with no knowledge of what that tensor
actually feeds into. EasyQuant's own idea is different in kind: instead of
asking "what threshold best represents this tensor's own values", ask "what
scale, for this weight and this activation *together*, makes the actual
downstream layer's quantized output (``X_q @ W_q^T`` for a MatMul/Gemm)
closest to the real float output" -- directly optimizing the metric that
matters (the layer's own output), the same reconstruction-error framing
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.quantease` already
use for weight *rounding*, but applied here to the **scale** itself for a
plain W8A8 (INT8 weight, INT8 activation) quantizer, with no gradient
descent or Hessian at all -- just a **coordinate-descent grid search**:

1. Start from an ordinary per-output-channel weight scale and per-tensor
   activation scale (each tensor's own ``max(abs(.)) / 127``, the same
   starting point :func:`onnxsim.calibrate`'s ``"minmax"`` method would
   give).
2. **Weight step**: holding the activation scale fixed, search a small grid
   of candidate multipliers around each output channel's own current scale,
   picking whichever minimizes that channel's own quantized-output MSE
   against the float output -- exact and independent per output channel,
   since (for ``Y = X @ W^T``) column ``n`` of ``Y`` depends only on row
   ``n`` of ``W``, never on any other channel.
3. **Activation step**: holding the (now updated) weight scale fixed,
   search a small grid of candidate multipliers on the single activation
   scale, picking whichever maximizes the *whole* quantized layer output's
   cosine similarity against the float output -- the paper's own metric,
   and not separable per-channel the way the weight step is, so this step
   evaluates the full output.
4. Repeat 2-3 for a small, fixed number of rounds (each round only ever
   improves or holds the previous round's own chosen objective, since a
   grid search always considers "no change" as a candidate).

This module's own honest simplification of the paper: EasyQuant's own
reference procedure searches activation scale per-tensor exactly as here,
but this module's *weight* step decomposes the paper's own overall
cosine-similarity objective into independent per-channel MSE minimization
(justified above -- the decomposition is exact, not approximate, since
column ``n``'s error genuinely depends only on row ``n`` of ``W``) rather
than jointly re-evaluating whole-output cosine similarity for every
candidate combination of all channels at once (combinatorially
infeasible for a grid search) -- this module does not claim its weight
step reproduces the paper's own exact search procedure, only the same
data-driven, output-aware spirit.

Quantization itself is applied as a **float32 round-trip** (quantize then
immediately dequantize) exactly the same simplification :mod:`onnxsim.
attention_quantization`'s own per-token INT8 quantization already makes --
the weight side is folded directly into a new float32 initializer (no new
graph nodes needed, the same pattern every weight-only ``quantize_*``
function in this repo uses), while the activation side needs
``Div``/``Round``/``Clip``/``Mul`` nodes inserted at graph-run time (since
it's a runtime tensor, not a constant). This module has no lower-than-
float32 arithmetic ONNX op to express genuine ``int8 x int8`` execution in
anyway -- the same limitation :mod:`onnxsim.zeroquant`'s own docstring
names for onnxsim's other float-simulated activation-quantization passes.

**Scope note**: only ``MatMul`` and "vanilla" ``Gemm`` (``transA=0``,
``alpha=1``, and ``beta=1`` when a bias is present) with a constant 2-D
float32 weight are matched (via :func:`onnxsim.llm_int8._match_matmul_like`,
already shared with :mod:`onnxsim.llm_int8`) -- ``Conv`` is left untouched,
a scope decision consistent with several other onnxsim modules that target
only MatMul/Gemm.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_easyquant_cpp


def apply_easyquant(
    float_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 3,
    num_candidates: int = 21,
    search_span: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """W8A8-quantizes every matched MatMul/"vanilla" Gemm layer, choosing
    both the per-output-channel weight scale and the per-tensor activation
    scale via EasyQuant's own coordinate-descent search against real
    calibration activations -- see this module's own docstring for the
    technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param calibration_data: representative input batches to run the scale
            search against -- see :func:`onnxsim.correct_bias`'s own
            parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: coordinate-descent rounds (weight step, then
            activation step) -- each round only ever improves or holds the
            previous round's own chosen scales
    :param num_candidates: grid resolution per coordinate-descent step
    :param search_span: candidate multipliers span
            ``[1 - search_span, 1 + search_span]`` around each step's
            current scale
    :param providers: onnxruntime execution providers to run
            ``float_model`` on when capturing calibration activations
    :returns: ``float_model`` with every matched layer's weight replaced by
            its quantize-dequantize round-tripped float32 version, and a
            ``Div``/``Round``/``Clip``/``Mul`` round-trip inserted before
            its activation input -- layers with a non-constant, non-2-D
            weight, an activation with no feature axis at all (rank < 2;
            a higher-rank ``[batch, seq, K]`` one is flattened to
            ``[batch * seq, K]``, which is exact), or whose activation's
            feature dimension doesn't match the weight's own reduction
            size, are
            left untouched

    Delegates to :func:`onnxsim.apply_easyquant_cpp` (the verified C++
    port, which has full parameter parity with this function -- see
    ``easyquant_entry.h`` for its own scope note); this pure-Python name
    is kept only for backward compatibility with existing callers.
    """
    return apply_easyquant_cpp(
        float_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_iterations=num_iterations,
        num_candidates=num_candidates,
        search_span=search_span,
        providers=providers,
    )
