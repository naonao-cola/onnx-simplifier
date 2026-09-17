"""FlexRound -- Learnable Rounding based on Element-wise Division for
Post-Training Quantization (Lee et al., 2023, ICML 2023,
https://arxiv.org/abs/2306.00317). Fourth onnxsim-native PTQ technique
alongside :mod:`onnxsim.adaround`, :mod:`onnxsim.gptq`, and
:mod:`onnxsim.awq`, each pulling a different lever on the exact same target
scheme (:func:`onnxsim.quantize_weight_only_int4`'s block-wise symmetric
INT4): AdaRound perturbs each weight element's own rounding *additively*
(:mod:`onnxsim.adaround`'s rectified-sigmoid ``delta``); GPTQ quantizes
input channels one at a time and propagates each one's rounding error into
every not-yet-quantized channel via a Hessian; AWQ rescales whole input
channels before quantizing. FlexRound instead reparametrizes the *divisor
itself*, multiplicatively, per weight element.

The paper's own formula (Eq. 1-2, per-tensor/per-channel uniform PTQ, a
linear layer): the quantized weight is ``W_hat = s1 * round(W / S)``, where
the *effective* per-element divisor ``S = s1 (x) S2 (x) s3`` ((x) = the
paper's element-wise product) decomposes a common quantization grid size
``s1`` (what a plain round-to-nearest quantizer already picks -- a scalar
or a per-output-channel vector in the paper), an element-wise learnable
correction ``S2`` (same shape as ``W``), and a per-output-channel learnable
correction ``s3``. (A 2D convolution's ``S`` gets a further per-input-
channel factor ``s4``; not applicable here since this module, like
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.awq`, only
targets MatMul/Gemm.) All of ``S2``/``s3`` are initialized to 1, so
optimization starts exactly at round-to-nearest and only departs from it as
gradient descent (Adam, straight-through through ``round()`` -- Bengio
et al., 2013 -- the same "post-hoc adjustment from real activations" style
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.awq` all share)
finds it worthwhile against ``||W X - W_hat X||^2`` on real calibration
activations.

The paper's own reciprocal-rule argument (Proposition 3.1) is precisely why
this differs from AdaRound's additive perturbation: because ``S`` divides
rather than adds, ``d(W/S)/dS = -W/S^2`` is proportional to ``W`` itself,
so (under a straight-through gradient) a large-magnitude weight naturally
receives a proportionally larger nudge and a small-magnitude one a
proportionally smaller one. AdaRound's additive ``delta``, by contrast, is
squashed into the same fixed range regardless of the weight's own
magnitude, so it costs a large weight the same absolute nudge as a small
one -- the paper's stated reason FlexRound's own reparametrization scales
better to heavy-tailed weight (or, in the paper's activation-quantization
experiments, activation) magnitude distributions than a fixed-range
additive perturbation does.

**What's ported vs. not, relative to the paper:**

- The paper jointly *learns* ``s1`` (the quantization grid size) alongside
  ``S2``/``s3``. This module does not: like every other onnxsim PTQ pass
  targeting :func:`onnxsim.quantize_weight_only_int4`'s output, it keeps
  the block scale ``quantized_model`` already computed completely
  unchanged and only rewrites *which integer* each element rounds to --
  ``s1`` here is fixed at that pre-existing per-(block, output channel)
  scale rather than optimized. This is the same scope restriction
  :mod:`onnxsim.adaround` and :mod:`onnxsim.gptq` both make, for the same
  reason: the scale is a shared tensor :func:`onnxsim.quantize_weight_only_int4`
  already committed to the graph, not a free parameter this pass owns.
- ``S2`` (element-wise) and ``s3`` (per-output-channel) are both
  implemented, matching the paper's own linear-layer formula (Eq. 2's
  ``S = s1 (x) S2 (x) s3``) exactly. The paper's ``s4`` (an additional
  per-input-channel factor) only applies to its 2D convolution formula,
  which is out of scope here for the reason above.
- The paper leaves the positivity constraint on ``S2``/``s3`` unspecified
  beyond "positive and learnable." This module enforces it by optimizing
  in log-space (``S2 = exp(v2)``, ``s3 = exp(v3)``, ``v2``/``v3``
  initialized to 0) -- a standard reparametrization for a
  positive-multiplicative learned quantity, and, as a useful side effect,
  algebraically simplifies the gradient (``d(S)/d(v2) == S`` and
  ``d(S)/d(v3) == S``; see ``onnxsim/flexround_entry.cpp``'s own
  ``OptimizeDivisor`` for the C++ port :func:`apply_flexround` delegates
  to).
- The paper's own forward pass applies ``round()`` every iteration
  (straight-through for the backward pass). This module instead keeps the
  *continuous* relaxation ``clip(W / S, n_min, n_max)`` throughout
  optimization and rounds once at the very end -- the same structure
  :mod:`onnxsim.adaround` uses for its own relaxation. Unlike AdaRound,
  no regularization/annealing schedule pulls this relaxation toward a hard
  decision (the paper's own formulation doesn't have one either): the
  reconstruction loss alone shapes ``S2``/``s3`` from start to finish.
- The paper also explores W4A4 / activation-quantization variants (jointly
  reparametrizing the activation's own quantizer) and block-wise
  reconstruction for large language models; this module only implements
  weight-only quantization with plain, whole-layer reconstruction --
  matching every other onnxsim PTQ pass (see :mod:`onnxsim.adaround`'s own
  docstring) and this repo's existing calibration-driven-pass style.

No calibration data beyond what :mod:`onnxsim.calibration` already
provides, activation quantization, or gradient framework other than what
this module implements itself is required -- everything here is plain
numpy, matching :mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/
:mod:`onnxsim.awq`'s shared style.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_flexround(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.05,
    log_clip: float = 4.0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes FlexRound-style learnable-division rounding for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``, using
    real activations captured from ``float_model``. See this module's own
    docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to optimize the
            divisor on. Each batch is a ``{input_name: np.ndarray}`` dict
            matching ``float_model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative optimization target than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per layer
    :param learning_rate: Adam learning rate for the log-space element-wise
            (``S2``) and per-output-channel (``s3``) divisor corrections.
            The paper's own hyperparameter tables tune this per model/layer
            (values from ``1e-6`` to ``1e-1`` appear across its
            experiments) -- this reparametrization's own sensitivity to
            learning rate, not just a quirk of this port; too high
            overshoots past round-to-nearest's own optimum, too low never
            leaves it within a practical iteration budget
    :param log_clip: clamps ``log(S2)``/``log(s3)`` to
            ``[-log_clip, log_clip]`` after every step -- a numerical
            safety bound (not part of the paper's own formulation) keeping
            the learned divisor from drifting so far from the original
            scale that ``round(W / S)`` becomes numerically degenerate
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its FlexRound-optimized codes (same
            shape, dtype, and scale -- only which integer each element
            rounds to changes)

    This entry point is a thin alias for the verified C++ port
    :func:`onnxsim.apply_flexround_cpp` (``onnxsim/flexround_entry.cpp``'s
    own ``ApplyFlexround``), forwarding every argument unchanged. **Not**
    bit-exact with this function's own former in-process numpy loop (this
    module's own former ``_optimize_divisor``, now removed as dead code --
    no other module imported it): FlexRound's reciprocal parametrization is
    measurably MORE sensitive to floating-point summation-order/libm
    differences than e.g. AdaRound's own rectified-sigmoid relaxation is,
    since its gradient divides by the effective divisor squared -- see
    ``flexround_entry.h``'s own accepted numerical scope note and
    tests/test_flexround_cpp.py for exactly how closely (or not) the C++
    port tracks what this function's own numpy loop used to compute.
    Imported lazily (inside the function body, not at module scope) to
    avoid a circular import: ``onnxsim.onnx_simplifier`` already imports
    from this module, so importing it back at module load time here would
    deadlock the import machinery.
    """
    from onnxsim.onnx_simplifier import apply_flexround_cpp

    return apply_flexround_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        log_clip=log_clip,
        providers=providers,
    )
