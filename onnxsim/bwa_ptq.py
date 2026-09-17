"""Binary Weight-Activation PTQ (Song, Wang, Wang, Yang, Zhang, 2025,
"Achieving binary weight and activation for LLMs using Post-Training
Quantization", ACL 2025 Findings, https://arxiv.org/abs/2504.05352, code at
https://github.com/JimmyCrave/LLM-PTQ-binarization). onnxsim ports the
*algorithm*, not any framework's code, per the same rationale as
:mod:`onnxsim.billm`/:mod:`onnxsim.gptq` (the paper's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

The paper's own configuration name is ``W(1+1)A(1x4)``. This module ports
its **weight** side only -- the ``W(1+1)`` piece -- and documents why the
``A(1x4)`` activation side is deliberately out of scope, see below.

**Weight side, W(1+1): Hessian-aware two-scale binary grouping.** Read
:mod:`onnxsim.billm` first -- both modules binarize an ordinary dense
float32 MatMul/Gemm weight straight from the float model (not a
rounding-refinement lever on an already-INT4-quantized model the way
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` are), and
both use a calibration-data Hessian. Where this module differs from BiLLM
is the *shape* of the extra bit each weight gets beyond its sign:

- :mod:`onnxsim.billm` spends its extra bit *structurally*, on whole
  Hessian-salient **columns**: a small, data-dependent fraction of columns
  get a second, *additive* binarization level (``W ~= B1 + B2``, a residual
  correction), while every other column stays a single flat sign*scale.
  Most of a BiLLM-quantized layer is exactly 1 bit/element; a few columns
  cost close to 2.
- This module (and the source paper) instead spends the extra bit
  *uniformly*, on every weight, as a **selector between two candidate
  scales** for its own group (a group being ``group_size`` contiguous
  elements of the reduction axis within one output channel row, matching
  the group convention :func:`onnxsim.quantize_weight_only_int4` already
  uses) rather than as an additive correction. Concretely, for each group,
  the paper's own "EM-based quantization scheme" alternates:

    1. (M-step) each of the group's two scale candidates is set to the
       Hessian-diagonal-weighted mean of ``|w|`` over whichever elements
       are currently assigned to it (the closed-form minimizer of a
       weighted-L2 binary fit, the same argument
       :func:`onnxsim.billm._binary` uses unweighted);
    2. (E-step) every element is reassigned to whichever of the two scales
       gives it lower Hessian-weighted squared error;

  repeated to convergence (a 2-component weighted Lloyd/k-means run on
  ``|w|``, seeded from a median split). Every weight ends up encoded as
  ``sign(w) * scale[group_select]`` -- exactly 1 sign bit + 1 group-select
  bit/element, no column-level structure and no additive residual term.
  Both this module and BiLLM are honest, different simplifications of
  their own papers' more elaborate schemes (see :mod:`onnxsim.billm`'s own
  docstring for its own documented simplification) -- neither supersedes
  the other; this module exists to make PTQ's uniform-two-scale family
  available alongside BiLLM's own salient-column-residual family.

  This is also distinct from :mod:`onnxsim.pb_llm` (keeps a salient
  *column fraction* at a much higher bit-width, e.g. INT8, and only
  binarizes the rest) -- this module never raises any weight above ~1 bit,
  uniformly, everywhere.

**Activation side, A(1x4): deliberately not ported.** The paper's own
activation scheme decomposes an INT4-quantized activation code into
``4 x INT1`` "bit-planes" (``q = sum_{k=0}^{3} 2**k * bit_k``) purely so a
bit-serial accelerator can compute the matmul via four cheap binary
popcount passes instead of one INT4 pass. Algebraically this decomposition
reconstructs *exactly* the same real number an ordinary calibrated INT4
quantizer already would -- it changes how a specific accelerator computes
the matmul, not the quantized value onnxsim's own QDQ graph would ever
represent. Since onnxsim doesn't emit bit-serial popcount kernels, this
module's activations are left for onnxsim's existing calibrated INT4/INT8
activation quantizers (e.g. :func:`onnxsim.quantize_static`) to handle,
exactly as :mod:`onnxsim.billm`/:mod:`onnxsim.pb_llm` also only handle the
weight side. Also not ported: the paper's own error-aware smoothing of the
weight/activation scaling factors (a joint calibration step tying the two
sides together) -- out of scope alongside the activation side it smooths.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_bwa_ptq_cpp


def apply_bwa_ptq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    group_size: int = 128,
    max_em_iters: int = 10,
    skip_names: Optional[Iterable[str]] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Binarizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight to exactly 1 sign bit + 1 group-select bit/element,
    using Hessian-weighted two-scale binary EM -- see this module's own
    docstring for the technique and its documented scope (weight side
    only; the source paper's activation-side bit-plane decomposition is
    numerically equivalent to ordinary INT4 activation quantization, see
    :func:`onnxsim.quantize_static`).

    Needs real calibration activations to compute each layer's Hessian
    diagonal (the per-input-channel importance weight the EM step uses),
    the same requirement :mod:`onnxsim.billm`/:mod:`onnxsim.gptq` have.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see :func:`onnxsim.gptq.apply_gptq`'s
            own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param group_size: contiguous reduction-axis elements sharing one pair
            of candidate scales -- matches
            :func:`onnxsim.quantize_weight_only_int4`'s own group
            convention
    :param max_em_iters: upper bound on EM iterations per group (the loop
            already stops early once the assignment stops changing)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Cast(Sign), Add(Scale0, Mul(Cast(GroupSelect), Sub(Scale1,
            Scale0))))`` feeding the original MatMul/Gemm node -- ordinary
            ONNX ops only, opset 11+. Layers with a non-constant, non-2-D,
            or non-float32 weight, or whose activation input has no
            feature axis at all (rank < 2), are left untouched; a
            higher-rank ``[batch, seq, K]`` activation is flattened to
            ``[batch * seq, K]``, which is exact.

    Delegates to :func:`onnxsim.apply_bwa_ptq_cpp` (the verified C++
    port), which emits the exact same node pattern this function's own
    docstring describes. This pure-Python name is kept for backward
    compatibility with existing callers; ``skip_names`` is not supported
    by the C++ port (which has no established way to plumb a denylist
    through its own candidate search), so a non-empty value raises
    ``ValueError`` rather than being silently ignored.
    """
    if skip_names:
        raise ValueError(
            "apply_bwa_ptq now delegates to apply_bwa_ptq_cpp, which does "
            "not support skip_names; filter candidates yourself, or call "
            "apply_bwa_ptq_cpp directly"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_bwa_ptq_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        group_size=group_size,
        max_em_iters=max_em_iters,
        providers=providers,
    )
