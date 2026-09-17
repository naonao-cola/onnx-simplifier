"""I-BERT (Kim, Gholami, Yao, Mahoney, Keutzer, 2021, ICML 2021, "I-BERT:
Integer-only BERT Quantization", https://arxiv.org/abs/2101.01321) -- the
paper's own "i-GELU" piece. onnxsim ports the *algorithm* (the closed-form
polynomial itself), not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (I-BERT's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

Every other quantizer already in onnxsim targets the same kind of node:
a MatMul/Gemm/Conv whose *weight* (and sometimes activation) gets
quantized, while the *nonlinear* activation functions in between
(``Erf``/``Gelu``, ``Softmax``, ``LayerNorm``) stay exactly as the float
model computed them -- QDQ quantization (:func:`onnxsim.quantize_static`
and friends) wraps a calibrated integer range *around* those nonlinear
ops without changing what they compute internally. I-BERT's own
contribution is different in kind: it replaces the nonlinear function's
own *computation* with an integer-arithmetic-friendly polynomial
approximation, so a genuinely integer-only accelerator (no floating-point
unit at all) can evaluate it -- not just quantize its input/output like
every other technique in this repo does.

This module ports the paper's own **i-GELU** piece specifically: GELU is
almost universally exported as ``0.5 * x * (1 + Erf(x / sqrt(2)))``
(BERT/RoBERTa/GPT-family transformers' own standard decomposition, present
as a plain ``Erf`` node in the exported graph), and ``Erf`` is the one
piece of that formula with no polynomial-friendly closed form. The paper's
own idea: fit a second-order polynomial to ``erf`` of the shape

    L(x) = sign(x) * (a * (clip(|x|, max=-b) + b)**2 + c)

(clipped so the polynomial only has to fit the region where ``erf`` is not
already saturated at +-1), and substitute ``L(x)`` for ``Erf(x)``
everywhere. Two of the three coefficients are fixed by ``L``'s own
boundary behavior, not free: continuity at ``x = 0`` (where ``sign``
itself jumps from -1 to +1) forces ``c = -a * b**2``, and matching
``erf``'s own asymptote of +-1 past the clip point forces ``c = 1`` --
together pinning ``a = -1 / b**2`` for *any* choice of ``b``. This module
fits the one remaining free parameter, ``b``, by a numeric min-max search
minimizing ``L``'s worst-case absolute error against the true ``erf`` over
``[-4, 4]`` (``b ~= -1.691``, ``a ~= -0.3495``, max error ~= 0.021) --
this repo's own from-scratch numeric fit of the paper's *functional form*
(the same "port the algorithm's shape, verify the actual behavior rather
than trust an unverifiable literature constant" practice this project has
followed since its own MSE-calibration threshold and BWA-PTQ EM search),
not a transcription of the paper's own reported coefficients, which this
module does not claim to reproduce exactly. Because every operation in
``L(x)`` (``Abs``, ``Clip``, ``Add``, ``Mul``, ``Sign``) is itself already
piecewise-linear or low-order-polynomial, this is the piece an
integer-only accelerator can evaluate with fixed-point arithmetic instead
of a hardware ``erf``/transcendental unit -- the paper's own point.

**Deliberately not ported**: I-BERT's other two pieces, integer-only
Softmax (a polynomial approximation of ``exp`` plus an integer-only
iterative reciprocal for the normalization) and integer-only LayerNorm
(an integer-only iterative reciprocal square root) -- both need an
iterative fixed-point division/rsqrt loop with a paper-specified iteration
count, materially more involved than i-GELU's own single closed-form
polynomial, and are left as a follow-up rather than risked here without
the same level of confidence in the exact reproduced constants.

This module represents ``L(x)`` using ordinary float32 ONNX ops (the same
simplification :mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4` already
make for their own packed-bit formats): the *polynomial shape* the paper
introduces is reproduced exactly, but the actual fixed-point/dyadic
integer arithmetic a real integer-only accelerator would use to evaluate
it is not -- onnxsim has no lower-than-float32 arithmetic ONNX op to
express that reproduction in anyway (the same reason
:func:`onnxsim.quantize_static`'s own QDQ nodes still compute in float32
between a `QuantizeLinear`/`DequantizeLinear` pair).
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import onnx

from onnxsim.onnx_simplifier import apply_ibert_gelu_cpp

# This module's own numeric min-max fit of the paper's L(x) functional
# form (see the module docstring): `b` is the one free parameter (found by
# grid search minimizing max|L(x) - erf(x)| over [-4, 4]); `a` and `c` are
# then pinned by L(x)'s own continuity-at-0 and asymptote-at-+-1
# constraints (c = 1, a = -1/b**2) -- not independently fit. Kept here as
# documentation of the derivation; the actual computation now happens in
# the C++ port (``passes/ibert_gelu.h``), which hardcodes the same
# constants.
_IBERT_GELU_B = -1.69148
_IBERT_GELU_A = -1.0 / (_IBERT_GELU_B**2)
_IBERT_GELU_C = 1.0


def apply_ibert_gelu(
    model: Union[str, onnx.ModelProto],
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Replaces every standalone ``Erf`` node (the ``erf`` in GELU's
    standard ``0.5 * x * (1 + Erf(x / sqrt(2)))`` export decomposition,
    among any other use) with I-BERT's own closed-form polynomial
    approximation -- see this module's own docstring for the technique and
    its derivation. Needs no calibration data: the polynomial's own
    coefficients are fixed by the paper, not fit to any particular
    model's data.

    Delegates to the verified C++ port (:func:`onnxsim.apply_ibert_gelu_cpp`,
    ``passes/ibert_gelu.h``), which builds the exact same five ONNX ops
    from the exact same three float32 constants -- numerically identical
    up to onnxruntime's own float32 evaluation.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param skip_names: ``Erf`` node names to leave untouched even if
            otherwise eligible. **Not supported by the C++ port** -- passing
            a non-empty value raises ``NotImplementedError`` rather than
            silently ignoring it (unlike a numeric divergence, silently
            rewriting a node the caller explicitly asked to protect could
            break correctness-critical code).
    :returns: ``model`` with every matched ``Erf(x)`` node's output fed by
            ``Mul(Sign(x), Add(Mul(a, Mul(t, t)), c))`` where
            ``t = Add(Clip(Abs(x), 0, -b), b)`` -- ordinary ONNX ops only
            (``Abs``/``Clip``/``Add``/``Mul``/``Sign``), opset 11+ (the
            2-input/3-input ``Clip`` form).
    """
    if skip_names:
        raise NotImplementedError(
            "apply_ibert_gelu's C++ backend does not support skip_names; "
            "call apply_ibert_gelu_cpp directly if you don't need it, or "
            "filter the model's own Erf nodes before/after calling this "
            "function."
        )
    return apply_ibert_gelu_cpp(model)
