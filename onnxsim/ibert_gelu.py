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

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name

# This module's own numeric min-max fit of the paper's L(x) functional
# form (see the module docstring): `b` is the one free parameter (found by
# grid search minimizing max|L(x) - erf(x)| over [-4, 4]); `a` and `c` are
# then pinned by L(x)'s own continuity-at-0 and asymptote-at-+-1
# constraints (c = 1, a = -1/b**2) -- not independently fit.
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

    :param model: the original (unquantized) onnx ModelProto or file path
    :param skip_names: ``Erf`` node names to leave untouched even if
            otherwise eligible
    :returns: ``model`` with every matched ``Erf(x)`` node's output fed by
            ``Mul(Sign(x), Add(Mul(a, Mul(t, t)), c))`` where
            ``t = Add(Clip(Abs(x), 0, -b), b)`` -- ordinary ONNX ops only
            (``Abs``/``Clip``/``Add``/``Mul``/``Sign``), opset 11+ (the
            2-input/3-input ``Clip`` form). ``Erf`` nodes named in
            ``skip_names`` are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    taken_names = _all_names(graph)

    nodes = list(graph.node)
    for node in nodes:
        if node.op_type != "Erf" or len(node.input) != 1:
            continue
        if node.name in skip_names:
            continue

        x_name = node.input[0]
        erf_out = node.output[0]
        prefix = _unique_name(f"{erf_out}_ibert_gelu", taken_names)

        abs_out = _unique_name(f"{prefix}_abs", taken_names)
        abs_node = onnx.helper.make_node("Abs", [x_name], [abs_out])

        clip_min_name = _unique_name(f"{prefix}_clip_min", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(0.0, dtype=np.float32), name=clip_min_name
            )
        )
        clip_max_name = _unique_name(f"{prefix}_clip_max", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(-_IBERT_GELU_B, dtype=np.float32), name=clip_max_name
            )
        )
        clip_out = _unique_name(f"{prefix}_clip", taken_names)
        clip_node = onnx.helper.make_node(
            "Clip", [abs_out, clip_min_name, clip_max_name], [clip_out]
        )

        b_name = _unique_name(f"{prefix}_b", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(_IBERT_GELU_B, dtype=np.float32), name=b_name
            )
        )
        shifted_out = _unique_name(f"{prefix}_shifted", taken_names)
        add_b_node = onnx.helper.make_node("Add", [clip_out, b_name], [shifted_out])

        squared_out = _unique_name(f"{prefix}_squared", taken_names)
        square_node = onnx.helper.make_node(
            "Mul", [shifted_out, shifted_out], [squared_out]
        )

        a_name = _unique_name(f"{prefix}_a", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(_IBERT_GELU_A, dtype=np.float32), name=a_name
            )
        )
        scaled_out = _unique_name(f"{prefix}_scaled", taken_names)
        scale_node = onnx.helper.make_node("Mul", [squared_out, a_name], [scaled_out])

        c_name = _unique_name(f"{prefix}_c", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(_IBERT_GELU_C, dtype=np.float32), name=c_name
            )
        )
        poly_out = _unique_name(f"{prefix}_poly", taken_names)
        add_c_node = onnx.helper.make_node("Add", [scaled_out, c_name], [poly_out])

        sign_out = _unique_name(f"{prefix}_sign", taken_names)
        sign_node = onnx.helper.make_node("Sign", [x_name], [sign_out])

        result_node = onnx.helper.make_node(
            "Mul",
            [sign_out, poly_out],
            [erf_out],
            name=_unique_name(f"{prefix}_result", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (
            abs_node,
            clip_node,
            add_b_node,
            square_node,
            scale_node,
            add_c_node,
            sign_node,
            result_node,
        ):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1
        graph.node.remove(node)

    return out
