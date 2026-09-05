"""I-BERT (Kim, Gholami, Yao, Mahoney, Keutzer, 2021, ICML 2021, "I-BERT:
Integer-only BERT Quantization", https://arxiv.org/abs/2101.01321) -- the
paper's own **integer-only Softmax** piece, a follow-up to
:mod:`onnxsim.ibert_gelu` (i-GELU). That module's own docstring names this
one explicitly: "Deliberately not ported [there]: integer-only Softmax...
left as a follow-up". Read :mod:`onnxsim.ibert_gelu` first -- the framing,
scope discipline, and "port the algorithm, not any framework's code"
rationale are identical here.

A plain ``Softmax`` node needs two things a genuinely integer-only
accelerator (no floating-point unit at all) can't evaluate directly: the
transcendental ``exp`` in its numerator, and a division in its
normalization. I-BERT's own contribution replaces both with
integer-arithmetic-friendly substitutes:

1. **exp, via the same polynomial-approximation idea as i-GELU's own erf
   fit, plus an exact power-of-two rescale.** After the usual
   numerical-stability max-subtraction (``x <= 0`` for every element),
   decompose each ``x`` as ``x = -z*ln2 + p`` with ``z = floor(-x / ln2)``
   a *non-negative integer* and ``p`` the remainder, ``p in (-ln2, 0]``.
   Then ``exp(x) = exp(p) * exp(-z*ln2) = exp(p) * 2**(-z)``: ``exp(p)``
   only ever needs fitting over the single fixed short interval
   ``(-ln2, 0]`` (a good target for a low-order polynomial, exactly
   i-GELU's own move), and ``2**(-z)`` -- an *integer* power-of-two -- is a
   trivial bit-shift in genuine fixed-point hardware, no transcendental
   evaluation needed at all (the paper's own point: the whole reason
   ``exp`` is hard to approximate directly over its *full* domain is that
   naive polynomial fits degrade badly far from the fitting point, but
   restricting the fit to one fixed short interval and handling the rest
   via an exact multiplicative rescale sidesteps that entirely).

2. **An integer-only iterative reciprocal for the final normalization
   (dividing by the row sum).** This module does **not** port that piece:
   like i-GELU's own explicitly-deferred integer-only Softmax/LayerNorm
   (see that module's docstring), a paper-specified fixed-point
   Newton-Raphson iteration is materially more involved than a single
   closed-form polynomial substitution, and onnxsim has no
   lower-than-float32 arithmetic ONNX op to express its actual fixed-point
   behavior in even if ported (the same limitation i-GELU's own docstring
   already names). The final division uses a plain ``Div`` node instead --
   mathematically the same value the paper's own iteration converges to,
   just not an integer-only op. This module's honest scope is the exp
   piece only.

This module's own numeric fit of ``exp(p)`` (a good-faith, numerically
verified reproduction of the paper's own described mechanism -- fit a
polynomial of the form ``a*(p+b)**2+c`` to ``exp(p)`` over ``p in
[-ln2, 0]``, not a transcription of the paper's own reported coefficients,
which this module does not claim to reproduce exactly): unlike i-GELU's own
``L(x)``, there is no ``sign``-driven boundary here, only one exact
constraint -- ``p=0`` corresponds to ``z=0`` (the maximum element itself,
where ``exp(0) == 1`` must hold exactly for the max-subtracted formulation
to be consistent). Parameterizing the fit directly as a general quadratic
``A*p**2 + B*p + 1`` (any ``a*(p+b)**2+c`` with ``a*b**2+c=1`` is exactly
such a quadratic, and vice versa -- fitting ``(A, B)`` directly avoids a
degenerate blow-up as ``a -> 0`` that the ``(a, b)`` parameterization
suffers), a 2-D numeric min-max search minimizing worst-case *relative*
error against true ``exp`` over ``[-ln2, 0]`` (relative, not absolute,
since ``exp`` spans ``[0.5, 1]`` there and the final softmax ratios are
what matters, not the raw magnitude) finds ``A ~= 0.36118``,
``B ~= 0.9701`` (max relative error ~= 0.22%, verified directly against
``math.exp`` in this module's own test file) -- this repo's own
from-scratch numeric fit of the paper's functional form, the same "port
the algorithm's shape, verify the actual behavior" practice as i-GELU's
own polynomial fit. (For reference against the paper's own reported
constants in its own ``a*(x+b)**2+c`` notation, ``a = A ~= 0.361``,
``b = B/(2A) ~= 1.343``, ``c = 1 - A*b**2 ~= 0.349`` -- close to, but not
claimed to exactly reproduce, the paper's own reported ``0.3585``/
``1.353``/``0.344``.)

**Scope note**: like every op this module inserts, ``Pow(2.0, -z)`` is
represented using an ordinary ONNX ``Pow`` node -- a genuine integer-only
accelerator would implement an integer power-of-two as a bit-shift (no
transcendental evaluation needed at all, which is the paper's own point
about *that* piece specifically), but onnxsim has no lower-than-float32
arithmetic type to express that distinction in an ONNX graph, the same
"polynomial *shape* is reproduced exactly, the actual fixed-point
arithmetic underneath is not" compromise i-GELU's own docstring already
documents for its own ops.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name

# This module's own numeric min-max fit of exp(p) ~= A*p**2 + B*p + 1 over
# p in [-ln2, 0] (see the module docstring): a 2-D grid search minimizing
# max relative error against math.exp, subject to the exact p=0 ->
# exp(0)==1 constraint (the "+1" -- not independently fit).
_LN2 = math.log(2.0)
_IBERT_SOFTMAX_QUAD_A = 0.36118
_IBERT_SOFTMAX_QUAD_B = 0.9701


def apply_ibert_softmax(
    model: Union[str, onnx.ModelProto],
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Replaces every standalone ``Softmax`` node's ``exp``-and-normalize
    computation with I-BERT's own polynomial-plus-power-of-two-rescale
    approximation of ``exp`` (see this module's own docstring for the
    technique, the derivation, and what's honestly not ported -- the
    integer-only reciprocal for the final division). Needs no calibration
    data: the polynomial's own coefficients are fixed by this module's own
    numeric fit, not fit to any particular model's data.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param skip_names: ``Softmax`` node names to leave untouched even if
            otherwise eligible
    :returns: ``model`` with every matched ``Softmax(x, axis=k)`` node
            replaced by the exp-polynomial-plus-rescale-then-normalize
            decomposition described above -- ordinary ONNX ops only
            (``ReduceMax``/``Sub``/``Neg``/``Div``/``Floor``/``Mul``/
            ``Add``/``Pow``/``ReduceSum``), requiring opset 18+ for
            ``ReduceMax``/``ReduceSum``'s axes-as-input form. A ``Softmax``
            node named in ``skip_names``, or present in a model whose
            opset is below 18, is left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    opset_ge_18 = any(
        o.domain in ("", "ai.onnx") and o.version >= 18 for o in out.opset_import
    )
    if not opset_ge_18:
        return out  # ReduceMax/ReduceSum's axes-as-input form needs opset >= 18

    nodes = list(graph.node)
    candidates = [
        n
        for n in nodes
        if n.op_type == "Softmax" and len(n.input) == 1 and n.name not in skip_names
    ]
    if not candidates:
        return out

    taken_names = _all_names(graph)

    def _const(value: np.ndarray, tag: str) -> str:
        name = _unique_name(f"ibert_softmax_{tag}", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(value, name=name))
        return name

    quad_a_name = _const(np.asarray(_IBERT_SOFTMAX_QUAD_A, dtype=np.float32), "quad_a")
    quad_b_name = _const(np.asarray(_IBERT_SOFTMAX_QUAD_B, dtype=np.float32), "quad_b")
    one_name = _const(np.asarray(1.0, dtype=np.float32), "one")
    ln2_name = _const(np.asarray(_LN2, dtype=np.float32), "ln2")
    two_name = _const(np.asarray(2.0, dtype=np.float32), "two")

    for node in candidates:
        x_name = node.input[0]
        softmax_out = node.output[0]
        axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
        prefix = _unique_name(f"{softmax_out}_ibert_softmax", taken_names)

        axes_name = _const(np.asarray([axis], dtype=np.int64), f"{prefix}_axes")

        new_nodes = []

        def _op(op_type, inputs, tag, **attrs):
            out_name = _unique_name(f"{prefix}_{tag}", taken_names)
            n = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{tag}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n)
            return out_name

        row_max = _op("ReduceMax", [x_name, axes_name], "max", keepdims=1)
        shifted = _op("Sub", [x_name, row_max], "shifted")  # x - max <= 0
        neg_shifted = _op("Neg", [shifted], "neg_shifted")  # -(x - max) >= 0
        z = _op("Floor", [_op("Div", [neg_shifted, ln2_name], "z_raw")], "z")
        z_ln2 = _op("Mul", [z, ln2_name], "z_ln2")
        p = _op("Add", [shifted, z_ln2], "p")  # p in (-ln2, 0]

        p_sq = _op("Mul", [p, p], "p_sq")
        quad_term = _op("Mul", [p_sq, quad_a_name], "quad_term")
        lin_term = _op("Mul", [p, quad_b_name], "lin_term")
        sum_terms = _op("Add", [quad_term, lin_term], "sum_terms")
        exp_p = _op("Add", [sum_terms, one_name], "exp_p")  # polynomial exp(p)

        neg_z = _op("Neg", [z], "neg_z")
        pow2_neg_z = _op("Pow", [two_name, neg_z], "pow2_neg_z")  # 2**(-z)

        exp_x = _op("Mul", [exp_p, pow2_neg_z], "exp_x")  # approx exp(x - max)
        sum_exp = _op("ReduceSum", [exp_x, axes_name], "sum_exp", keepdims=1)

        result_node = onnx.helper.make_node(
            "Div",
            [exp_x, sum_exp],
            [softmax_out],
            name=_unique_name(f"{prefix}_result_node", taken_names),
        )
        new_nodes.append(result_node)

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in new_nodes:
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1
        graph.node.remove(node)

    return out
