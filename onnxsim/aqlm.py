"""AQLM (Egiazarian et al., 2024, "Extreme Compression of Large Language
Models via Additive Quantization", https://arxiv.org/abs/2401.06118).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.hqq`
(AQLM's own reference implementation quantizes live PyTorch weights via a
custom calibration-aware beam-search optimizer, with no ONNX export path).

Every other codebook-based scheme in onnxsim (:mod:`onnxsim.nf4`'s one
*fixed* global codebook, :mod:`onnxsim.squeezellm`'s one *fit-per-group*
codebook) represents each weight element (NF4) or each small group of
elements (SqueezeLLM) as a single lookup into a single codebook. AQLM's
own idea, **additive** (a.k.a. residual) **quantization**: represent each
group instead as the *sum* of ``M`` lookups, one from each of ``M``
separate codebooks shared across every group in the layer:

    ŵ_group = C_1[i_1] + C_2[i_2] + ... + C_M[i_M]

With ``M`` codebooks of ``codebook_size`` entries each, this can represent
up to ``codebook_size ** M`` distinct group values while storing only
``M`` small codebooks plus ``M`` per-group indices -- far more
representational richness per stored bit than a single codebook of the
same total size, the same reason residual/product quantization is a
classical, well-established technique in the vector quantization
literature AQLM's own paper builds on and cites.

Fitting the ``M`` codebooks: this module uses the classical **greedy
residual k-means** strategy -- fit codebook 1 with ordinary k-means
(Lloyd's algorithm) to every group's own raw values (each of the many
groups in a layer treated as one point in ``group_dim``-dimensional
space, all sharing the *same* fitted codebook, unlike
:mod:`onnxsim.squeezellm`'s independent per-group codebooks), subtract
what it reconstructs, fit codebook 2 to *that residual*, and so on --
rather than AQLM's own more sophisticated joint beam-search code
assignment calibrated against a Hessian-weighted objective. Greedy
residual fitting is the textbook baseline additive/residual quantization
is built on, independently verifiable (see this module's own tests: more
codebooks can only reduce reconstruction error, since each new codebook
targets exactly the error the previous stages left over), and needs no
calibration data at all -- consistent with :mod:`onnxsim.hqq`/
:mod:`onnxsim.nf4`/:mod:`onnxsim.quip_sharp`'s own choice to solve the
same representational problem via a classical technique rather than risk
an unverifiable reproduction of a paper's own bespoke calibrated
optimizer.

Dequantization is ``M`` ordinary ``Gather`` operations (one per codebook,
exactly :mod:`onnxsim.nf4`'s own pattern, since every group shares the
same global codebook per stage -- no per-group indexing like
:mod:`onnxsim.squeezellm`'s ``GatherND`` is needed) followed by ``M - 1``
``Add`` nodes summing the stages together. No custom op or contrib
domain, and no opset requirement beyond ordinary ``Gather``/``Add``.
"""

from __future__ import annotations

from typing import Union

import onnx

from onnxsim.onnx_simplifier import apply_aqlm_cpp


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


def quantize_weight_only_aqlm(
    model: Union[str, onnx.ModelProto],
    group_dim: int = 8,
    num_codebooks: int = 2,
    codebook_size: int = 256,
    num_iterations: int = 10,
    seed: int = 0,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``group_dim``) into AQLM-style additive (multi-codebook) quantization
    -- see this module's own docstring for the technique. Needs no
    calibration data: every codebook is fit directly to the weight's own
    values.

    Delegates to :func:`onnxsim.apply_aqlm_cpp`, which hardcodes
    ``group_dim=8``/``num_codebooks=2``/``codebook_size=256``/
    ``num_iterations=10`` and uses a deterministic, magnitude-sorted
    codebook initialization rather than this function's own seeded random
    sample (an ACCEPTED, PERMANENT DIVERGENCE already documented on the
    C++ port itself -- both are independently-correct fits, not bit-for-bit
    identical). A non-default ``group_dim``/``num_codebooks``/
    ``codebook_size``/``num_iterations`` cannot be honored by the C++
    implementation and raises ``ValueError`` rather than silently ignoring
    it; ``seed`` is accepted for backward compatibility but has no effect
    (the C++ port's own initialization takes no seed at all).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param group_dim: must be ``8`` (the only value the delegated C++
            implementation supports)
    :param num_codebooks: must be ``2``
    :param codebook_size: must be ``256``
    :param num_iterations: must be ``10``
    :param seed: unused (kept for backward compatibility)
    :returns: ``model`` with every matched layer's weight replaced by its
            AQLM-reconstructed float32 version; layers with a
            non-constant, non-2-D, or non-group-divisible weight are left
            untouched
    """
    if (
        group_dim != 8
        or num_codebooks != 2
        or codebook_size != 256
        or num_iterations != 10
    ):
        raise ValueError(
            "quantize_weight_only_aqlm now delegates to apply_aqlm_cpp, which "
            "hardcodes group_dim=8, num_codebooks=2, codebook_size=256, "
            "num_iterations=10 and cannot honor other values"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_aqlm_cpp(model)
