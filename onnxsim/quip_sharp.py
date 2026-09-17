"""QuIP# (Tseng et al., 2024, "QuIP#: Even Better LLM Quantization with
Hadamard Incoherence and Lattice Codebooks", https://arxiv.org/abs/2402.04396),
building on QuIP's own incoherence-processing idea (Chee et al., 2023,
"QuIP: 2-Bit Quantization of Large Language Models With Guarantees",
https://arxiv.org/abs/2307.13304). onnxsim ports the algorithm, not any
framework's code, per the same rationale as :mod:`onnxsim.awq`/
:mod:`onnxsim.gptq`/:mod:`onnxsim.hqq` (QuIP#'s own reference implementation
quantizes live PyTorch weights with custom CUDA kernels, with no ONNX
export path).

QuIP# combines two ideas, both reproduced here in a form that is
independently verifiable rather than a byte-for-byte reproduction of the
paper's own (CUDA-kernel-optimized, partly proprietary) implementation --
see the two subsections below for exactly what differs and why.

**Incoherence processing.** Round-to-nearest quantization (what every other
onnxsim weight-only scheme does, uniform grid or not) struggles when a
weight matrix has a few directions of unusually large magnitude relative
to the rest -- the same "outlier" problem :mod:`onnxsim.llm_int8`/
:mod:`onnxsim.smoothquant` address for *activations*. QuIP's insight is
that conjugating the weight by a pair of random orthogonal matrices --
``Wtilde = V @ W @ U`` for random orthogonal ``U`` (applied along the
reduction/K dimension), ``V`` (applied along the output/N dimension) --
makes the *transformed* weight's entries look like i.i.d. Gaussian noise
with overwhelming probability, regardless of the original weight's own
structure (a concentration-of-measure argument: a fixed vector conjugated
by a uniformly random rotation is, with high probability, spread evenly
across all coordinates). A uniform quantization grid -- or, here, a fixed
lattice codebook -- fits Gaussian-like data far better than data with a
handful of outlier directions, with *no per-layer calibration needed to
find where the outliers are*: the randomization itself is what removes
them. Reconstructing the original weight is exact, since ``U``/``V`` are
orthogonal (``Ŵ = V.T @ Ŵtilde @ U.T``), and the extra rotations are
folded into the graph as two extra ``MatMul``s sandwiching the
quantized core -- see :func:`apply_quip_sharp`'s own docstring for the
exact node structure.

The real QuIP#/QuIP construct ``U``/``V`` as *randomized Hadamard
transforms* (a fixed, power-of-2-sized Hadamard matrix times a random
``+-1`` diagonal sign matrix) specifically so the rotation can be applied
in ``O(n log n)`` via the Fast Walsh-Hadamard Transform instead of a dense
``O(n^2)`` matrix multiply. This module uses a plain Haar-random
orthogonal matrix instead (via QR-decomposing a random Gaussian matrix,
with the sign of ``R``'s diagonal corrected so the result is uniformly
Haar-distributed rather than merely "some" orthogonal matrix) -- exactly
as effective for the concentration argument the incoherence processing
relies on (that argument only needs a uniformly random rotation, not a
Hadamard-structured one), simpler to construct correctly for any
dimension without power-of-2 padding, but applied via an ordinary dense
``MatMul`` rather than a butterfly network: a real deployment-efficiency
cost (``O(n^2)`` instead of ``O(n log n)`` per rotation) this module does
not attempt to avoid, matching the rest of onnxsim's PTQ modules'
consistent choice of graph simplicity/verifiability over peak deployment
efficiency (e.g. :mod:`onnxsim.nf4`'s plain ``Gather`` over a packed
bitstream).

**E8 lattice vector quantization.** Rather than quantizing each weight
element independently (a *scalar* quantizer, what every other onnxsim
INT4 scheme does), QuIP# jointly quantizes groups of 8 consecutive
(post-incoherence-processing) weight elements to the nearest point in the
**E8 lattice** -- the densest known sphere packing in 8 dimensions, so
quantizing onto it wastes less of the available precision than 8
independent scalar roundings would. E8 is the union of the "checkerboard"
lattice ``D8`` (integer points whose coordinate sum is even) and its
half-integer coset ``D8 + (1/2,...,1/2)``; nearest-point decoding is the
classical fast algorithm of Conway & Sloane ("Fast quantizing and
decoding algorithms for lattice quantizers and codes", 1982): round each
candidate lattice's coordinates to their own grid, and if that lands on
the wrong (odd) coordinate-sum parity, flip the single coordinate whose
rounding was least confident (largest residual) to fix it at minimum
cost -- then take whichever of the two candidates (integer or
half-integer coset) is actually closest. Implemented here exactly per
that classical algorithm (:func:`_closest_point_e8`), independently
verifiable (see this module's own tests: brute-force distance checks
against nearby candidates, and exact recovery of points already on the
lattice).

QuIP#'s own codebook (called "E8P") is a specific, *finite* subset of E8
-- about 2^16 points, curated with extra symmetry so a codeword can be
looked up/applied via a compact, hardware-friendly index (their own
paper's main engineering contribution, alongside the incoherence
processing) -- achieving roughly 2 bits/weight. This module does not
reproduce that curated subset (its exact construction is tied to
specific hardware-kernel engineering choices that are hard to verify
independently without the paper's own code); instead it stores each
lattice coordinate directly, doubled (E8 points are always all-integer or
all-half-integer, so doubling always yields an exact integer) and clipped
to a signed 4-bit range, alongside one float32 scale per 8-element group
-- the same "codes + scale" INT4 representation every other onnxsim
weight-only scheme already uses (down to reusing
:mod:`onnxsim.adaround`'s own ``_pack_int4``), just decoded with a
different (still closed-form, no codebook lookup needed) arithmetic
formula. This trades away QuIP#'s own ~2-bit/weight rate for a simpler,
verifiable representation at roughly INT4 rate -- the incoherence
processing and the E8 lattice quantization itself (the two ideas this
module sets out to port) are both faithfully reproduced either way.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import onnx

from onnxsim.onnx_simplifier import apply_quip_sharp_cpp


def _random_orthogonal_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random (uniformly distributed) ``n x n`` orthogonal matrix,
    via QR-decomposing an i.i.d. standard-Gaussian matrix and correcting
    ``Q``'s sign using ``R``'s own diagonal -- the standard construction
    (a plain QR's ``Q`` is orthogonal but not uniformly distributed over
    the orthogonal group without this correction).
    """
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    return q * d[np.newaxis, :]


def _closest_point_d8(v: np.ndarray) -> np.ndarray:
    """Nearest point in ``D8`` (integer vectors with an even coordinate
    sum) to each row of ``v`` (``[..., 8]``), via Conway & Sloane's fast
    algorithm: round to the nearest integers, then if the sum is odd, flip
    the least-confidently-rounded coordinate (largest residual) to the
    other side.
    """
    f = np.round(v)
    delta = v - f
    flat_delta = delta.reshape(-1, 8)
    flat_f = f.reshape(-1, 8)
    odd = (np.sum(flat_f, axis=-1).astype(np.int64) % 2) != 0
    idx = np.argmax(np.abs(flat_delta), axis=-1)
    rows = np.arange(flat_f.shape[0])
    adjustment = np.sign(flat_delta[rows, idx])
    adjustment = np.where(adjustment == 0, 1.0, adjustment)
    flat_f_adjusted = flat_f.copy()
    flat_f_adjusted[rows, idx] += adjustment
    result = np.where(odd[:, np.newaxis], flat_f_adjusted, flat_f)
    return result.reshape(v.shape)


def _closest_point_e8(v: np.ndarray) -> np.ndarray:
    """Nearest point in the E8 lattice (``D8`` union its half-integer
    coset) to each row of ``v`` (``[..., 8]``): the closer of the two
    candidates :func:`_closest_point_d8` gives for ``v`` itself and for
    ``v`` shifted onto the coset.
    """
    g0 = _closest_point_d8(v)
    g1 = _closest_point_d8(v - 0.5) + 0.5
    d0 = np.sum((v - g0) ** 2, axis=-1, keepdims=True)
    d1 = np.sum((v - g1) ** 2, axis=-1, keepdims=True)
    return np.where(d0 <= d1, g0, g1)


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


def apply_quip_sharp(
    model: Union[str, onnx.ModelProto],
    seed: int = 0,
    epsilon: float = 1e-8,
) -> onnx.ModelProto:
    """Applies QuIP#-style incoherence processing plus E8 lattice
    quantization to every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight whose reduction dimension ``K`` is divisible by 8. See
    this module's own docstring for the technique. Needs no calibration
    data: both the random rotation and the per-group scale come from the
    weight's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param seed: seed for the random orthogonal matrices (a fresh
            ``numpy.random.Generator`` is derived per matched layer, in
            graph node order, so results are deterministic and
            reproducible for a given model and seed)
    :param epsilon: floor applied to a group's own RMS magnitude before
            using it as a scale, avoiding a divide-by-zero on an all-zero
            group
    :returns: ``model`` with every matched layer's weight replaced by its
            QuIP#-reconstructed float32 version (see the "Delegates to"
            note below for a storage-format caveat); layers with a
            non-constant, non-2-D weight, or a reduction dimension not
            divisible by 8, are left untouched.

    Delegates to :func:`onnxsim.apply_quip_sharp_cpp`, which hardcodes
    ``seed=0``/``epsilon=1e-8`` (this function's own defaults) and cannot
    honor other values -- a non-default value raises ``ValueError`` rather
    than being silently ignored. Unlike this function's own former
    implementation (which kept ``U``, ``V`` and the packed INT4 lattice
    codes as explicit initializers and new ``MatMul`` nodes), the C++ port
    folds the entire rotate/quantize/rotate-back sandwich into a single
    replacement weight initializer -- exact, not an approximation, since
    ``U``/``V`` are square and only sandwich the weight, but a genuine
    storage-format difference from the old graph shape. The C++ port's own
    random orthogonal matrices are also generated via a different
    (equally Haar-uniform) construction, so the specific rotation applied
    -- though not the technique's own correctness -- differs from what
    this function used to produce for the same ``seed``.
    """
    if seed != 0 or epsilon != 1e-8:
        raise ValueError(
            "apply_quip_sharp now delegates to apply_quip_sharp_cpp, which "
            "hardcodes seed=0, epsilon=1e-8 and cannot honor other values"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_quip_sharp_cpp(model)
