"""IF4 (Adaptive Block-Scaled Data Types, MIT-IBM/MIT-Han-Lab, 2026,
"Adaptive Block-Scaled Data Types", https://arxiv.org/abs/2603.28765, code
at https://github.com/mit-han-lab/fouroversix). onnxsim ports the *format's*
own definition, not any framework's code -- the same rationale
:mod:`onnxsim.mx_quantization` already gives for MXFP4 (a data
representation, not a fitting algorithm someone else's reference
implementation could diverge from).

Read :mod:`onnxsim.mx_quantization` first. NVFP4 (and this repo's own
MXFP4) share a real weakness the IF4 paper's own motivation calls out: a
block's shared scale is chosen so the block's own *largest*-magnitude
element just fits the 4-bit codebook's own max representable value -- for
E2M1 (MXFP4's codebook), that max value is ``6.0``, reached only by one
specific bit pattern, with the *next* representable value down at ``4.0``
-- a 33% relative gap right where a block's own biggest values land,
exactly the region every value in a heavy-tailed block clusters near.
Plain INT4's own uniform grid has the opposite problem: no gap near the
max, but *coarser* resolution than E2M1 gets for its *small*-magnitude
values (E2M1's ``{0, 0.5, 1, 1.5}`` near zero is denser than INT4's evenly
spaced grid). Neither format is uniformly better -- which one wins depends
on that specific block's own value distribution.

IF4's own fix: **decide per block, from the block's own data, which of the
two 4-bit formats (INT4 or E2M1/FP4) reconstructs it with lower error, and
use that one** -- both formats share the one scale field the same way OCP
MX's own spec already multiplexes multiple element formats onto one shared
E8M0/E4M3 scale, so there is no extra scale storage either way; only a
1-bit-per-block format selector is new (the paper's own hardware
implementation reuses the scale's own otherwise-unused sign bit for this,
since a shared block scale is always positive -- a specific bit-packing
trick this module does not reproduce, the same way :mod:`onnxsim.nf4`/
:mod:`onnxsim.mx_quantization` don't reproduce their own formats' packed
on-disk bit layouts either).

Needs no calibration data: the format choice, the codebook indices, and
the scale all come from the weight's own values, the same as
:mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`.

:func:`quantize_weight_only_if4` now delegates to the verified C++ port
(:func:`onnxsim.quantize_weight_only_if4_cpp`), which folds the
reconstruction directly into a replacement float32 initializer instead of
this module's own former ``Cast``/``Gather``/``Reshape``/``Mul`` graph
rewrite -- see ``onnxsim/passes/if4_quantization.h`` for the current
implementation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import onnx

from onnxsim.onnx_simplifier import quantize_weight_only_if4_cpp

# The paper's own reference block size (NVFP4's convention: groups of 16,
# smaller than OCP MX's own 32 -- a finer grain matches better since each
# block gets its own format choice too, not just its own scale).
IF4_BLOCK_SIZE = 16


def quantize_weight_only_if4(
    model: Union[str, onnx.ModelProto],
    block_size: int = IF4_BLOCK_SIZE,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) to 4 bits/element, choosing per block whichever of
    INT4 or FP4 (E2M1) reconstructs that block's own values with lower
    error -- see this module's own docstring for the technique. Needs no
    calibration data: everything comes from the weight's own values.

    Delegates to the verified C++ port
    (:func:`onnxsim.quantize_weight_only_if4_cpp`), which hardcodes
    ``block_size=16`` and does not support ``skip_names`` (this repo's own
    established convention: a C++ port need not mirror every optional knob
    its Python counterpart has). Called with only default arguments, this
    function is fully backward compatible; a non-default ``block_size`` or
    a non-``None`` ``skip_names`` raises ``NotImplementedError`` rather
    than silently ignoring the request.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale/format
            group along the reduction dimension; must be 16 (the only
            value the delegated C++ implementation supports)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible -- not supported by the delegated C++
            implementation; must be ``None``
    :returns: ``model`` with every matched layer's weight replaced by its
            IF4 round-tripped float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if block_size != IF4_BLOCK_SIZE or skip_names is not None:
        raise NotImplementedError(
            "quantize_weight_only_if4 now delegates to the C++ port "
            "(quantize_weight_only_if4_cpp), which only supports the "
            f"default block_size={IF4_BLOCK_SIZE} and does not support "
            "skip_names; call quantize_weight_only_if4_cpp directly if "
            "that's sufficient, or file an issue if you need these knobs "
            "back."
        )
    return quantize_weight_only_if4_cpp(model)
