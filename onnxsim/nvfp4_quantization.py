"""NVFP4 (NVIDIA, "Pretraining Large Language Models with NVFP4",
https://arxiv.org/abs/2509.25149, 2025; the same two-level scaling scheme
NVIDIA's own Model-Optimizer implements at
``modelopt/torch/quantization/qtensor/nvfp4_tensor.py`` and Transformer
Engine documents as its ``NVFP4`` recipe). onnxsim ports the *format's*
own definition, not any framework's fitting code -- the same rationale
:mod:`onnxsim.mx_quantization` and :mod:`onnxsim.if4_quantization` already
give for MXFP4/IF4 (a data representation, not an algorithm someone else's
reference implementation could diverge from).

Read :mod:`onnxsim.mx_quantization` first. NVFP4 shares MXFP4's exact 4-bit
element format -- E2M1, the same 16-value codebook (:data:`onnxsim.
mx_quantization.MXFP4_CODEBOOK`) -- but makes two different choices for the
**scale**:

- **Block size 16, not 32.** A finer grain than OCP MX's canonical choice
  (the same tradeoff :mod:`onnxsim.if4_quantization` already notes for its
  own block size).
- **The per-block scale is not a pure power of two.** MXFP4's E8M0 scale
  can only shrink or grow a block by an exact factor of 2, which wastes up
  to 41% of a block's own dynamic range (half a binade, on average) between
  E2M1's coarse codebook and the block's actual magnitude. NVFP4 instead
  stores each block's own scale as an **E4M3** FP8 value (1 sign, 4
  exponent, 3 mantissa bits; max representable magnitude ``448.0``) --
  three more bits than E8M0's mantissa-less exponent-only field buys a much
  closer fit, at the cost of the scale itself needing a real multiply
  instead of an exponent add.

E4M3 (like any float format) still has a bounded dynamic range of its own,
so NVFP4 adds a **second, per-tensor FP32 scale** (``global_scale``) that
the *whole tensor* shares, chosen so the largest block's own scale still
fits inside what E4M3 can represent. Given ``FLOAT8_E4M3_MAX = 448.0`` and
``FLOAT4_E2M1_MAX = 6.0`` (E2M1's own largest representable magnitude, same
constant as :mod:`onnxsim.mx_quantization`'s ``_MXFP4_MAX_MAGNITUDE``):

```
global_scale       = amax(tensor) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
raw_block_scale_i  = amax(block_i) / (global_scale * FLOAT4_E2M1_MAX)
block_scale_i      = round_to_nearest_e4m3(raw_block_scale_i)
dequant(code, i)   = E2M1_CODEBOOK[code] * block_scale_i * global_scale
```

(this is the formula NVIDIA's own Model-Optimizer and Transformer Engine
implementations use; verified against both independently of this repo).
Since ``amax(block_i) <= amax(tensor)``, ``raw_block_scale_i`` is always
``<= FLOAT8_E4M3_MAX``, so it never needs clamping before rounding.

Exactly as :mod:`onnxsim.mx_quantization` stores E8M0's *value* rather than
its raw 8-bit exponent field (ONNX has no E8M0 tensor type to store it in),
this module stores ``block_scale_i * global_scale`` -- already E4M3-rounded,
then combined with the per-tensor scale -- as one plain float32 value per
``(output channel, block)`` group, rather than the on-disk E4M3 byte plus a
separate FP32 scalar. Reconstruction is numerically identical to what a
real two-level NVFP4 dequantize produces (the E4M3 rounding step, which is
where this format's accuracy actually comes from relative to a naive
float32-scaled block, still happens during fitting); only the two-field
on-disk *bit* layout isn't reproduced, same simplification as
:mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`/:mod:`onnxsim.
if4_quantization` already make for their own formats.

Needs no calibration data: the block scale, the global scale, and the
codebook indices all come from the weight's own values.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import onnx

from onnxsim.mx_quantization import _MXFP4_MAX_MAGNITUDE
from onnxsim.onnx_simplifier import quantize_weight_only_nvfp4_cpp

# NVFP4's own reference block size: groups of 16, half OCP MX's canonical
# 32 -- see module docstring.
NVFP4_BLOCK_SIZE = 16

# E2M1's own largest representable magnitude (same value as
# onnxsim.mx_quantization's _MXFP4_MAX_MAGNITUDE, re-exported under NVFP4's
# own naming for readers coming from the NVFP4 literature).
FLOAT4_E2M1_MAX = _MXFP4_MAX_MAGNITUDE

# E4M3 (1 sign, 4 exponent, 3 mantissa bits, the "e4m3fn" variant with no
# infinities and a single NaN pattern) largest representable magnitude.
FLOAT8_E4M3_MAX = 448.0


def _e4m3_positive_grid() -> np.ndarray:
    """Every nonnegative magnitude the OCP E4M3 ("e4m3fn") format can
    represent: 1 sign bit (dropped -- magnitudes only), 4 exponent bits
    (bias 7), 3 mantissa bits. Exponent field ``0`` is subnormal; fields
    ``1..15`` are normal, except the single reserved NaN pattern
    (exponent ``15``, mantissa ``0b111``). 127 distinct magnitudes,
    0.0 to 448.0.
    """
    magnitudes = set()
    for exponent_field in range(16):
        for mantissa in range(8):
            if exponent_field == 15 and mantissa == 7:
                continue  # reserved NaN pattern (S.1111.111)
            if exponent_field == 0:
                magnitude = (mantissa / 8.0) * 2.0 ** (1 - 7)
            else:
                magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent_field - 7)
            magnitudes.add(magnitude)
    return np.asarray(sorted(magnitudes), dtype=np.float64)


_E4M3_POSITIVE_GRID = _e4m3_positive_grid()


def _round_to_e4m3(magnitudes: np.ndarray) -> np.ndarray:
    """Rounds nonnegative magnitudes to the nearest value E4M3 can
    represent, clamping at ``FLOAT8_E4M3_MAX``.
    """
    clamped = np.clip(magnitudes, 0.0, FLOAT8_E4M3_MAX)
    hi_idx = np.clip(
        np.searchsorted(_E4M3_POSITIVE_GRID, clamped), 0, len(_E4M3_POSITIVE_GRID) - 1
    )
    lo_idx = np.clip(hi_idx - 1, 0, len(_E4M3_POSITIVE_GRID) - 1)
    lo, hi = _E4M3_POSITIVE_GRID[lo_idx], _E4M3_POSITIVE_GRID[hi_idx]
    return np.where((hi - clamped) < (clamped - lo), hi, lo)


def quantize_weight_only_nvfp4(
    model: Union[str, onnx.ModelProto],
    block_size: int = NVFP4_BLOCK_SIZE,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into NVFP4 -- see this module's own docstring for the
    format. Needs no calibration data: the codebook, the per-block E4M3
    scale, and the per-tensor global scale all come from the weight's own
    values.

    Delegates to the verified C++ port
    (:func:`onnxsim.quantize_weight_only_nvfp4_cpp`), which hardcodes
    ``block_size=16`` and does not support ``skip_names`` (this repo's own
    established convention: a C++ port need not mirror every optional knob
    its Python counterpart has). Called with only default arguments, this
    function is fully backward compatible; a non-default ``block_size`` or
    a non-``None`` ``skip_names`` raises ``NotImplementedError`` rather
    than silently ignoring the request.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale group
            along the reduction dimension; must be 16 (the only value the
            delegated C++ implementation supports)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible -- not supported by the delegated C++
            implementation; must be ``None``
    :returns: ``model`` with every matched layer's weight replaced by its
            NVFP4 round-tripped float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if block_size != NVFP4_BLOCK_SIZE or skip_names is not None:
        raise NotImplementedError(
            "quantize_weight_only_nvfp4 now delegates to the C++ port "
            "(quantize_weight_only_nvfp4_cpp), which only supports the "
            f"default block_size={NVFP4_BLOCK_SIZE} and does not support "
            "skip_names; call quantize_weight_only_nvfp4_cpp directly if "
            "that's sufficient, or file an issue if you need these knobs "
            "back."
        )
    return quantize_weight_only_nvfp4_cpp(model)
