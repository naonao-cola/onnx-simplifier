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
on-disk bit layouts either; this module stores the selector as its own
explicit per-block array instead).

This module reuses :mod:`onnxsim.mx_quantization`'s own E2M1 codebook
(``MXFP4_CODEBOOK``) for the FP4 candidate, and a plain signed 4-bit
integer grid (``-8..7``) for the INT4 candidate, tries both per block
(each with its own best-fit scale), and keeps whichever gives lower mean
squared reconstruction error on that block's own values. Both candidate
codebooks are concatenated into one 32-entry table (FP4 in slots
``0..15``, INT4 in slots ``16..31``) so dequantization is a single
``Gather`` into one combined table followed by the ordinary per-block
scale ``Mul`` -- exactly :mod:`onnxsim.mx_quantization`'s own graph shape,
just with a bigger table and the per-block index landing in whichever half
that block picked.

Needs no calibration data: the format choice, the codebook indices, and
the scale all come from the weight's own values, the same as
:mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.mx_quantization import MXFP4_CODEBOOK, _match_matmul_like

# The paper's own reference block size (NVFP4's convention: groups of 16,
# smaller than OCP MX's own 32 -- a finer grain matches better since each
# block gets its own format choice too, not just its own scale).
IF4_BLOCK_SIZE = 16

_FP4_MAX_MAGNITUDE = 6.0  # E2M1's own largest representable magnitude
_INT4_MAX_MAGNITUDE = 7.0  # symmetric range; code -8 is reachable but unused as "max"

# INT4 candidate codebook: plain signed 4-bit integers, two's-complement
# range -8..7 -- raw (unscaled) values, the same "codebook holds the raw
# per-format magnitude, scale normalizes a block's own max into it" shape
# onnxsim.mx_quantization's own FP4 codebook already uses.
_INT4_CODEBOOK: List[float] = [float(v) for v in range(-8, 8)]

# Combined 32-entry table: FP4 in [0, 16), INT4 in [16, 32).
_COMBINED_CODEBOOK: List[float] = list(MXFP4_CODEBOOK) + _INT4_CODEBOOK


def _nearest_index(normalized: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    diffs = np.abs(normalized[..., np.newaxis] - codebook)
    return np.argmin(diffs, axis=-1)


def _quantize_if4_blockwise(w_nk: np.ndarray, block_size: int):
    """Returns ``(codes_nk, scale_blocks)`` for ``w_nk`` ([N, K], output
    channel first): combined-table indices in ``[0, 32)`` and one scale
    per ``(output channel, block-of-K)`` group, shape
    ``[N, K // block_size]``. Assumes ``K % block_size == 0``.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    max_abs = np.maximum(np.abs(blocks).max(axis=2), 1e-30)  # [N, num_blocks]

    fp4_scale = max_abs / _FP4_MAX_MAGNITUDE
    fp4_normalized = blocks / fp4_scale[:, :, np.newaxis]
    fp4_codebook = np.asarray(MXFP4_CODEBOOK, dtype=np.float64)
    fp4_idx = _nearest_index(fp4_normalized, fp4_codebook)  # [N, num_blocks, bs]
    fp4_recon = fp4_codebook[fp4_idx] * fp4_scale[:, :, np.newaxis]
    fp4_mse = np.mean((blocks - fp4_recon) ** 2, axis=2)  # [N, num_blocks]

    int4_scale = max_abs / _INT4_MAX_MAGNITUDE
    int4_normalized = blocks / int4_scale[:, :, np.newaxis]
    int4_codebook = np.asarray(_INT4_CODEBOOK, dtype=np.float64)
    int4_idx = _nearest_index(int4_normalized, int4_codebook)
    int4_recon = int4_codebook[int4_idx] * int4_scale[:, :, np.newaxis]
    int4_mse = np.mean((blocks - int4_recon) ** 2, axis=2)

    use_int4 = int4_mse < fp4_mse  # [N, num_blocks]
    scale = np.where(use_int4, int4_scale, fp4_scale)
    codes = np.where(
        use_int4[:, :, np.newaxis], int4_idx + len(MXFP4_CODEBOOK), fp4_idx
    )
    return codes.reshape(n, k), scale


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

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale/format
            group along the reduction dimension -- the paper's own
            reference choice is 16 (NVFP4's own block size)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Reshape(Gather(combined_codebook, Cast(Wq, INT64)),
            ...), Ws) -> Reshape(..., original shape)`` feeding the
            original MatMul/Gemm node -- ordinary ONNX ops only, opset
            11+, no contrib op. Layers with a non-constant, non-2-D, or
            non-block-divisible weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    codebook_name = None  # created lazily on first match

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        w_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        if codebook_name is None:
            codebook_name = _unique_name("if4_codebook", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(_COMBINED_CODEBOOK, dtype=np.float32),
                    name=codebook_name,
                )
            )
        num_blocks = k // block_size

        codes_nk, scale_blocks = _quantize_if4_blockwise(w_nk, block_size)
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.numpy_helper.from_array(
            codes_orig.astype(np.uint8),
            name=_unique_name(f"{w_name}_if4_q", taken_names),
        )
        graph.initializer.append(wq)
        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_if4_scale", taken_names),
        )
        graph.initializer.append(ws)

        if weight_transposed:
            blocked_shape = [n, num_blocks, block_size]
            scale_shape = [n, num_blocks, 1]
        else:
            blocked_shape = [num_blocks, block_size, n]
            scale_shape = [num_blocks, 1, n]

        cast_out = _unique_name(f"{w_name}_if4_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [wq.name], [cast_out], to=onnx.TensorProto.INT64
        )

        gather_out = _unique_name(f"{w_name}_if4_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather", [codebook_name, cast_out], [gather_out], axis=0
        )

        blocked_shape_name = _unique_name(f"{w_name}_if4_blocked_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(blocked_shape, dtype=np.int64), name=blocked_shape_name
            )
        )
        reshaped_out = _unique_name(f"{w_name}_if4_reshaped", taken_names)
        reshape1_node = onnx.helper.make_node(
            "Reshape", [gather_out, blocked_shape_name], [reshaped_out]
        )

        scale_shape_name = _unique_name(f"{w_name}_if4_scale_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(scale_shape, dtype=np.int64), name=scale_shape_name
            )
        )
        scale_reshaped_out = _unique_name(f"{w_name}_if4_scale_reshaped", taken_names)
        reshape2_node = onnx.helper.make_node(
            "Reshape", [ws.name, scale_shape_name], [scale_reshaped_out]
        )

        scaled_out = _unique_name(f"{w_name}_if4_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul", [reshaped_out, scale_reshaped_out], [scaled_out]
        )

        orig_shape_name = _unique_name(f"{w_name}_if4_orig_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([dim0, dim1], dtype=np.int64), name=orig_shape_name
            )
        )
        dq_out = _unique_name(f"{w_name}_if4_dq", taken_names)
        reshape3_node = onnx.helper.make_node(
            "Reshape",
            [scaled_out, orig_shape_name],
            [dq_out],
            name=_unique_name(f"{w_name}_if4_dequant", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (
            cast_node,
            gather_node,
            reshape1_node,
            reshape2_node,
            mul_node,
            reshape3_node,
        ):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out
