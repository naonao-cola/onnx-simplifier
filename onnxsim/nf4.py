"""NF4 -- bitsandbytes' NormalFloat 4-bit quantization (Dettmers et al.,
2023, "QLoRA: Efficient Finetuning of Quantized LLMs",
https://arxiv.org/abs/2305.14314). One of the concrete, well-specified
quantization formats behind the tools Unsloth's own quantization paths
delegate to (bitsandbytes' 4-bit loading), rather than anything Unsloth
contributes itself -- see ``onnxsim/hqq.py`` and ``onnxsim/awq.py``'s own
docstrings for the earlier research establishing that Unsloth's own
"Dynamic Quantization" documentation does not match its actual (fixed,
non-data-dependent) open-source implementation. NF4's format, by contrast,
is exact and stable: a fixed 16-value codebook, published verbatim in
bitsandbytes' own source (``bitsandbytes/functional.py``) and unchanged
across releases since its introduction.

Unlike every other onnxsim INT4 scheme (``quantize_weight_only_int4`` and
everything built on it -- :mod:`onnxsim.adaround`, :mod:`onnxsim.awq`,
:mod:`onnxsim.gptq`, :mod:`onnxsim.autoround` -- plus :mod:`onnxsim.hqq`'s
own affine codes), NF4 does not quantize onto a *uniform* integer grid at
all. Neural network weights are typically close to normally distributed,
and a uniform grid wastes most of its resolution on the tails of that
distribution while under-resolving the dense region near zero where most
of the weight mass actually sits. NF4's codebook is instead the
(zero-symmetric, exact-zero-including) quantile points of a standard normal
distribution -- 16 fixed values, denser near zero, identical for every
tensor and every block, with no data-dependent fitting at all (simpler even
than :mod:`onnxsim.hqq`'s calibration-free-but-iterative zero-point fit:
NF4 needs only a single per-block scale, computed once, no iteration).
Quantizing a weight is then just: divide by that block's own max-abs value
(bringing it into the codebook's own ``[-1, 1]`` range) and snap to the
nearest of the 16 fixed codebook values.

This has no standard ONNX representation the way an affine
``DequantizeLinear`` does (ONNX has no non-uniform-codebook quantization
op), so this module builds the dequantization directly out of ordinary
ONNX ops any opset-11+ runtime already supports -- no contrib op, no
opset-21 features: ``Gather`` the (per-element) 4-bit code out of a
16-entry constant codebook tensor, then ``Mul`` by the per-block scale
(broadcast via ``Reshape``, exposing the block dimension so ordinary numpy-
style broadcasting handles the rest). Codes are stored one byte per element
(``[0, 15]``, unpacked) rather than 2-per-byte packed -- packing would need
either non-standard low-bit ``Gather`` indices (unsupported) or extra
bit-unpacking ops in the graph itself, adding real complexity for a
storage saving orthogonal to NF4's actual point (quantization *accuracy*
from a distribution-matched codebook, not maximal file size); a packed
variant is a possible future addition, not attempted here.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name

# bitsandbytes' own NF4 codebook (`bitsandbytes/functional.py`'s
# `create_normal_map`/hardcoded NF4 table) -- the 16 quantile points of a
# standard normal distribution, adjusted to be symmetric and include an
# exact 0, published verbatim and unchanged across bitsandbytes releases.
NF4_CODEBOOK: List[float] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(w_name, weight_transposed)`` or ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[1], False
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
        return node.input[1], weight_transposed
    return None


def _nearest_codebook_index(normalized: np.ndarray) -> np.ndarray:
    codebook = np.asarray(NF4_CODEBOOK, dtype=np.float64)
    # [..., 16] absolute differences -> nearest index per element. Blocks
    # are small (tens of thousands of elements at most for any single
    # layer), so a dense (E, 16) distance matrix is cheap; no need for a
    # sorted-codebook binary search.
    diffs = np.abs(normalized[..., np.newaxis] - codebook)
    return np.argmin(diffs, axis=-1).astype(np.uint8)


def _quantize_nf4_blockwise(
    w_nk: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Returns ``(codes_nk, scale_blocks)`` for ``w_nk`` ([N, K], output
    channel first): NF4 codebook indices in ``[0, 15]`` and one absmax
    scale per ``(output channel, block-of-K)`` group, shape
    ``[N, K // block_size]``. Assumes ``K % block_size == 0``.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=2), 1e-12)  # [N, num_blocks]
    normalized = blocks / scale[:, :, np.newaxis]
    codes = _nearest_codebook_index(normalized)  # [N, num_blocks, block_size]
    return codes.reshape(n, k), scale


def quantize_weight_only_nf4(
    model: Union[str, onnx.ModelProto], block_size: int = 64
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into bitsandbytes' NF4 format -- see this module's own
    docstring for the technique. Needs no calibration data: every
    quantization decision comes from the weight tensor's own values
    against NF4's fixed codebook.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale group
            along the reduction dimension; bitsandbytes' own QLoRA default
            is 64 (versus 32 for onnxsim's uniform-grid INT4 schemes)
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Reshape(Gather(codebook, Cast(Wq, INT64)), ...), Ws) ->
            Reshape(..., original shape)`` feeding the original MatMul/Gemm
            node -- ordinary ONNX ops only, no contrib op and no minimum
            opset beyond what ``Gather``/``Cast``/``Reshape``/``Mul``
            themselves need (opset 11+). Layers with a non-constant,
            non-2-D, or non-block-divisible weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    codebook_name = None  # created lazily on first match -- see below

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        w_name, weight_transposed = match
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
            # nothing to quantize comes back byte-identical to the input,
            # matching every other onnxsim quantize_* function's own
            # no-op convention.
            codebook_name = _unique_name("nf4_codebook", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(NF4_CODEBOOK, dtype=np.float32), name=codebook_name
                )
            )
        num_blocks = k // block_size

        codes_nk, scale_blocks = _quantize_nf4_blockwise(w_nk, block_size)
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.numpy_helper.from_array(
            codes_orig.astype(np.uint8),
            name=_unique_name(f"{w_name}_nf4_q", taken_names),
        )
        graph.initializer.append(wq)
        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_nf4_scale", taken_names),
        )
        graph.initializer.append(ws)

        # Reshape target exposing the block dimension so an ordinary
        # (broadcasting) Mul against the per-block scale lands on the
        # right elements, matching this layer's own storage layout: K on
        # axis 1 when weight_transposed (scale shape [N, num_blocks]),
        # else K on axis 0 (scale shape [num_blocks, N]).
        if weight_transposed:
            blocked_shape = [n, num_blocks, block_size]
            scale_shape = [n, num_blocks, 1]
        else:
            blocked_shape = [num_blocks, block_size, n]
            scale_shape = [num_blocks, 1, n]

        cast_out = _unique_name(f"{w_name}_nf4_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [wq.name], [cast_out], to=onnx.TensorProto.INT64
        )

        gather_out = _unique_name(f"{w_name}_nf4_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather", [codebook_name, cast_out], [gather_out], axis=0
        )

        blocked_shape_name = _unique_name(f"{w_name}_nf4_blocked_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(blocked_shape, dtype=np.int64), name=blocked_shape_name
            )
        )
        reshaped_out = _unique_name(f"{w_name}_nf4_reshaped", taken_names)
        reshape1_node = onnx.helper.make_node(
            "Reshape", [gather_out, blocked_shape_name], [reshaped_out]
        )

        scale_shape_name = _unique_name(f"{w_name}_nf4_scale_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(scale_shape, dtype=np.int64), name=scale_shape_name
            )
        )
        scale_reshaped_out = _unique_name(f"{w_name}_nf4_scale_reshaped", taken_names)
        reshape2_node = onnx.helper.make_node(
            "Reshape", [ws.name, scale_shape_name], [scale_reshaped_out]
        )

        scaled_out = _unique_name(f"{w_name}_nf4_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul", [reshaped_out, scale_reshaped_out], [scaled_out]
        )

        orig_shape_name = _unique_name(f"{w_name}_nf4_orig_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([dim0, dim1], dtype=np.int64), name=orig_shape_name
            )
        )
        dq_out = _unique_name(f"{w_name}_nf4_dq", taken_names)
        reshape3_node = onnx.helper.make_node(
            "Reshape",
            [scaled_out, orig_shape_name],
            [dq_out],
            name=_unique_name(f"{w_name}_nf4_dequant", taken_names),
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
