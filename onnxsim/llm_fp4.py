"""LLM-FP4 (Liu, Yuan, Yang, Cheng, Yang, Liu, Zhu and Xu, 2023, EMNLP,
"LLM-FP4: 4-Bit Floating-Point Quantized Transformers",
https://arxiv.org/abs/2310.16836). onnxsim ports the algorithm, not any
framework's code, per the same rationale as :mod:`onnxsim.smoothquant`/
:mod:`onnxsim.zeroquant`.

**Relationship to onnxsim's other 4-bit floating-point modules.** onnxsim
already has two other 4-bit *floating-point* codebook formats:
:mod:`onnxsim.mx_quantization` (MXFP4: fixed E2M1 elements, per-block scale
restricted to a **power of two**, per the OCP MX spec) and :mod:`onnxsim.nf4`
(NormalFloat4: a fixed, data-*independent* 16-value codebook fit to a
standard normal distribution's own quantile points, not a sign/exponent/
mantissa format at all). LLM-FP4 is neither: it is a **standard**
sign/exponent/mantissa FP4 format (like MXFP4's own E2M1), but

1. its per-block scale is an ordinary **real-valued** float, not restricted
   to a power of two (MXFP4's own restriction), and
2. it does not fix the exponent/mantissa bit split at E2M1 -- it **searches**
   a small set of splits (this module: E1M2, E2M1, E3M0 -- every way to
   divide FP4's 3 non-sign bits between exponent and mantissa) per tensor,
   picking whichever minimizes reconstruction error, rather than using one
   format for every tensor unconditionally.

Both properties come from the paper's own "pre-shifted exponent bias"
framing: for a *floating-point* quantizer (unlike an affine INT4 quantizer),
the per-block scale and the format's own exponent bias are two names for the
same real-valued degree of freedom -- multiplying a block by ``2^d`` before
quantizing to a fixed-bias FP4 codebook is identical to quantizing the
unscaled block against a codebook whose bias has been shifted by ``d``. This
module realizes that freedom directly as a per-block real-valued scale (the
same representation :mod:`onnxsim.nf4` already uses for its own non-power-
of-two scale), searched, together with the bit-split choice, by grid search
against direct reconstruction MSE -- the same "hold everything else fixed,
scan a small set of candidates, keep whichever minimizes a direct error
metric" shape :func:`onnxsim.calibration._mse_threshold`/:func:`onnxsim.
calibration._entropy_threshold` already use for INT8 range calibration
(``_mse_threshold`` scans candidate clip thresholds against *direct*
reconstruction MSE -- the closer analogue to this module's own objective;
``_entropy_threshold`` scans the same kind of candidates against a
histogram-based KL divergence instead. This module scans candidate
*(format, clip ratio)* pairs against MSE -- a different, two-dimensional
search space, same "grid search over candidates" shape).

    Before:
      Y = MatMul(X, W) [+ bias]      -- W constant, [K, N], float32

    After:
      Codebook: initializer, float32 [16]  -- winning format's 16 fixed
                                               values (shared across every
                                               layer that picks this format)
      Wq: initializer, uint8 [K, N]        -- codebook index per element
      Ws: initializer, float32 [K/block_size, N]  -- real-valued (not
                                               power-of-two) scale per block
      What_hat = Reshape(Mul(Reshape(Gather(Codebook, Wq)), Ws), ...)
      Y = MatMul(X, What_hat) [+ bias]

**Deliberately not ported: activation quantization (W4A4) via cross-layer
exponent-bias migration.** The paper's other headline contribution is
"pre-shifted exponent bias" applied to *activations*: transformer
activations carry per-channel outliers (the same phenomenon
:mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression` address for
INT8), so the paper computes a per-channel real-valued scale from
calibration data and migrates it algebraically into the preceding weight or
LayerNormalization (exactly :mod:`onnxsim.smoothquant`'s and
:mod:`onnxsim.outlier_suppression`'s own migration, with FP4's real-valued
scale-as-bias standing in for those modules' INT8 quantization range) so
that, after migration, a *single* shared exponent bias suffices for a
straightforward per-tensor FP4 activation quantizer at graph-run time. That
migration machinery already exists in this repo (:func:`onnxsim.
apply_smoothquant`, :func:`onnxsim.apply_outlier_suppression`) and composes
with this module unchanged: run either migration pass first, then quantize
the migrated model's activations with a per-tensor FP4 QDQ-style insertion.
Building that activation-side QDQ insertion itself -- the graph-run-time
"quantize X to FP4, matmul against Wq" pipeline, analogous to
:mod:`onnxsim.zeroquant`'s own runtime activation quantization but emitting
FP4 codes instead of INT8 -- is real, non-trivial additional scope (a new
runtime dequantization/quantization subgraph, not a data-flow migration),
and is not implemented here. This module covers weight-only W4 quantization
with the paper's own format-and-bias search as its differentiator from
:mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`; W4A4 is a legitimate
follow-up, not attempted in this module.

ONNX has no native FP4 tensor type (as of this writing), so -- following the
exact same approach :mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4` already
use for their own codebook formats -- this module builds the dequantization
out of ordinary ONNX ops any opset-11+ runtime already supports: ``Gather``
the per-element code out of a 16-entry constant codebook, then ``Mul`` by
the per-block real-valued scale.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name

# Every way to split FP4's 3 non-sign bits between exponent and mantissa --
# the paper's own candidate set for its per-tensor format search. Named
# "eXmY" for X exponent bits, Y mantissa bits (X + Y == 3 always, since FP4
# is 1 sign bit + 3 remaining bits).
FP4_FORMATS: Dict[str, Tuple[int, int]] = {
    "e1m2": (1, 2),
    "e2m1": (2, 1),  # MXFP4's own element format (onnxsim.mx_quantization)
    "e3m0": (3, 0),
}


def _fp4_magnitudes(e_bits: int, m_bits: int) -> List[float]:
    """The 8 non-negative magnitudes an ``e_bits``-exponent/``m_bits``-
    mantissa 4-bit float evaluates to, per the standard IEEE-754-style
    definition (bias ``2^(e_bits-1) - 1``, exponent field 0 == subnormal).
    Ascending, starting at ``0.0``. ``e_bits + m_bits`` must be 3 (FP4's 3
    non-sign bits).
    """
    assert e_bits + m_bits == 3, "FP4 has 3 non-sign bits to split"
    bias = (1 << (e_bits - 1)) - 1 if e_bits > 0 else 0
    magnitudes = set()
    for exp_field in range(1 << e_bits):
        for mant_field in range(1 << m_bits):
            frac = mant_field / float(1 << m_bits)
            if exp_field == 0:
                value = frac * (2.0 ** (1 - bias))  # subnormal
            else:
                value = (1.0 + frac) * (2.0 ** (exp_field - bias))  # normal
            magnitudes.add(value)
    return sorted(magnitudes)


def _fp4_codebook(e_bits: int, m_bits: int) -> List[float]:
    """The full 16 signed codes for an ``e_bits``/``m_bits`` FP4 format:
    ``[-max, ..., -0.0, 0.0, ..., max]`` -- the same negatives-then-
    positives-with-a-duplicate-zero layout :mod:`onnxsim.mx_quantization`'s
    own ``MXFP4_CODEBOOK`` and :mod:`onnxsim.nf4`'s own ``NF4_CODEBOOK``
    use, so ``_nearest_codebook_index``'s indexing convention matches
    theirs.
    """
    magnitudes = _fp4_magnitudes(e_bits, m_bits)  # 8 ascending, [0] == 0.0
    negatives = [-m for m in reversed(magnitudes)]  # -max ... -0.0
    return negatives + magnitudes  # 16: -max...-0.0, 0.0...max


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


def _search_llm_fp4_blockwise(
    w_nk: np.ndarray,
    block_size: int,
    formats: Sequence[str],
    clip_ratios: np.ndarray,
) -> "tuple[str, np.ndarray, np.ndarray]":
    """Searches, for ``w_nk`` ([N, K], output channel first), the ``formats``
    x ``clip_ratios`` grid that minimizes total reconstruction MSE, per this
    module's own docstring. Returns ``(best_format, codes_nk, scale_blocks)``:
    codebook indices in ``[0, 15]`` (shape ``[N, K]``) and one real-valued
    scale per ``(output channel, block-of-K)`` group (shape
    ``[N, K // block_size]``) for the winning format. Assumes
    ``K % block_size == 0`` and ``formats``/``clip_ratios`` non-empty.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    max_abs = np.maximum(np.abs(blocks).max(axis=2), 1e-30)  # [N, num_blocks]

    best_format = formats[0]
    best_total_error = np.inf
    best_codes = np.zeros((n, num_blocks, block_size), dtype=np.uint8)
    best_scale = np.zeros((n, num_blocks))

    for fmt in formats:
        e_bits, m_bits = FP4_FORMATS[fmt]
        codebook = np.asarray(_fp4_codebook(e_bits, m_bits), dtype=np.float64)
        max_mag = codebook[-1]

        fmt_best_error = np.full((n, num_blocks), np.inf)
        fmt_best_scale = np.zeros((n, num_blocks))
        fmt_best_codes = np.zeros((n, num_blocks, block_size), dtype=np.int64)

        for r in clip_ratios:
            # The "pre-shifted exponent bias" search, realized as a
            # real-valued per-block scale: r < 1 clips outliers harder but
            # sharpens resolution for the bulk of the block, exactly the
            # clip-vs-resolution trade _mse_threshold's own cutoff search
            # makes for INT8 ranges.
            scale = np.maximum(max_abs * r / max_mag, 1e-30)  # [N, num_blocks]
            normalized = blocks / scale[:, :, np.newaxis]
            diffs = np.abs(normalized[..., np.newaxis] - codebook)
            codes = np.argmin(diffs, axis=-1)  # [N, num_blocks, block_size]
            dequant_normalized = codebook[codes]
            error = (
                np.sum((dequant_normalized - normalized) ** 2, axis=-1) * scale**2
            )  # [N, num_blocks], in the original (unnormalized) units

            improved = error < fmt_best_error
            fmt_best_error = np.where(improved, error, fmt_best_error)
            fmt_best_scale = np.where(improved, scale, fmt_best_scale)
            fmt_best_codes = np.where(improved[:, :, np.newaxis], codes, fmt_best_codes)

        total_error = float(fmt_best_error.sum())
        if total_error < best_total_error:
            best_total_error = total_error
            best_format = fmt
            best_codes = fmt_best_codes
            best_scale = fmt_best_scale

    return best_format, best_codes.astype(np.uint8).reshape(n, k), best_scale


def quantize_weight_only_llm_fp4(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    formats: Sequence[str] = ("e1m2", "e2m1", "e3m0"),
    num_scale_candidates: int = 17,
    min_clip_ratio: float = 0.5,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D float32
    weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into LLM-FP4's weight format -- see this module's own
    docstring for the technique and its scope (weight-only; W4A4 activation
    quantization is out of scope, but composes with this repo's existing
    :func:`onnxsim.apply_smoothquant`/:func:`onnxsim.apply_outlier_suppression`
    migrations, run first). Needs no calibration data: both the per-tensor
    format choice and the per-block scale are fit directly to each weight's
    own values, by exhaustive grid search minimizing reconstruction MSE.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale group
            along the reduction dimension
    :param formats: candidate FP4 exponent/mantissa bit splits to search per
            tensor -- keys into :data:`FP4_FORMATS`. The paper's own
            ablation set (every way to split FP4's 3 non-sign bits) is the
            default; a subset restricts (and speeds up) the search
    :param num_scale_candidates: number of per-block clip-ratio candidates
            to grid-search (evenly spaced over ``[min_clip_ratio, 1.0]``)
            for each format; more candidates costs more search time for a
            finer-grained scale
    :param min_clip_ratio: lower end of the per-block clip-ratio search
            range -- ``1.0`` keeps the block's own max-abs element exactly
            at the format's largest representable magnitude (no clipping);
            values below ``1.0`` let the search trade a harder clip on
            outliers for sharper resolution on the rest of the block
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
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
    skip_names = set(skip_names) if skip_names is not None else frozenset()
    formats = list(formats)
    if not formats or any(fmt not in FP4_FORMATS for fmt in formats):
        raise ValueError(f"formats must be a non-empty subset of {sorted(FP4_FORMATS)}")
    clip_ratios = np.linspace(min_clip_ratio, 1.0, max(num_scale_candidates, 1))

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    codebook_names: Dict[str, str] = {}  # format -> initializer name, created lazily

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

        fmt, codes_nk, scale_blocks = _search_llm_fp4_blockwise(
            w_nk, block_size, formats, clip_ratios
        )
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        if fmt not in codebook_names:
            e_bits, m_bits = FP4_FORMATS[fmt]
            codebook_names[fmt] = _unique_name(f"llm_fp4_codebook_{fmt}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(_fp4_codebook(e_bits, m_bits), dtype=np.float32),
                    name=codebook_names[fmt],
                )
            )
        codebook_name = codebook_names[fmt]
        num_blocks = k // block_size

        wq = onnx.numpy_helper.from_array(
            codes_orig.astype(np.uint8),
            name=_unique_name(f"{w_name}_llmfp4_q", taken_names),
        )
        graph.initializer.append(wq)
        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_llmfp4_scale", taken_names),
        )
        graph.initializer.append(ws)

        if weight_transposed:
            blocked_shape = [n, num_blocks, block_size]
            scale_shape = [n, num_blocks, 1]
        else:
            blocked_shape = [num_blocks, block_size, n]
            scale_shape = [num_blocks, 1, n]

        cast_out = _unique_name(f"{w_name}_llmfp4_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [wq.name], [cast_out], to=onnx.TensorProto.INT64
        )

        gather_out = _unique_name(f"{w_name}_llmfp4_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather", [codebook_name, cast_out], [gather_out], axis=0
        )

        blocked_shape_name = _unique_name(f"{w_name}_llmfp4_blocked_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(blocked_shape, dtype=np.int64), name=blocked_shape_name
            )
        )
        reshaped_out = _unique_name(f"{w_name}_llmfp4_reshaped", taken_names)
        reshape1_node = onnx.helper.make_node(
            "Reshape", [gather_out, blocked_shape_name], [reshaped_out]
        )

        scale_shape_name = _unique_name(f"{w_name}_llmfp4_scale_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(scale_shape, dtype=np.int64), name=scale_shape_name
            )
        )
        scale_reshaped_out = _unique_name(
            f"{w_name}_llmfp4_scale_reshaped", taken_names
        )
        reshape2_node = onnx.helper.make_node(
            "Reshape", [ws.name, scale_shape_name], [scale_reshaped_out]
        )

        scaled_out = _unique_name(f"{w_name}_llmfp4_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul", [reshaped_out, scale_reshaped_out], [scaled_out]
        )

        orig_shape_name = _unique_name(f"{w_name}_llmfp4_orig_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([dim0, dim1], dtype=np.int64), name=orig_shape_name
            )
        )
        dq_out = _unique_name(f"{w_name}_llmfp4_dq", taken_names)
        reshape3_node = onnx.helper.make_node(
            "Reshape",
            [scaled_out, orig_shape_name],
            [dq_out],
            name=_unique_name(f"{w_name}_llmfp4_dequant", taken_names),
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
